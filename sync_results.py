"""
Live results sync via git (server -> GitHub -> laptop).

Run outputs are deliberately kept OFF main (repo hygiene). This script instead
ships them on a dedicated branch as ORPHAN commits, built through a throwaway
index (GIT_INDEX_FILE) — the repo's working tree, real index, and checked-out
branch are never touched, so it is safe to run beside a live training pool.
Each push force-replaces the branch tip: no history accumulates, and unchanged
files are never re-uploaded (same blobs).

What is synced: *.json, *.npz, *.log, *.txt, *.md under the roots
(probe_runs/, checkpoints/, parallel_logs/) — NEVER *.pt weights.

push mode — on the SERVER, beside the launcher (or let launch_parallel spawn
it via --sync_every):
    python sync_results.py --mode push --loop --interval 600
    python sync_results.py --mode push              # one shot (auth test)

pull mode — on the LAPTOP:
    python sync_results.py --mode pull
  Fetches the branch and restores the results paths into the working tree
  (paths are git-ignored locally; nothing is staged, local-only files that
  the server lacks are left alone).

Requires: `git push` authorized from the server clone (test with a one-shot
push before trusting the overnight loop).
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

ROOTS = ["probe_runs", "checkpoints", "parallel_logs"]
INCLUDE_EXT = (".json", ".npz", ".log", ".txt", ".md")

# Fixed identity so the sync never depends on the clone's git config.
_GIT_ENV_ID = {
    "GIT_AUTHOR_NAME":     "results-sync",
    "GIT_AUTHOR_EMAIL":    "results-sync@local",
    "GIT_COMMITTER_NAME":  "results-sync",
    "GIT_COMMITTER_EMAIL": "results-sync@local",
}


def _git(args, extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    r = subprocess.run(["git"] + args, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r.stdout.strip()


def collect_files(roots):
    files = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.endswith(INCLUDE_EXT):
                    files.append(os.path.join(dirpath, f))
    return sorted(files)


def _remote_tree(remote, branch):
    """Tree hash currently at the remote branch tip (None if unknown)."""
    try:
        out = _git(["ls-remote", remote, f"refs/heads/{branch}"])
        if not out:
            return None
        return _git(["rev-parse", out.split()[0] + "^{tree}"])
    except RuntimeError:
        return None       # branch object not local (pushed elsewhere) — fine


def push_once(remote, branch, roots, last_tree=None):
    """Build an orphan commit of the current results files; force-push it.
    Returns the tree hash (so callers can skip no-change cycles)."""
    if last_tree is None:
        last_tree = _remote_tree(remote, branch)
    files = collect_files(roots)
    if not files:
        print("[sync] nothing to sync yet")
        return last_tree

    git_dir = _git(["rev-parse", "--git-dir"])
    index = os.path.join(git_dir, "sync-index")
    if os.path.exists(index):
        os.remove(index)                       # fresh index → tree == disk now
    env = {"GIT_INDEX_FILE": index}

    # -f: results paths are git-ignored by design; NUL list avoids the
    # command-line length limit (hundreds of result files).
    with tempfile.NamedTemporaryFile("w", suffix=".lst", delete=False,
                                     newline="") as tf:
        tf.write("\0".join(files))
        lst = tf.name
    try:
        _git(["add", "-f", "--pathspec-from-file=" + lst,
              "--pathspec-file-nul"], env)
        tree = _git(["write-tree"], env)
    finally:
        os.remove(lst)

    if tree == last_tree:
        print(f"[sync] no changes ({len(files)} files)")
        return tree

    msg = f"results sync {time.strftime('%Y-%m-%d %H:%M:%S')} ({len(files)} files)"
    commit = _git(["commit-tree", tree, "-m", msg], {**env, **_GIT_ENV_ID})
    _git(["push", "--force", remote, f"{commit}:refs/heads/{branch}"])
    print(f"[sync] pushed {len(files)} files -> {remote}/{branch}  ({commit[:10]})")
    return tree


def pull_once(remote, branch, roots):
    _git(["fetch", remote, branch])
    restored = []
    for root in roots:
        try:
            _git(["restore", "--source", "FETCH_HEAD", "--worktree",
                  "--", root])
            restored.append(root)
        except RuntimeError:
            pass                                # root absent in synced tree
    print(f"[sync] restored from {remote}/{branch}: "
          f"{', '.join(restored) or '(nothing matched)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["push", "pull"], required=True)
    ap.add_argument("--remote",   type=str, default="origin")
    ap.add_argument("--branch",   type=str, default="results-live")
    ap.add_argument("--roots",    type=str, nargs="+", default=ROOTS)
    ap.add_argument("--loop",     action="store_true",
                    help="push mode: keep pushing every --interval seconds")
    ap.add_argument("--interval", type=int, default=600)
    cli = ap.parse_args()

    if cli.mode == "pull":
        pull_once(cli.remote, cli.branch, cli.roots)
        return

    last_tree = None
    while True:
        try:
            last_tree = push_once(cli.remote, cli.branch, cli.roots, last_tree)
        except RuntimeError as e:
            # never let a transient git/network failure kill the loop
            print(f"[sync] WARNING: {e}", file=sys.stderr)
        if not cli.loop:
            break
        time.sleep(cli.interval)


if __name__ == "__main__":
    main()
