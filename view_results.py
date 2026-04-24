"""
CLI utility to inspect saved training result JSON files.

Supports multiple detail levels so users can quickly switch between
summary-only and per-episode logs.

Examples
--------
python view_results.py checkpoints/ue10WithBL/il_results.json
python view_results.py checkpoints/ue10WithBL/il_results.json --detail brief
python view_results.py checkpoints/ue10WithBL/il_results.json --detail full --last 50
python view_results.py checkpoints/ue10WithBL/il_results.json --detail standard --window 25
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return None
        return xf
    except (TypeError, ValueError):
        return None


def _rolling_average(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = [v for v in values[lo : i + 1] if v is not None]
        out.append(mean(chunk) if chunk else None)
    return out


def _best_index(values: List[Optional[float]], mode: str = "min") -> Optional[int]:
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return None
    if mode == "min":
        return min(valid, key=lambda t: t[1])[0]
    return max(valid, key=lambda t: t[1])[0]


def _fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_history(history_raw: Any) -> List[Dict[str, Optional[float]]]:
    if not isinstance(history_raw, list):
        return []

    out: List[Dict[str, Optional[float]]] = []
    for row in history_raw:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "avg_cost": _safe_float(row.get("avg_cost")),
                "total_loss": _safe_float(row.get("total_loss")),
                "eps": _safe_float(row.get("eps")),
            }
        )
    return out


def _print_header(data: Dict[str, Any], src: Path):
    method = data.get("method", "unknown")
    eval_mean = _safe_float(data.get("eval_mean"))
    eval_std = _safe_float(data.get("eval_std"))
    best_train = _safe_float(data.get("best_train"))

    print("=" * 78)
    print("RESULT FILE")
    print("-" * 78)
    print(f"Path       : {src}")
    print(f"Method     : {method}")
    print(f"Eval mean  : {_fmt(eval_mean)}")
    print(f"Eval std   : {_fmt(eval_std)}")
    print(f"Best train : {_fmt(best_train)}")


def _print_config(data: Dict[str, Any]):
    cfg = data.get("config", {})
    if not isinstance(cfg, dict) or not cfg:
        print("Config     : n/a")
        return

    print("-" * 78)
    print("CONFIG")
    print("-" * 78)
    key_width = max(len(str(k)) for k in cfg.keys())
    for k in sorted(cfg.keys()):
        print(f"{str(k):<{key_width}} : {cfg[k]}")


def _build_summary(history: List[Dict[str, Optional[float]]], window: int) -> Dict[str, Any]:
    costs = [h["avg_cost"] for h in history]
    losses = [h["total_loss"] for h in history]
    eps = [h["eps"] for h in history]

    n = len(history)
    first_cost = costs[0] if n else None
    last_cost = costs[-1] if n else None
    first_loss = losses[0] if n else None
    last_loss = losses[-1] if n else None
    first_eps = eps[0] if n else None
    last_eps = eps[-1] if n else None

    best_cost_idx = _best_index(costs, mode="min")
    best_loss_idx = _best_index(losses, mode="min")

    cost_ma = _rolling_average(costs, max(1, window)) if n else []
    loss_ma = _rolling_average(losses, max(1, window)) if n else []

    return {
        "episodes": n,
        "first_cost": first_cost,
        "last_cost": last_cost,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "first_eps": first_eps,
        "last_eps": last_eps,
        "best_cost_idx": best_cost_idx,
        "best_loss_idx": best_loss_idx,
        "best_cost": costs[best_cost_idx] if best_cost_idx is not None else None,
        "best_loss": losses[best_loss_idx] if best_loss_idx is not None else None,
        "cost_ma_last": cost_ma[-1] if cost_ma else None,
        "loss_ma_last": loss_ma[-1] if loss_ma else None,
        "cost_drop": (
            ((first_cost - last_cost) / first_cost) * 100.0
            if first_cost not in (None, 0.0) and last_cost is not None
            else None
        ),
    }


def _print_summary(summary: Dict[str, Any], window: int):
    print("-" * 78)
    print("SUMMARY")
    print("-" * 78)
    print(f"Episodes                   : {summary['episodes']}")
    print(
        f"Avg cost (first -> last)   : {_fmt(summary['first_cost'])}"
        f" -> {_fmt(summary['last_cost'])}"
    )
    print(
        f"Loss (first -> last)       : {_fmt(summary['first_loss'])}"
        f" -> {_fmt(summary['last_loss'])}"
    )
    print(
        f"Epsilon (first -> last)    : {_fmt(summary['first_eps'])}"
        f" -> {_fmt(summary['last_eps'])}"
    )
    print(f"Best avg cost              : {_fmt(summary['best_cost'])}")
    if summary["best_cost_idx"] is not None:
        print(f"Best avg cost episode      : {summary['best_cost_idx'] + 1}")
    print(f"Best loss                  : {_fmt(summary['best_loss'])}")
    if summary["best_loss_idx"] is not None:
        print(f"Best loss episode          : {summary['best_loss_idx'] + 1}")
    print(f"Cost improvement (%)       : {_fmt(summary['cost_drop'], 2)}")
    print(
        f"Rolling avg cost (w={window}) : {_fmt(summary['cost_ma_last'])}"
    )
    print(
        f"Rolling avg loss (w={window}) : {_fmt(summary['loss_ma_last'])}"
    )


def _slice_rows(history: List[Dict[str, Optional[float]]], last: Optional[int]) -> Tuple[int, List[Dict[str, Optional[float]]]]:
    if not last or last <= 0 or last >= len(history):
        return 0, history
    start = len(history) - last
    return start, history[start:]


def _print_episode_table(history: List[Dict[str, Optional[float]]], last: Optional[int]):
    start_idx, rows = _slice_rows(history, last)
    print("-" * 78)
    print("EPISODE LOG")
    print("-" * 78)
    print(f"Showing episodes {start_idx + 1}..{start_idx + len(rows)}")
    print("Ep      AvgCost     Loss        Eps")
    print("------  ----------  ----------  ----------")
    for i, r in enumerate(rows, start=start_idx + 1):
        print(
            f"{i:>6}  {_fmt(r['avg_cost']):>10}  {_fmt(r['total_loss']):>10}  {_fmt(r['eps']):>10}"
        )


def _print_checkpoints(history: List[Dict[str, Optional[float]]], points: int):
    if not history:
        return

    points = max(2, points)
    total = len(history)
    idxs = sorted(set(round(i * (total - 1) / (points - 1)) for i in range(points)))

    print("-" * 78)
    print("CHECKPOINT SNAPSHOT")
    print("-" * 78)
    print("Ep      AvgCost     Loss        Eps")
    print("------  ----------  ----------  ----------")
    for i in idxs:
        r = history[i]
        ep = i + 1
        print(f"{ep:>6}  {_fmt(r['avg_cost']):>10}  {_fmt(r['total_loss']):>10}  {_fmt(r['eps']):>10}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pretty-print training result JSON logs")
    p.add_argument("json_path", type=str, help="Path to results JSON")
    p.add_argument(
        "--detail",
        type=str,
        default="standard",
        choices=["brief", "standard", "full"],
        help="brief: summary only, standard: summary + sampled checkpoints, full: per-episode table",
    )
    p.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window size used in summary stats",
    )
    p.add_argument(
        "--last",
        type=int,
        default=0,
        help="For full detail, show only the last N episodes (0 means all)",
    )
    p.add_argument(
        "--checkpoints",
        type=int,
        default=10,
        help="Number of sampled checkpoint rows in standard mode",
    )
    p.add_argument(
        "--show-config",
        action="store_true",
        help="Print the full config section",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    src = Path(args.json_path)

    if not src.exists():
        print(f"Error: file not found: {src}")
        return 1

    try:
        data = _load_json(src)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON ({e})")
        return 1
    except OSError as e:
        print(f"Error: cannot read file ({e})")
        return 1

    history = _extract_history(data.get("history", []))
    summary = _build_summary(history, args.window)

    _print_header(data, src)
    if args.show_config:
        _print_config(data)
    _print_summary(summary, args.window)

    if args.detail == "standard":
        _print_checkpoints(history, args.checkpoints)
    elif args.detail == "full":
        _print_episode_table(history, args.last if args.last > 0 else None)

    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
