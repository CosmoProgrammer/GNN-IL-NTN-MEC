# diagnight analysis

COLLAPSED = eval_mean > 0.9; onset/lead definitions in analyze_diagnight.py docstring (pre-registered caveats in RESEARCH_LOG S18.4/S18.6: freeze_replay asymmetric [tandem], ctde/ = CTDE(Bn), freeze_learn eval = frozen policy).

## Arms / cells

| M | cell | n | collapse | mean eval | med eval | mean trigger_ep | best-eval rescue* |
|---|------|---|----------|-----------|----------|-----------------|---|
| 20 | base | 10 | 2/10 | 0.7848 | 0.7368 | — | 2/2 |
| 20 | eps0 | 10 | 1/10 | 0.7127 | 0.6605 | 10 | 1/1 |
| 20 | eps0.9975_ep1000 | 10 | 1/10 | 0.9738 | 0.6922 | — | 1/1 |
| 20 | eps0.999_ep1000 | 10 | 2/10 | 0.7934 | 0.7582 | — | 2/2 |
| 20 | floor0.1 | 10 | 2/10 | 0.7738 | 0.6809 | — | 2/2 |
| 20 | floor0.2 | 10 | 4/10 | 0.8114 | 0.7948 | — | 4/4 |
| 20 | floor0.3 | 10 | 3/10 | 0.8157 | 0.7068 | — | 3/3 |
| 20 | floor0.4 | 10 | 5/10 | 0.8521 | 0.8286 | — | 5/5 |
| 20 | freezelearn | 10 | 6/10 | 0.8923 | 0.9471 | 10 | 6/6 |
| 20 | freezereplay | 10 | 9/10 | 2.6361 | 2.5489 | 10 | 9/9 |
| 20 | tgt5 | 10 | 5/10 | 1.0644 | 0.8979 | — | 5/5 |
| 20 | tgt50 | 10 | 4/10 | 0.9435 | 0.6902 | — | 4/4 |
| 30 | base | 10 | 3/10 | 0.8170 | 0.6971 | — | 3/3 |
| 30 | eps0 | 10 | 1/10 | 0.6842 | 0.6340 | 45 | 1/1 |
| 30 | floor0.1 | 10 | 3/10 | 0.8326 | 0.7303 | — | 3/3 |
| 30 | floor0.2 | 10 | 3/10 | 0.8639 | 0.7661 | — | 3/3 |
| 30 | floor0.3 | 10 | 4/10 | 0.8331 | 0.7586 | — | 4/4 |
| 30 | floor0.4 | 10 | 4/10 | 0.8883 | 0.8668 | — | 4/4 |
| 30 | freezelearn | 10 | 0/10 | 0.7842 | 0.7859 | 45 | 0/0 |
| 30 | freezereplay | 10 | 8/10 | 1.1847 | 1.0341 | 45 | 8/8 |
| 30 | tgt5 | 10 | 4/10 | 1.0741 | 0.8395 | — | 4/4 |
| 30 | tgt50 | 10 | 5/10 | 0.8972 | 0.8678 | — | 5/5 |

*collapsed runs whose best greedy CRN eval during training was <= 0.9 — the 'early stopping would have rescued it' count.

## Dose-response: collapse rate vs eps floor

M=20: floor 0.05 -> 20% (n=10)  floor 0.1 -> 20% (n=10)  floor 0.2 -> 40% (n=10)  floor 0.3 -> 30% (n=10)  floor 0.4 -> 50% (n=10)
M=30: floor 0.05 -> 30% (n=10)  floor 0.1 -> 30% (n=10)  floor 0.2 -> 30% (n=10)  floor 0.3 -> 40% (n=10)  floor 0.4 -> 40% (n=10)

## Blowup ordering (which signal moves first before divergence)

run shapes: 32 converged-and-stayed, 162 converged-then-diverged (onset defined), 26 never-learned

| signal | n | median lead (eps) | p25 | p75 | first mover |
|--------|---|-------------------|-----|-----|-------------|
| q_max_abs | 82 | -75 | -178 | -20 | 31 |
| td_p95 | 50 | -90 | -130 | 0 | 17 |
| grad_max | 35 | -100 | -140 | -20 | 1 |
| cong_mean | 53 | -90 | -150 | 0 | 16 |
| act_churn | 68 | -95 | -240 | 0 | 23 |

## Method x M (10-seed, same hardware)

| method | M=5 | M=10 | M=20 | M=30 | M=40 |
|--------|---|---|---|---|---|
| IL | 0.601±0.013 (0/10X) | 0.786±0.074 (1/10X) | 0.908±0.061 (4/10X) | 1.327±0.129 (10/10X) | 1.018±0.064 (9/10X) |
| CTDE-plain | — | — | — | — | — |
| CTDE(Bn) | 0.639±0.047 (0/10X) | 0.657±0.088 (0/10X) | 1.294±0.959 (6/10X) | 0.806±0.156 (3/10X) | 0.852±0.140 (2/10X) |
| GNN-NoShare | 0.628±0.017 (0/10X) | 0.839±0.075 (2/10X) | 1.289±0.133 (10/10X) | 1.241±0.111 (10/10X) | 1.557±0.115 (10/10X) |
| GNN-IL | 0.610±0.037 (0/10X) | 0.589±0.040 (0/10X) | 0.785±0.202 (2/10X) | 0.817±0.241 (3/10X) | 0.768±0.102 (1/10X) |

(mean±std eval over seeds; X = collapsed count. CTDE(Bn) sees Bn in obs — fairness cell, not the 2x2 cell.)

## PoA deep (20 BR restarts)

| M | N | opt_est | PNE best | PNE worst | PoA_est | PoS_est |
|---|---|---------|----------|-----------|---------|---------|
| 5 | 2 | 0.6782 | 0.6839 | 0.8469 | 1.249 | 1.008 |
| 10 | 2 | 0.7692 | 0.7692 | 0.8441 | 1.097 | 1.000 |
| 20 | 2 | 0.9302 | 0.9302 | 1.0028 | 1.078 | 1.000 |
| 30 | 2 | 0.9195 | 0.9229 | 0.9548 | 1.038 | 1.004 |
| 40 | 2 | 0.8959 | 0.8997 | 0.9300 | 1.038 | 1.004 |
| 5 | 4 | 0.6021 | 0.6021 | 0.6179 | 1.026 | 1.000 |
| 10 | 4 | 0.6865 | 0.6982 | 0.7746 | 1.128 | 1.017 |
| 20 | 4 | 0.8625 | 0.8990 | 0.9892 | 1.147 | 1.042 |
| 30 | 4 | 0.8646 | 0.8922 | 0.9493 | 1.098 | 1.032 |
| 40 | 4 | 0.8780 | 0.8780 | 0.9147 | 1.042 | 1.000 |

