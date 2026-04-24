(rlProject) PS D:\Projects\RL Project> python train.py --episodes 500 --log_every 50   
Device: cuda

Training IL — 10 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
      50      1.4374      1.4932   0.782
     100      0.9580      1.4088   0.609
     150      1.2051      1.5211   0.474
     200      0.7059      1.5243   0.369
     250      0.8878      1.4632   0.287
     300      0.8882      1.3881   0.223
     350      0.7950      1.3383   0.174
     400      0.7382      1.2687   0.135
     450      0.9263      1.1190   0.105
     500      0.6656      1.0175   0.082

Eval (20 eps): mean=0.6626  std=0.0681
Saved to checkpoints/

Baselines  —  random: 1.4585  |  local: 0.9513
IL eval    —  0.6626 ± 0.0681
(rlProject) PS D:\Projects\RL Project> python trainGnn.py --episodes 500 --log_every 50
Device: cuda

Training GNN-IL — 10 UEs, 2 UAVs, 500 episodes
GNN: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
      50      1.3014      1.0316   0.782
     100      0.8891      0.8632   0.609
     150      0.7719      0.6922   0.474
     200      0.5940      0.5008   0.369
     250      0.6156      0.4467   0.287
     300      0.6480      0.3224   0.223
     350      0.7907      0.4218   0.174
     400      0.5840      0.3009   0.135
     450      0.6067      0.2618   0.105
     500      0.5364      0.3048   0.082

Eval (20 eps): mean=0.5650  std=0.0279
Saved to checkpoints/

GNN-IL eval  —  0.5650 ± 0.0279
(rlProject) PS D:\Projects\RL Project> 



(rlProject) PS D:\Projects\RL Project> .\runILandGNNAblation1.bat
Device: cuda

Training IL — 5 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.6174      0.4364   0.082

Eval (20 eps): mean=0.6037  std=0.0307
Saved to checkpoints/ue5/

Baselines  —  random: 0.6305  |  local: 0.9176
IL eval    —  0.6037 ± 0.0307
Device: cuda

Training GNN-IL — 5 UEs, 2 UAVs, 500 episodes
GNN: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.5784      0.2733   0.082

Eval (20 eps): mean=0.5719  std=0.0253
Saved to checkpoints/ue5/

GNN-IL eval  —  0.5719 ± 0.0253
Device: cuda

Training IL — 10 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.6656      1.0175   0.082

Eval (20 eps): mean=0.6626  std=0.0681
Saved to checkpoints/ue10/

Baselines  —  random: 1.4585  |  local: 0.9513
IL eval    —  0.6626 ± 0.0681
Device: cuda

Training GNN-IL — 10 UEs, 2 UAVs, 500 episodes
GNN: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.5364      0.3048   0.082

Eval (20 eps): mean=0.5650  std=0.0279
Saved to checkpoints/ue10/

GNN-IL eval  —  0.5650 ± 0.0279
Device: cuda

Training IL — 20 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.8588      1.7712   0.082

Eval (20 eps): mean=0.8120  std=0.0666
Saved to checkpoints/ue20/

Baselines  —  random: 3.6671  |  local: 0.9139
IL eval    —  0.8120 ± 0.0666
Device: cuda

Training GNN-IL — 20 UEs, 2 UAVs, 500 episodes
GNN: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.5688      0.5559   0.082

Eval (20 eps): mean=0.5833  std=0.0210
Saved to checkpoints/ue20/

GNN-IL eval  —  0.5833 ± 0.0210
Press any key to continue . . . 

UEs,  Random,     Local,      IL,               GNN-IL,           IL$\rightarrow$GNN-IL gain
5,    0.6305,     0.9176,     0.6037 ± 0.031,   0.5719 ± 0.025,   5.3%
10,   1.4585,     0.9513,     0.6626 ± 0.068,   0.5650 ± 0.028,   14.7%
20,   3.6671,     0.9139,     0.8120 ± 0.067,   0.5833 ± 0.021,   28.2%



====================================================
RUNNING: UEs=5 | SEED=42
====================================================
Training Standard IL...
Device: cuda

Training IL — 5 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.6174      0.4364   0.082

Eval (20 eps): mean=0.6037  std=0.0307
Saved to checkpoints/ue5/seed42/standard/

Baselines  —  random: 0.6305  |  local: 0.9176
IL eval    —  0.6037 ± 0.0307
Training GNN IL...
Device: cuda

Training GNN-IL — 5 UEs, 2 UAVs, 500 episodes
GNN: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.5784      0.2733   0.082

Eval (20 eps): mean=0.5719  std=0.0253
Saved to checkpoints/ue5/seed42/gnn/

GNN-IL eval  —  0.5719 ± 0.0253

====================================================
RUNNING: UEs=5 | SEED=52
====================================================
Training Standard IL...
Device: cuda

Training IL — 5 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.6100      0.4835   0.082

Eval (20 eps): mean=0.6127  std=0.0756
Saved to checkpoints/ue5/seed52/standard/

Baselines  —  random: 0.6198  |  local: 1.0019
IL eval    —  0.6127 ± 0.0756
Training GNN IL...
Device: cuda

Training GNN-IL — 5 UEs, 2 UAVs, 500 episodes
GNN: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.5845      0.2392   0.082

Eval (20 eps): mean=0.5806  std=0.0734
Saved to checkpoints/ue5/seed52/gnn/

GNN-IL eval  —  0.5806 ± 0.0734

====================================================
RUNNING: UEs=5 | SEED=62
====================================================
Training Standard IL...
Device: cuda

Training IL — 5 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.6150      0.4507   0.082

Eval (20 eps): mean=0.6247  std=0.0579
Saved to checkpoints/ue5/seed62/standard/

Baselines  —  random: 0.6353  |  local: 0.9948
IL eval    —  0.6247 ± 0.0579
Training GNN IL...
Device: cuda

Training GNN-IL — 5 UEs, 2 UAVs, 500 episodes
GNN: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.6121      0.2438   0.082

Eval (20 eps): mean=0.6336  std=0.0579
Saved to checkpoints/ue5/seed62/gnn/

GNN-IL eval  —  0.6336 ± 0.0579

====================================================
RUNNING: UEs=5 | SEED=72
====================================================
Training Standard IL...
Device: cuda

Training IL — 5 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.5788      0.4280   0.082

Eval (20 eps): mean=0.5654  std=0.0426
Saved to checkpoints/ue5/seed72/standard/

Baselines  —  random: 0.6569  |  local: 0.9831
IL eval    —  0.5654 ± 0.0426
Training GNN IL...
Device: cuda

Training GNN-IL — 5 UEs, 2 UAVs, 500 episodes
GNN: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.5508      0.3570   0.082

Eval (20 eps): mean=0.5772  std=0.0385
Saved to checkpoints/ue5/seed72/gnn/

GNN-IL eval  —  0.5772 ± 0.0385

====================================================
RUNNING: UEs=5 | SEED=42
====================================================
Skipping Standard: Folder already exists.
Skipping GNN: Folder already exists.

====================================================
RUNNING: UEs=5 | SEED=52
====================================================
Skipping Standard: Folder already exists.
Skipping GNN: Folder already exists.

====================================================
RUNNING: UEs=5 | SEED=62
====================================================
Skipping Standard: Folder already exists.
Skipping GNN: Folder already exists.

====================================================
RUNNING: UEs=10 | SEED=42
====================================================
Training Standard IL...
Device: cuda

Training IL — 10 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.6656      1.0175   0.082

Eval (20 eps): mean=0.6626  std=0.0681
Saved to checkpoints/ue10/seed42/standard/

Baselines  —  random: 1.4585  |  local: 0.9513
IL eval    —  0.6626 ± 0.0681
Training GNN IL...
Device: cuda

Training GNN-IL — 10 UEs, 2 UAVs, 500 episodes
GNN: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.5364      0.3048   0.082

Eval (20 eps): mean=0.5650  std=0.0279
Saved to checkpoints/ue10/seed42/gnn/

GNN-IL eval  —  0.5650 ± 0.0279

====================================================
RUNNING: UEs=10 | SEED=52
====================================================
Training Standard IL...
Device: cuda

Training IL — 10 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.7892      1.3565   0.082

Eval (20 eps): mean=0.7819  std=0.1269
Saved to checkpoints/ue10/seed52/standard/

Baselines  —  random: 1.3854  |  local: 1.0143
IL eval    —  0.7819 ± 0.1269
Training GNN IL...
Device: cuda

Training GNN-IL — 10 UEs, 2 UAVs, 500 episodes
GNN: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.5445      0.3985   0.082

Eval (20 eps): mean=0.6638  std=0.0501
Saved to checkpoints/ue10/seed52/gnn/

GNN-IL eval  —  0.6638 ± 0.0501

====================================================
RUNNING: UEs=10 | SEED=62
====================================================
Training Standard IL...
Device: cuda

Training IL — 10 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.7244      1.0271   0.082

Eval (20 eps): mean=0.7323  std=0.0959
Saved to checkpoints/ue10/seed62/standard/

Baselines  —  random: 1.4982  |  local: 0.9129
IL eval    —  0.7323 ± 0.0959
Training GNN IL...
Device: cuda

Training GNN-IL — 10 UEs, 2 UAVs, 500 episodes
GNN: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.5397      0.3062   0.082

Eval (20 eps): mean=0.5687  std=0.0337
Saved to checkpoints/ue10/seed62/gnn/

GNN-IL eval  —  0.5687 ± 0.0337

====================================================
RUNNING: UEs=20 | SEED=42
====================================================
Training Standard IL...
Device: cuda

Training IL — 20 UEs, 2 UAVs, 500 episodes

====================================================
RUNNING: UEs=5 | SEED=42
====================================================
Skipping Standard: Folder already exists.
Skipping GNN: Folder already exists.

====================================================
RUNNING: UEs=5 | SEED=52
====================================================
Skipping Standard: Folder already exists.
Skipping GNN: Folder already exists.

====================================================
RUNNING: UEs=5 | SEED=62
====================================================
Skipping Standard: Folder already exists.
Skipping GNN: Folder already exists.

====================================================
RUNNING: UEs=10 | SEED=42
====================================================
Skipping Standard: Folder already exists.
Skipping GNN: Folder already exists.

====================================================
RUNNING: UEs=10 | SEED=52
====================================================
Skipping Standard: Folder already exists.
Skipping GNN: Folder already exists.

====================================================
RUNNING: UEs=10 | SEED=62
====================================================
Skipping Standard: Folder already exists.
Skipping GNN: Folder already exists.

====================================================
RUNNING: UEs=20 | SEED=42
====================================================
Training Standard IL...
Device: cuda

Training IL — 20 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.8588      1.7712   0.082

Eval (20 eps): mean=0.8120  std=0.0666
Saved to checkpoints/ue20/seed42/standard/

Baselines  —  random: 3.6671  |  local: 0.9139
IL eval    —  0.8120 ± 0.0666
Training GNN IL...
Device: cuda

Training GNN-IL — 20 UEs, 2 UAVs, 500 episodes
GNN: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.5688      0.5559   0.082

Eval (20 eps): mean=0.5833  std=0.0210
Saved to checkpoints/ue20/seed42/gnn/

GNN-IL eval  —  0.5833 ± 0.0210

====================================================
RUNNING: UEs=20 | SEED=52
====================================================
Training Standard IL...
Device: cuda

Training IL — 20 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      1.2004      1.9305   0.082

Eval (20 eps): mean=0.9145  std=0.1277
Saved to checkpoints/ue20/seed52/standard/

Baselines  —  random: 3.6205  |  local: 1.0160
IL eval    —  0.9145 ± 0.1277
Training GNN IL...
Device: cuda

Training GNN-IL — 20 UEs, 2 UAVs, 500 episodes
GNN: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.7120      0.3682   0.082

Eval (20 eps): mean=0.6579  std=0.0321
Saved to checkpoints/ue20/seed52/gnn/

GNN-IL eval  —  0.6579 ± 0.0321

====================================================
RUNNING: UEs=20 | SEED=62
====================================================
Training Standard IL...
Device: cuda

Training IL — 20 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      1.1619      1.9083   0.082

Eval (20 eps): mean=1.0497  std=0.1196
Saved to checkpoints/ue20/seed62/standard/

Baselines  —  random: 3.5769  |  local: 0.9550
IL eval    —  1.0497 ± 0.1196
Training GNN IL...
Device: cuda

Training GNN-IL — 20 UEs, 2 UAVs, 500 episodes
GNN: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      1.1971      2.8508   0.082

Eval (20 eps): mean=0.9217  std=0.0556
Saved to checkpoints/ue20/seed62/gnn/

GNN-IL eval  —  0.9217 ± 0.0556

====================================================
RUNNING: UEs=30 | SEED=42
====================================================
Training Standard IL...
Device: cuda

Training IL — 30 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      1.4887      1.3527   0.082

Eval (20 eps): mean=1.3383  std=0.1029
Saved to checkpoints/ue30/seed42/standard/

Baselines  —  random: 4.0575  |  local: 0.9503
IL eval    —  1.3383 ± 0.1029
Training GNN IL...
Device: cuda

Training GNN-IL — 30 UEs, 2 UAVs, 500 episodes
GNN: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
     500      0.7005      0.7444   0.082

Eval (20 eps): mean=0.6595  std=0.0303
Saved to checkpoints/ue30/seed42/gnn/

GNN-IL eval  —  0.6595 ± 0.0303


| UEs | Seed | Random Baseline | Local Baseline | Standard IL Eval (Mean ± Std) | GNN IL Eval (Mean ± Std) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5 | 42 | 0.6305 | 0.9176 | 0.6037 ± 0.0307 | 0.5719 ± 0.0253 |
| 5 | 52 | 0.6198 | 1.0019 | 0.6127 ± 0.0756 | 0.5806 ± 0.0734 |
| 5 | 62 | 0.6353 | 0.9948 | 0.6247 ± 0.0579 | 0.6336 ± 0.0579 |
| 5 | 72 | 0.6569 | 0.9831 | 0.5654 ± 0.0426 | 0.5772 ± 0.0385 |
| 10 | 42 | 1.4585 | 0.9513 | 0.6626 ± 0.0681 | 0.5650 ± 0.0279 |
| 10 | 52 | 1.3854 | 1.0143 | 0.7819 ± 0.1269 | 0.6638 ± 0.0501 |
| 10 | 62 | 1.4982 | 0.9129 | 0.7323 ± 0.0959 | 0.5687 ± 0.0337 |
| 20 | 42 | 3.6671 | 0.9139 | 0.8120 ± 0.0666 | 0.5833 ± 0.0210 |
| 20 | 52 | 3.6205 | 1.0160 | 0.9145 ± 0.1277 | 0.6579 ± 0.0321 |
| 20 | 62 | 3.5769 | 0.9550 | 1.0497 ± 0.1196 | 0.9217 ± 0.0556 |
| 30 | 42 | 4.0575 | 0.9503 | 1.3383 ± 0.1029 | 0.6595 ± 0.0303 |

(rlProject) PS D:\Projects\RL Project> python trainGru.py     --episodes 500 --log_every 50
Device: cuda

Training IL+GRU — 10 UEs, 2 UAVs, 500 episodes  |  chunk_len=8

 Episode     AvgCost        Loss     Eps
------------------------------------------
      50      0.9662      1.2932   0.782
     100      0.9856      1.7683   0.609
     150      1.6236      2.4689   0.474
     200      1.2089      2.1754   0.369
     250      0.7475      2.6693   0.287
     300      1.0994      2.7771   0.223
     350      1.1207      2.9673   0.174
     400      0.9709      3.4222   0.135
     450      1.0836      2.4596   0.105
     500      0.9559      2.7606   0.082

Eval (20 eps): mean=0.9574  std=0.1712
Saved to checkpoints/

IL+GRU eval  —  0.9574 ± 0.1712
(rlProject) PS D:\Projects\RL Project> cpython train_il_stack.py --episodes 500 --log_every 50^C
(rlProject) PS D:\Projects\RL Project> python trainStack.py --episodes 500 --log_every 50
Device: cuda

Training IL+Stack — 10 UEs, 2 UAVs, 500 episodes  |  K=4  stacked_dim=44

 Episode     AvgCost        Loss     Eps
------------------------------------------
      50      1.0978      1.3987   0.782
     100      0.8377      1.1874   0.609
     150      1.2153      1.2629   0.474
     200      0.8261      1.2727   0.369
     250      1.1232      1.3527   0.287
     300      1.1358      1.2197   0.223
     350      1.0639      1.2593   0.174
     400      0.6915      1.2561   0.135
     450      0.9732      1.1609   0.105
     500      0.6749      1.1180   0.082

Eval (20 eps): mean=0.6847  std=0.0646
Saved to checkpoints/

IL+Stack eval  —  0.6847 ± 0.0646
(rlProject) PS D:\Projects\RL Project>