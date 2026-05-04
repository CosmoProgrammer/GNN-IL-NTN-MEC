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

(rlProject) PS D:\Projects\RL Project> python .\view_results.py checkpoints\ue10WithBL\il_results.json --detail standard --show-config
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : checkpoints\ue10WithBL\il_results.json
Method     : IL
Eval mean  : 0.6882
Eval std   : 0.0479
Best train : 0.5586
------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------
batch_size    : 32
cpu           : False
episodes      : 500
eps_decay     : 0.995
eval_episodes : 20
gamma         : 0.9
hidden        : 128
log_every     : 500
lr            : 0.001
n_uavs        : 2
n_ues         : 10
save_dir      : checkpoints/ue10WithBL
seed          : 42
steps_per_ep  : 100
target_update : 20
task_prob     : 0.3
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 1.5235 -> 0.6713
Loss (first -> last)       : 2.1314 -> 0.7998
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5586
Best avg cost episode      : 430
Best loss                  : 0.7559
Best loss episode          : 490
Cost improvement (%)       : 55.93
Rolling avg cost (w=20) : 0.7575
Rolling avg loss (w=20) : 0.7963
------------------------------------------------------------------------------
CHECKPOINT SNAPSHOT
------------------------------------------------------------------------------
Ep      AvgCost     Loss        Eps
------  ----------  ----------  ----------
     1      1.5235      2.1314      1.0000
    56      0.8879      1.1230      0.7590
   112      0.8000      1.0264      0.5733
   167      0.9707      1.0239      0.4351
   223      0.9768      1.0188      0.3286
   278      0.8851      0.9998      0.2495
   334      0.6681      0.9257      0.1884
   389      0.7592      0.8855      0.1430
   445      0.7746      0.8696      0.1080
   500      0.6713      0.7998      0.0820
==============================================================================
(rlProject) PS D:\Projects\RL Project> python .\view_results.py checkpoints\ue10\seed42\standard\il_results.json --detail standard --show-config
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : checkpoints\ue10\seed42\standard\il_results.json
Method     : IL
Eval mean  : 0.6626
Eval std   : 0.0681
Best train : 0.5214
------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------
batch_size    : 32
cpu           : False
episodes      : 500
eps_decay     : 0.995
eval_episodes : 20
gamma         : 0.9
hidden        : 128
log_every     : 500
lr            : 0.001
n_uavs        : 2
n_ues         : 10
save_dir      : checkpoints/ue10/seed42/standard
seed          : 42
steps_per_ep  : 100
target_update : 20
task_prob     : 0.3
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 1.5235 -> 0.6656
Loss (first -> last)       : 2.0661 -> 1.0175
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5214
Best avg cost episode      : 430
Best loss                  : 1.0083
Best loss episode          : 483
Cost improvement (%)       : 56.31
Rolling avg cost (w=20) : 0.7181
Rolling avg loss (w=20) : 1.0768
------------------------------------------------------------------------------
CHECKPOINT SNAPSHOT
------------------------------------------------------------------------------
Ep      AvgCost     Loss        Eps
------  ----------  ----------  ----------
     1      1.5235      2.0661      1.0000
    56      1.1064      1.4599      0.7590
   112      0.9715      1.5179      0.5733
   167      1.1475      1.5401      0.4351
   223      1.0265      1.4866      0.3286
   278      1.0418      1.4066      0.2495
   334      0.7083      1.3331      0.1884
   389      0.7440      1.2377      0.1430
   445      0.6771      1.2066      0.1080
   500      0.6656      1.0175      0.0820
==============================================================================

(rlProject) PS D:\Projects\RL Project> python trainCtde.py --episodes 500 --log_every 50 --n_ues 10 --save_dir checkpoints/ue10CTDEWithoutBI                                                                                              
Device: cuda                                                                                           
                                                                      
Training CTDE — 10 UEs, 2 UAVs, 500 episodes   
                                                
 Episode     AvgCost        Loss     Eps
------------------------------------------
      50      1.6664      1.8205   0.782
     100      1.2767      1.7923   0.609                                  
     150      1.4261      1.7795   0.474
     200      0.6711      1.8532   0.369
     250      1.4437      1.5655   0.287         
     300      1.0639      1.5162   0.223
     350      0.9486      1.4158   0.174
     400      0.8007      1.6948   0.135
     450      1.0128      1.4287   0.105
     500      0.8202      1.4679   0.082

Eval (20 eps): mean=0.9335  std=0.1454
Saved to checkpoints/ue10CTDEWithoutBI/

Baselines  —  random: 1.4585  |  local: 0.9513
CTDE eval  —  0.9335 ± 0.1454

(rlProject) PS D:\Projects\RL Project> python trainCtde.py --episodes 500 --log_every 50 --n_ues 10 --save_dir checkpoints/ue10CTDEWithBI    
Device: cuda

Training CTDE — 10 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
      50      1.4082      1.4431   0.782
     100      0.8813      1.2970   0.609
     150      1.1091      1.1214   0.474
     200      0.7390      1.0147   0.369
     250      1.1662      0.9648   0.287
     300      0.8128      0.8786   0.223
     350      0.8870      0.8228   0.174
     400      0.6779      0.8815   0.135
     450      1.0425      0.7165   0.105
     500      0.6355      0.6326   0.082

Eval (20 eps): mean=0.6286  std=0.0345
Saved to checkpoints/ue10CTDEWithBI/

Baselines  —  random: 1.4585  |  local: 0.9513
CTDE eval  —  0.6286 ± 0.0345

(rlProject) PS D:\Projects\RL Project> python .\view_results.py checkpoints\ue10CTDEWithoutBI\ctde_results.json --detail standard --show-config 
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : checkpoints\ue10CTDEWithoutBI\ctde_results.json
Method     : CTDE
Eval mean  : 0.9335
Eval std   : 0.1454
Best train : 0.5229
------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------
batch_size    : 32
cpu           : False
episodes      : 500
eps_decay     : 0.995
eval_episodes : 20
gamma         : 0.9
hidden        : 128
log_every     : 50
lr            : 0.001
n_uavs        : 2
n_ues         : 10
per_agent_cap : 10000
save_dir      : checkpoints/ue10CTDEWithoutBI
seed          : 42
steps_per_ep  : 100
target_update : 20
task_prob     : 0.3
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 1.3957 -> 0.8202
Loss (first -> last)       : 1.8405 -> 1.4679
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5229
Best avg cost episode      : 289
Best loss                  : 1.2452
Best loss episode          : 495
Cost improvement (%)       : 41.23
Rolling avg cost (w=20) : 0.9481
Rolling avg loss (w=20) : 1.4341
------------------------------------------------------------------------------
CHECKPOINT SNAPSHOT
------------------------------------------------------------------------------
Ep      AvgCost     Loss        Eps
------  ----------  ----------  ----------
     1      1.3957      1.8405      1.0000
    56      0.9146      1.8049      0.7590
   112      1.1510      1.9413      0.5733
   167      1.0380      1.8224      0.4351
   223      1.0099      1.7645      0.3286
   278      1.1067      1.5835      0.2495
   334      0.8618      1.6616      0.1884
   389      0.8114      1.4964      0.1430
   445      1.1177      1.3641      0.1080
   500      0.8202      1.4679      0.0820
==============================================================================
(rlProject) PS D:\Projects\RL Project> python .\view_results.py checkpoints\ue10CTDEWithBI\ctde_results.json --detail standard --show-config   
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : checkpoints\ue10CTDEWithBI\ctde_results.json
Method     : CTDE
Eval mean  : 0.6286
Eval std   : 0.0345
Best train : 0.5771
------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------
batch_size    : 32
cpu           : False
episodes      : 500
eps_decay     : 0.995
eval_episodes : 20
gamma         : 0.9
hidden        : 128
log_every     : 50
lr            : 0.001
n_uavs        : 2
n_ues         : 10
per_agent_cap : 10000
save_dir      : checkpoints/ue10CTDEWithBI
seed          : 42
steps_per_ep  : 100
target_update : 20
task_prob     : 0.3
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 1.3957 -> 0.6355
Loss (first -> last)       : 1.7072 -> 0.6326
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5771
Best avg cost episode      : 289
Best loss                  : 0.5689
Best loss episode          : 485
Cost improvement (%)       : 54.46
Rolling avg cost (w=20) : 0.7703
Rolling avg loss (w=20) : 0.6817
------------------------------------------------------------------------------
CHECKPOINT SNAPSHOT
------------------------------------------------------------------------------
Ep      AvgCost     Loss        Eps
------  ----------  ----------  ----------
     1      1.3957      1.7072      1.0000
    56      0.8512      1.4217      0.7590
   112      0.8389      1.2476      0.5733
   167      1.0441      1.0161      0.4351
   223      0.8000      1.0113      0.3286
   278      0.7516      0.9827      0.2495
   334      0.6988      0.9594      0.1884
   389      0.9085      0.8896      0.1430
   445      0.8833      0.6421      0.1080
   500      0.6355      0.6326      0.0820
==============================================================================
(rlProject) PS D:\Projects\RL Project> 

(rlProject) PS D:\Projects\RL Project> python trainCtde.py --episodes 500 --log_every 50 --n_ues 20 --save_dir checkpoints/ue20CTDEWithBI      
Device: cuda

Training CTDE — 20 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
      50      2.4470      1.6620   0.782
     100      2.0973      1.5434   0.609
     150      1.6047      1.5470   0.474
     200      1.7445      1.6964   0.369
     250      1.2507      1.6228   0.287
     300      1.2821      1.6680   0.223
     350      1.1945      1.7508   0.174
     400      1.0367      1.6121   0.135
     450      1.1528      1.6765   0.105
     500      0.9461      1.5881   0.082

Eval (20 eps): mean=1.1416  std=0.0585
Saved to checkpoints/ue20CTDEWithBI/

Baselines  —  random: 3.6671  |  local: 0.9139
CTDE eval  —  1.1416 ± 0.0585
(rlProject) PS D:\Projects\RL Project> python trainCtde.py --episodes 500 --log_every 50 --n_ues 30 --save_dir checkpoints/ue30CTDEWithBI   
Device: cuda

Training CTDE — 30 UEs, 2 UAVs, 500 episodes

 Episode     AvgCost        Loss     Eps
------------------------------------------
      50      3.2329      1.1697   0.782
     100      2.7314      1.6536   0.609
     150      1.8386      1.9515   0.474
     200      1.6719      1.9297   0.369
     250      1.1163      1.8926   0.287
     300      1.2074      1.7575   0.223
     350      1.0669      1.5599   0.174
     400      0.8593      1.6068   0.135
     450      0.8924      1.4842   0.105
     500      0.7980      1.3668   0.082

Eval (20 eps): mean=0.6175  std=0.0327
Saved to checkpoints/ue30CTDEWithBI/

Baselines  —  random: 4.0575  |  local: 0.9503
CTDE eval  —  0.6175 ± 0.0327

(rlProject) PS D:\Projects\RL Project> python .\view_results.py checkpoints\ue20CTDEWithBI\ctde_results.json --detail
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : checkpoints\ue20CTDEWithBI\ctde_results.json
Method     : CTDE
Eval mean  : 1.1416
Eval std   : 0.0585
Best train : 0.8747
------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------
batch_size    : 32
cpu           : False
episodes      : 500
eps_decay     : 0.995
eval_episodes : 20
gamma         : 0.9
hidden        : 128
log_every     : 50
lr            : 0.001
n_uavs        : 2
n_ues         : 20
per_agent_cap : 10000
save_dir      : checkpoints/ue20CTDEWithBI
seed          : 42
steps_per_ep  : 100
target_update : 20
task_prob     : 0.3
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.6683 -> 0.9461
Loss (first -> last)       : 2.9523 -> 1.5881
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.8747
Best avg cost episode      : 498
Best loss                  : 1.4144
Best loss episode          : 120
Cost improvement (%)       : 74.21
Rolling avg cost (w=20) : 1.0380
Rolling avg loss (w=20) : 1.5477
------------------------------------------------------------------------------
CHECKPOINT SNAPSHOT
------------------------------------------------------------------------------
Ep      AvgCost     Loss        Eps
------  ----------  ----------  ----------
     1      3.6683      2.9523      1.0000
    56      2.4413      1.5572      0.7590
   112      2.3713      1.5395      0.5733
   167      1.5439      1.5218      0.4351
   223      1.4715      1.6981      0.3286
   278      1.4678      1.6392      0.2495
   334      1.3862      1.7145      0.1884
   389      1.3885      1.7465      0.1430
   445      1.0878      1.6599      0.1080
   500      0.9461      1.5881      0.0820
==============================================================================
(rlProject) PS D:\Projects\RL Project> python .\view_results.py checkpoints\ue30CTDEWithBI\ctde_results.json --detail standard --show-config
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : checkpoints\ue30CTDEWithBI\ctde_results.json
Method     : CTDE
Eval mean  : 0.6175
Eval std   : 0.0327
Best train : 0.6462
------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------
batch_size    : 32
cpu           : False
episodes      : 500
eps_decay     : 0.995
eval_episodes : 20
gamma         : 0.9
hidden        : 128
log_every     : 50
lr            : 0.001
n_uavs        : 2
n_ues         : 30
per_agent_cap : 10000
save_dir      : checkpoints/ue30CTDEWithBI
seed          : 42
steps_per_ep  : 100
target_update : 20
task_prob     : 0.3
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.5096 -> 0.7980
Loss (first -> last)       : 1.9857 -> 1.3668
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6462
Best avg cost episode      : 462
Best loss                  : 0.9551
Best loss episode          : 5
Cost improvement (%)       : 82.30
Rolling avg cost (w=20) : 0.8257
Rolling avg loss (w=20) : 1.4019
------------------------------------------------------------------------------
CHECKPOINT SNAPSHOT
------------------------------------------------------------------------------
Ep      AvgCost     Loss        Eps
------  ----------  ----------  ----------
     1      4.5096      1.9857      1.0000
    56      3.1999      1.2204      0.7590
   112      2.0889      1.7740      0.5733
   167      1.6375      1.7610      0.4351
   223      1.4007      1.9651      0.3286
   278      1.4911      1.7430      0.2495
   334      1.1263      1.7478      0.1884
   389      0.9310      1.5168      0.1430
   445      0.8084      1.5463      0.1080
   500      0.7980      1.3668      0.0820
==============================================================================
(rlProject) PS D:\Projects\RL Project> 

(rlProject) PS D:\Projects\RL Project> python trainCtdeGNN.py --episodes 500 --log_every 50 --n_ues 10 --save_dir checkpoints/ue10CTDEGNN     
Device: cuda

Training GNN-CTDE — 10 UEs, 2 UAVs, 500 episodes
GNN: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
      50      1.3745      1.5196   0.782
     100      0.9383      1.4339   0.609
     150      1.4477      1.3973   0.474
     200      0.8324      1.2427   0.369
     250      0.8666      1.2480   0.287
     300      0.8464      1.1372   0.223
     350      0.8008      1.0113   0.174
     400      0.8028      1.0247   0.135
     450      0.7820      0.8127   0.105
     500      0.6977      0.7240   0.082

Eval (20 eps): mean=0.6933  std=0.0646
Saved to checkpoints/ue10CTDEGNN/

GNN-CTDE eval  —  0.6933 ± 0.0646
(rlProject) PS D:\Projects\RL Project> python .\view_results.py checkpoints\ue10CTDEGNN\gnn_ctde_results.json --detail standard --show-config
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : checkpoints\ue10CTDEGNN\gnn_ctde_results.json
Method     : GNN-CTDE
Eval mean  : 0.6933
Eval std   : 0.0646
Best train : 0.5622
------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------
batch_size    : 32
cpu           : False
dqn_hidden    : 128
episodes      : 500
eps_decay     : 0.995
eval_episodes : 20
gamma         : 0.9
gnn_hidden    : 64
gnn_out       : 32
log_every     : 50
lr            : 0.001
n_uavs        : 2
n_ues         : 10
per_agent_cap : 5000
save_dir      : checkpoints/ue10CTDEGNN
seed          : 42
steps_per_ep  : 100
target_update : 20
task_prob     : 0.3
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 1.3957 -> 0.6977
Loss (first -> last)       : 1.7581 -> 0.7240
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5622
Best avg cost episode      : 430
Best loss                  : 0.6731
Best loss episode          : 487
Cost improvement (%)       : 50.01
Rolling avg cost (w=20) : 0.6960
Rolling avg loss (w=20) : 0.7853
------------------------------------------------------------------------------
CHECKPOINT SNAPSHOT
------------------------------------------------------------------------------
Ep      AvgCost     Loss        Eps
------  ----------  ----------  ----------
     1      1.3957      1.7581      1.0000
    56      0.9317      1.4501      0.7590
   112      1.0396      1.3347      0.5733
   167      1.0947      1.3949      0.4351
   223      0.9437      1.4028      0.3286
   278      0.9231      1.1182      0.2495
   334      0.7753      1.1156      0.1884
   389      0.8742      0.9811      0.1430
   445      0.6866      0.9493      0.1080
   500      0.6977      0.7240      0.0820
==============================================================================

==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : checkpoints\ue10NoShare\gnn_noshare_results.json
Method     : GNN-IL-NoShare
Eval mean  : 0.7401
Eval std   : 0.0711
Best train : 0.5680
------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------
batch_size    : 32
buffer_cap    : 5000
cpu           : False
dqn_hidden    : 128
episodes      : 500
eps_decay     : 0.995
eval_episodes : 20
gamma         : 0.9
gnn_hidden    : 64
gnn_out       : 32
log_every     : 50
lr            : 0.001
n_uavs        : 2
n_ues         : 10
save_dir      : checkpoints/ue10NoShare
seed          : 42
steps_per_ep  : 100
target_update : 20
task_prob     : 0.3
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 1.6998 -> 0.7462
Loss (first -> last)       : 2.8001 -> 4.6332
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5680
Best avg cost episode      : 352
Best loss                  : 0.8263
Best loss episode          : 157
Cost improvement (%)       : 56.10
Rolling avg cost (w=20) : 0.8340
Rolling avg loss (w=20) : 3.7049
------------------------------------------------------------------------------
CHECKPOINT SNAPSHOT
------------------------------------------------------------------------------
Ep      AvgCost     Loss        Eps
------  ----------  ----------  ----------
     1      1.6998      2.8001      1.0000
    56      0.7917      1.0883      0.7590
   112      0.7600      0.8844      0.5733
   167      0.7872      0.9155      0.4351
   223      0.8667      1.4106      0.3286
   278      0.7834      2.1669      0.2495
   334      0.6612      1.8190      0.1884
   389      0.7942      1.9450      0.1430
   445      0.8119      2.4039      0.1080
   500      0.7462      4.6332      0.0820
==============================================================================
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : checkpoints\ue20NoShare\gnn_noshare_results.json
Method     : GNN-IL-NoShare
Eval mean  : 1.2463
Eval std   : 0.0846
Best train : 0.9794
------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------
batch_size    : 32
buffer_cap    : 5000
cpu           : False
dqn_hidden    : 128
episodes      : 500
eps_decay     : 0.995
eval_episodes : 20
gamma         : 0.9
gnn_hidden    : 64
gnn_out       : 32
log_every     : 50
lr            : 0.001
n_uavs        : 2
n_ues         : 20
save_dir      : checkpoints/ue20NoShare
seed          : 42
steps_per_ep  : 100
target_update : 20
task_prob     : 0.3
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.8089 -> 1.2714
Loss (first -> last)       : 5.6058 -> 3.2628
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.9794
Best avg cost episode      : 338
Best loss                  : 1.5160
Best loss episode          : 313
Cost improvement (%)       : 66.62
Rolling avg cost (w=20) : 1.3125
Rolling avg loss (w=20) : 2.8450
------------------------------------------------------------------------------
CHECKPOINT SNAPSHOT
------------------------------------------------------------------------------
Ep      AvgCost     Loss        Eps
------  ----------  ----------  ----------
     1      3.8089      5.6058      1.0000
    56      2.8155      1.8630      0.7590
   112      2.4228      2.8075      0.5733
   167      1.5924      2.1428      0.4351
   223      1.7121      1.7411      0.3286
   278      1.5318      1.6931      0.2495
   334      1.0538      2.0477      0.1884
   389      1.4143      2.7413      0.1430
   445      1.2087      2.7071      0.1080
   500      1.2714      3.2628      0.0820
==============================================================================
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : checkpoints\ue30NoShare\gnn_noshare_results.json
Method     : GNN-IL-NoShare
Eval mean  : 1.0987
Eval std   : 0.0955
Best train : 1.0069
------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------
batch_size    : 32
buffer_cap    : 5000
cpu           : False
dqn_hidden    : 128
episodes      : 500
eps_decay     : 0.995
eval_episodes : 20
gamma         : 0.9
gnn_hidden    : 64
gnn_out       : 32
log_every     : 50
lr            : 0.001
n_uavs        : 2
n_ues         : 30
save_dir      : checkpoints/ue30NoShare
seed          : 42
steps_per_ep  : 100
target_update : 20
task_prob     : 0.3
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.3801 -> 1.1932
Loss (first -> last)       : 7.5185 -> 2.5257
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 1.0069
Best avg cost episode      : 470
Best loss                  : 1.0596
Best loss episode          : 16
Cost improvement (%)       : 72.76
Rolling avg cost (w=20) : 1.2056
Rolling avg loss (w=20) : 2.3112
------------------------------------------------------------------------------
CHECKPOINT SNAPSHOT
------------------------------------------------------------------------------
Ep      AvgCost     Loss        Eps
------  ----------  ----------  ----------
     1      4.3801      7.5185      1.0000
    56      3.2720      1.6491      0.7590
   112      2.3212      2.0423      0.5733
   167      1.8927      2.5414      0.4351
   223      1.6014      2.3668      0.3286
   278      1.6837      3.5250      0.2495
   334      1.6095      3.6200      0.1884
   389      1.5713      3.8532      0.1430
   445      1.3536      2.7999      0.1080
   500      1.1932      2.5257      0.0820
==============================================================================
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : checkpoints\ue20CTDEGNN\gnn_ctde_results.json
Method     : GNN-CTDE
Eval mean  : 1.7979
Eval std   : 0.0739
Best train : 0.6085
------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------
batch_size    : 32
cpu           : False
dqn_hidden    : 128
episodes      : 500
eps_decay     : 0.995
eval_episodes : 20
gamma         : 0.9
gnn_hidden    : 64
gnn_out       : 32
log_every     : 50
lr            : 0.001
n_uavs        : 2
n_ues         : 20
per_agent_cap : 10000
save_dir      : checkpoints/ue20CTDEGNN
seed          : 42
steps_per_ep  : 100
target_update : 20
task_prob     : 0.3
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.6683 -> 1.7649
Loss (first -> last)       : 3.3251 -> 1.3516
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6085
Best avg cost episode      : 373
Best loss                  : 1.2114
Best loss episode          : 388
Cost improvement (%)       : 51.89
Rolling avg cost (w=20) : 1.7366
Rolling avg loss (w=20) : 1.3755
------------------------------------------------------------------------------
CHECKPOINT SNAPSHOT
------------------------------------------------------------------------------
Ep      AvgCost     Loss        Eps
------  ----------  ----------  ----------
     1      3.6683      3.3251      1.0000
    56      2.4163      1.8890      0.7590
   112      2.5093      1.9718      0.5733
   167      2.0894      1.6085      0.4351
   223      1.5915      1.5981      0.3286
   278      1.8028      1.5187      0.2495
   334      1.4786      1.4208      0.1884
   389      1.8702      1.5031      0.1430
   445      1.8460      1.3969      0.1080
   500      1.7649      1.3516      0.0820
==============================================================================
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : checkpoints\ue30CTDEGNN\gnn_ctde_results.json
Method     : GNN-CTDE
Eval mean  : 2.1109
Eval std   : 0.0437
Best train : 1.8288
------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------
batch_size    : 32
cpu           : False
dqn_hidden    : 128
episodes      : 500
eps_decay     : 0.995
eval_episodes : 20
gamma         : 0.9
gnn_hidden    : 64
gnn_out       : 32
log_every     : 50
lr            : 0.001
n_uavs        : 2
n_ues         : 30
per_agent_cap : 10000
save_dir      : checkpoints/ue30CTDEGNN
seed          : 42
steps_per_ep  : 100
target_update : 20
task_prob     : 0.3
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.5096 -> 2.3552
Loss (first -> last)       : 2.3684 -> 1.2550
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 1.8288
Best avg cost episode      : 478
Best loss                  : 1.0895
Best loss episode          : 7
Cost improvement (%)       : 47.77
Rolling avg cost (w=20) : 2.3988
Rolling avg loss (w=20) : 1.3460
------------------------------------------------------------------------------
CHECKPOINT SNAPSHOT
------------------------------------------------------------------------------
Ep      AvgCost     Loss        Eps
------  ----------  ----------  ----------
     1      4.5096      2.3684      1.0000
    56      3.2126      1.3788      0.7590
   112      2.3141      1.6391      0.5733
   167      2.3171      1.6980      0.4351
   223      2.6206      1.5758      0.3286
   278      2.6767      1.3198      0.2495
   334      2.9023      1.5487      0.1884
   389      2.6563      1.4241      0.1430
   445      2.4833      1.3740      0.1080
   500      2.3552      1.2550      0.0820
==============================================================================
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : checkpoints\ue20CTDEWithB1000E\ctde_results.json
Method     : CTDE
Eval mean  : 0.8038
Eval std   : 0.0451
Best train : 0.6928
------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------
batch_size    : 32
cpu           : False
episodes      : 1000
eps_decay     : 0.995
eval_episodes : 20
gamma         : 0.9
hidden        : 128
log_every     : 100
lr            : 0.001
n_uavs        : 2
n_ues         : 20
per_agent_cap : 10000
save_dir      : checkpoints/ue20CTDEWithB1000E
seed          : 42
steps_per_ep  : 100
target_update : 20
task_prob     : 0.3
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 1000
Avg cost (first -> last)   : 3.6683 -> 0.7739
Loss (first -> last)       : 2.9523 -> 1.2586
Epsilon (first -> last)    : 1.0000 -> 0.0500
Best avg cost              : 0.6928
Best avg cost episode      : 970
Best loss                  : 1.0854
Best loss episode          : 984
Cost improvement (%)       : 78.90
Rolling avg cost (w=20) : 0.8678
Rolling avg loss (w=20) : 1.2089
------------------------------------------------------------------------------
CHECKPOINT SNAPSHOT
------------------------------------------------------------------------------
Ep      AvgCost     Loss        Eps
------  ----------  ----------  ----------
     1      3.6683      2.9523      1.0000
   112      2.3713      1.5395      0.5733
   223      1.4715      1.6981      0.3286
   334      1.3862      1.7145      0.1884
   445      1.0878      1.6599      0.1080
   556      1.1125      1.5437      0.0619
   667      0.9824      1.3572      0.0500
   778      1.0494      1.4367      0.0500
   889      0.9097      1.4998      0.0500
  1000      0.7739      1.2586      0.0500
==============================================================================

===== ue5 / seed42 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue5\seed42\standard\il_results.json
Method     : IL
Eval mean  : 0.5984
Eval std   : 0.0315
Best train : 0.4855
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 0.6976 -> 0.6046
Loss (first -> last)       : 0.7724 -> 0.4241
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.4855
Best avg cost episode      : 440
Best loss                  : 0.2634
Best loss episode          : 3
Cost improvement (%)       : 13.34
Rolling avg cost (w=20) : 0.5944
Rolling avg loss (w=20) : 0.4474
==============================================================================

===== ue5 / seed42 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue5\seed42\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.6146
Eval std   : 0.0330
Best train : 0.4700
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 0.6919 -> 0.5881
Loss (first -> last)       : 0.4570 -> 0.2632
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.4700
Best avg cost episode      : 13
Best loss                  : 0.2424
Best loss episode          : 493
Cost improvement (%)       : 14.99
Rolling avg cost (w=20) : 0.5749
Rolling avg loss (w=20) : 0.2763
==============================================================================

===== ue5 / seed52 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue5\seed52\standard\il_results.json
Method     : IL
Eval mean  : 0.6006
Eval std   : 0.0704
Best train : 0.4750
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 0.5649 -> 0.6078
Loss (first -> last)       : 2.0700 -> 0.4380
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.4750
Best avg cost episode      : 271
Best loss                  : 0.2903
Best loss episode          : 3
Cost improvement (%)       : -7.59
Rolling avg cost (w=20) : 0.6054
Rolling avg loss (w=20) : 0.4547
==============================================================================

===== ue5 / seed52 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue5\seed52\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.5724
Eval std   : 0.0641
Best train : 0.4266
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 0.5384 -> 0.5902
Loss (first -> last)       : 0.2868 -> 0.2363
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.4266
Best avg cost episode      : 342
Best loss                  : 0.2228
Best loss episode          : 486
Cost improvement (%)       : -9.62
Rolling avg cost (w=20) : 0.5867
Rolling avg loss (w=20) : 0.2562
==============================================================================

===== ue5 / seed62 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue5\seed62\standard\il_results.json
Method     : IL
Eval mean  : 0.6482
Eval std   : 0.0581
Best train : 0.4734
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 0.5683 -> 0.6444
Loss (first -> last)       : 1.1165 -> 0.4471
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.4734
Best avg cost episode      : 43
Best loss                  : 0.2170
Best loss episode          : 2
Cost improvement (%)       : -13.40
Rolling avg cost (w=20) : 0.6244
Rolling avg loss (w=20) : 0.4693
==============================================================================

===== ue5 / seed62 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue5\seed62\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.5906
Eval std   : 0.0654
Best train : 0.4681
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 0.5638 -> 0.5791
Loss (first -> last)       : 0.2662 -> 0.2903
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.4681
Best avg cost episode      : 497
Best loss                  : 0.2294
Best loss episode          : 2
Cost improvement (%)       : -2.72
Rolling avg cost (w=20) : 0.5713
Rolling avg loss (w=20) : 0.3052
==============================================================================

===== ue5 / seed72 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue5\seed72\standard\il_results.json
Method     : IL
Eval mean  : 0.5720
Eval std   : 0.0432
Best train : 0.4616
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 0.6165 -> 0.5813
Loss (first -> last)       : 0.7303 -> 0.4519
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.4616
Best avg cost episode      : 439
Best loss                  : 0.2288
Best loss episode          : 3
Cost improvement (%)       : 5.71
Rolling avg cost (w=20) : 0.5562
Rolling avg loss (w=20) : 0.4381
==============================================================================

===== ue5 / seed72 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue5\seed72\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.5872
Eval std   : 0.0522
Best train : 0.4656
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 0.6112 -> 0.5812
Loss (first -> last)       : 0.3060 -> 0.2381
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.4656
Best avg cost episode      : 286
Best loss                  : 0.2016
Best loss episode          : 394
Cost improvement (%)       : 4.92
Rolling avg cost (w=20) : 0.5491
Rolling avg loss (w=20) : 0.2924
==============================================================================

===== ue5 / seed82 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue5\seed82\standard\il_results.json
Method     : IL
Eval mean  : 0.5981
Eval std   : 0.0512
Best train : 0.4594
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 0.5841 -> 0.6011
Loss (first -> last)       : 1.5941 -> 0.4223
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.4594
Best avg cost episode      : 249
Best loss                  : 0.1861
Best loss episode          : 3
Cost improvement (%)       : -2.91
Rolling avg cost (w=20) : 0.6004
Rolling avg loss (w=20) : 0.4365
==============================================================================

===== ue5 / seed82 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue5\seed82\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.5792
Eval std   : 0.0533
Best train : 0.4359
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 0.6214 -> 0.5492
Loss (first -> last)       : 0.3446 -> 0.3441
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.4359
Best avg cost episode      : 249
Best loss                  : 0.2655
Best loss episode          : 331
Cost improvement (%)       : 11.62
Rolling avg cost (w=20) : 0.6980
Rolling avg loss (w=20) : 0.3458
==============================================================================

===== ue10 / seed42 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue10\seed42\standard\il_results.json
Method     : IL
Eval mean  : 0.7317
Eval std   : 0.1068
Best train : 0.5285
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 1.5235 -> 0.6888
Loss (first -> last)       : 2.0661 -> 0.9989
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5285
Best avg cost episode      : 430
Best loss                  : 0.9860
Best loss episode          : 497
Cost improvement (%)       : 54.79
Rolling avg cost (w=20) : 0.7332
Rolling avg loss (w=20) : 1.0500
==============================================================================

===== ue10 / seed42 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue10\seed42\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.6252
Eval std   : 0.0455
Best train : 0.4698
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 1.6355 -> 0.5356
Loss (first -> last)       : 2.1106 -> 0.3792
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.4698
Best avg cost episode      : 430
Best loss                  : 0.2396
Best loss episode          : 348
Cost improvement (%)       : 67.25
Rolling avg cost (w=20) : 0.6222
Rolling avg loss (w=20) : 0.3569
==============================================================================

===== ue10 / seed52 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue10\seed52\standard\il_results.json
Method     : IL
Eval mean  : 0.9584
Eval std   : 0.1750
Best train : 0.5860
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 0.6894 -> 0.6576
Loss (first -> last)       : 1.1295 -> 1.3398
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5860
Best avg cost episode      : 363
Best loss                  : 0.9769
Best loss episode          : 2
Cost improvement (%)       : 4.62
Rolling avg cost (w=20) : 0.9146
Rolling avg loss (w=20) : 1.3382
==============================================================================

===== ue10 / seed52 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue10\seed52\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 1.1660
Eval std   : 0.0976
Best train : 0.5316
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 0.8459 -> 0.7383
Loss (first -> last)       : 0.4366 -> 2.3409
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5316
Best avg cost episode      : 283
Best loss                  : 0.1876
Best loss episode          : 310
Cost improvement (%)       : 12.72
Rolling avg cost (w=20) : 0.9852
Rolling avg loss (w=20) : 2.3981
==============================================================================

===== ue10 / seed62 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue10\seed62\standard\il_results.json
Method     : IL
Eval mean  : 0.7981
Eval std   : 0.1194
Best train : 0.5534
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 1.1306 -> 0.6347
Loss (first -> last)       : 2.9535 -> 1.1314
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5534
Best avg cost episode      : 332
Best loss                  : 1.0627
Best loss episode          : 497
Cost improvement (%)       : 43.86
Rolling avg cost (w=20) : 0.8707
Rolling avg loss (w=20) : 1.1178
==============================================================================

===== ue10 / seed62 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue10\seed62\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.5508
Eval std   : 0.0265
Best train : 0.4830
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 1.3204 -> 0.5364
Loss (first -> last)       : 2.1590 -> 0.3194
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.4830
Best avg cost episode      : 388
Best loss                  : 0.2253
Best loss episode          : 495
Cost improvement (%)       : 59.38
Rolling avg cost (w=20) : 0.5869
Rolling avg loss (w=20) : 0.2835
==============================================================================

===== ue10 / seed72 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue10\seed72\standard\il_results.json
Method     : IL
Eval mean  : 0.8137
Eval std   : 0.0862
Best train : 0.5277
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 1.7984 -> 1.0422
Loss (first -> last)       : 4.9422 -> 1.0957
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5277
Best avg cost episode      : 457
Best loss                  : 1.0161
Best loss episode          : 477
Cost improvement (%)       : 42.05
Rolling avg cost (w=20) : 0.8265
Rolling avg loss (w=20) : 1.0918
==============================================================================

===== ue10 / seed72 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue10\seed72\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.5805
Eval std   : 0.0259
Best train : 0.4714
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 1.7641 -> 0.5885
Loss (first -> last)       : 2.3856 -> 0.2869
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.4714
Best avg cost episode      : 372
Best loss                  : 0.2016
Best loss episode          : 461
Cost improvement (%)       : 66.64
Rolling avg cost (w=20) : 0.5722
Rolling avg loss (w=20) : 0.2730
==============================================================================

===== ue10 / seed82 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue10\seed82\standard\il_results.json
Method     : IL
Eval mean  : 0.7018
Eval std   : 0.1052
Best train : 0.5056
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 1.9680 -> 0.6266
Loss (first -> last)       : 4.5537 -> 1.0278
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5056
Best avg cost episode      : 325
Best loss                  : 0.9763
Best loss episode          : 481
Cost improvement (%)       : 68.16
Rolling avg cost (w=20) : 0.7167
Rolling avg loss (w=20) : 1.0446
==============================================================================

===== ue10 / seed82 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue10\seed82\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.8621
Eval std   : 0.0710
Best train : 0.4608
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 1.7402 -> 0.8396
Loss (first -> last)       : 2.7431 -> 3.1508
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.4608
Best avg cost episode      : 325
Best loss                  : 0.2726
Best loss episode          : 298
Cost improvement (%)       : 51.75
Rolling avg cost (w=20) : 1.0770
Rolling avg loss (w=20) : 3.5423
==============================================================================

===== ue20 / seed42 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue20\seed42\standard\il_results.json
Method     : IL
Eval mean  : 0.9613
Eval std   : 0.1284
Best train : 0.6318
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.7767 -> 0.7854
Loss (first -> last)       : 9.2823 -> 1.8761
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6318
Best avg cost episode      : 228
Best loss                  : 1.6322
Best loss episode          : 422
Cost improvement (%)       : 79.21
Rolling avg cost (w=20) : 1.1415
Rolling avg loss (w=20) : 1.8604
==============================================================================

===== ue20 / seed42 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue20\seed42\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 1.5061
Eval std   : 0.0741
Best train : 0.6223
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.6087 -> 1.7246
Loss (first -> last)       : 3.4774 -> 4.9071
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6223
Best avg cost episode      : 297
Best loss                  : 1.2528
Best loss episode          : 46
Cost improvement (%)       : 52.21
Rolling avg cost (w=20) : 1.9782
Rolling avg loss (w=20) : 6.2647
==============================================================================

===== ue20 / seed52 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue20\seed52\standard\il_results.json
Method     : IL
Eval mean  : 0.9163
Eval std   : 0.1598
Best train : 0.6225
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.2003 -> 1.2466
Loss (first -> last)       : 8.3511 -> 1.8765
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6225
Best avg cost episode      : 180
Best loss                  : 1.6126
Best loss episode          : 427
Cost improvement (%)       : 61.05
Rolling avg cost (w=20) : 1.0692
Rolling avg loss (w=20) : 1.8408
==============================================================================

===== ue20 / seed52 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue20\seed52\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 1.2366
Eval std   : 0.0591
Best train : 0.7047
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.0901 -> 1.3572
Loss (first -> last)       : 3.7240 -> 1.1095
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.7047
Best avg cost episode      : 488
Best loss                  : 0.8397
Best loss episode          : 490
Cost improvement (%)       : 56.08
Rolling avg cost (w=20) : 1.0670
Rolling avg loss (w=20) : 1.0019
==============================================================================

===== ue20 / seed62 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue20\seed62\standard\il_results.json
Method     : IL
Eval mean  : 0.8329
Eval std   : 0.1309
Best train : 0.6421
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.2654 -> 1.0510
Loss (first -> last)       : 9.3961 -> 1.8566
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6421
Best avg cost episode      : 187
Best loss                  : 1.6938
Best loss episode          : 416
Cost improvement (%)       : 67.81
Rolling avg cost (w=20) : 1.1127
Rolling avg loss (w=20) : 1.8585
==============================================================================

===== ue20 / seed62 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue20\seed62\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.5988
Eval std   : 0.0293
Best train : 0.5685
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.5929 -> 0.6144
Loss (first -> last)       : 3.4906 -> 0.5740
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5685
Best avg cost episode      : 482
Best loss                  : 0.4575
Best loss episode          : 483
Cost improvement (%)       : 82.90
Rolling avg cost (w=20) : 0.6714
Rolling avg loss (w=20) : 0.5262
==============================================================================

===== ue20 / seed72 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue20\seed72\standard\il_results.json
Method     : IL
Eval mean  : 0.9024
Eval std   : 0.1190
Best train : 0.5746
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.6559 -> 1.2242
Loss (first -> last)       : 8.8649 -> 1.8910
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5746
Best avg cost episode      : 206
Best loss                  : 1.6741
Best loss episode          : 406
Cost improvement (%)       : 66.51
Rolling avg cost (w=20) : 1.1059
Rolling avg loss (w=20) : 1.8281
==============================================================================

===== ue20 / seed72 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue20\seed72\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.5998
Eval std   : 0.0240
Best train : 0.5455
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.8238 -> 0.6146
Loss (first -> last)       : 3.7734 -> 0.5792
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5455
Best avg cost episode      : 390
Best loss                  : 0.2704
Best loss episode          : 440
Cost improvement (%)       : 83.93
Rolling avg cost (w=20) : 0.6589
Rolling avg loss (w=20) : 0.6055
==============================================================================

===== ue20 / seed82 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue20\seed82\standard\il_results.json
Method     : IL
Eval mean  : 0.9831
Eval std   : 0.0934
Best train : 0.6230
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.8487 -> 1.0056
Loss (first -> last)       : 9.7893 -> 1.8138
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6230
Best avg cost episode      : 145
Best loss                  : 1.5439
Best loss episode          : 407
Cost improvement (%)       : 73.87
Rolling avg cost (w=20) : 1.1865
Rolling avg loss (w=20) : 1.7624
==============================================================================

===== ue20 / seed82 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue20\seed82\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.6193
Eval std   : 0.0285
Best train : 0.6247
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.6566 -> 0.6890
Loss (first -> last)       : 3.3332 -> 1.1914
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6247
Best avg cost episode      : 497
Best loss                  : 1.0564
Best loss episode          : 498
Cost improvement (%)       : 81.16
Rolling avg cost (w=20) : 0.9464
Rolling avg loss (w=20) : 2.0055
==============================================================================

===== ue30 / seed42 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue30\seed42\standard\il_results.json
Method     : IL
Eval mean  : 1.3903
Eval std   : 0.1189
Best train : 0.5999
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.4364 -> 1.5695
Loss (first -> last)       : 9.9416 -> 1.3490
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5999
Best avg cost episode      : 400
Best loss                  : 0.9877
Best loss episode          : 18
Cost improvement (%)       : 64.62
Rolling avg cost (w=20) : 1.5366
Rolling avg loss (w=20) : 1.3592
==============================================================================

===== ue30 / seed42 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue30\seed42\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.6012
Eval std   : 0.0299
Best train : 0.5820
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.3148 -> 0.6309
Loss (first -> last)       : 2.4554 -> 0.6415
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5820
Best avg cost episode      : 491
Best loss                  : 0.6268
Best loss episode          : 498
Cost improvement (%)       : 85.38
Rolling avg cost (w=20) : 0.6911
Rolling avg loss (w=20) : 0.7392
==============================================================================

===== ue30 / seed52 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue30\seed52\standard\il_results.json
Method     : IL
Eval mean  : 1.3134
Eval std   : 0.1466
Best train : 0.6132
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.9438 -> 1.5833
Loss (first -> last)       : 11.1181 -> 1.5039
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6132
Best avg cost episode      : 348
Best loss                  : 0.9481
Best loss episode          : 12
Cost improvement (%)       : 59.85
Rolling avg cost (w=20) : 1.4453
Rolling avg loss (w=20) : 1.5369
==============================================================================

===== ue30 / seed52 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue30\seed52\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.7723
Eval std   : 0.0301
Best train : 0.6577
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.8677 -> 0.7627
Loss (first -> last)       : 2.7095 -> 1.0482
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6577
Best avg cost episode      : 476
Best loss                  : 0.9569
Best loss episode          : 479
Cost improvement (%)       : 80.28
Rolling avg cost (w=20) : 0.8069
Rolling avg loss (w=20) : 1.0319
==============================================================================

===== ue30 / seed62 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue30\seed62\standard\il_results.json
Method     : IL
Eval mean  : 1.2132
Eval std   : 0.1232
Best train : 0.6294
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.0903 -> 1.5963
Loss (first -> last)       : 10.4689 -> 1.5703
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6294
Best avg cost episode      : 280
Best loss                  : 0.9676
Best loss episode          : 15
Cost improvement (%)       : 60.97
Rolling avg cost (w=20) : 1.4685
Rolling avg loss (w=20) : 1.5600
==============================================================================

===== ue30 / seed62 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue30\seed62\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.6567
Eval std   : 0.0336
Best train : 0.6348
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.0470 -> 0.7411
Loss (first -> last)       : 2.7063 -> 0.8131
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6348
Best avg cost episode      : 498
Best loss                  : 0.8131
Best loss episode          : 500
Cost improvement (%)       : 81.69
Rolling avg cost (w=20) : 0.7394
Rolling avg loss (w=20) : 0.8913
==============================================================================

===== ue30 / seed72 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue30\seed72\standard\il_results.json
Method     : IL
Eval mean  : 1.3209
Eval std   : 0.0989
Best train : 0.6131
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.0652 -> 1.7271
Loss (first -> last)       : 10.6451 -> 1.5971
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6131
Best avg cost episode      : 390
Best loss                  : 0.9725
Best loss episode          : 22
Cost improvement (%)       : 57.52
Rolling avg cost (w=20) : 1.5462
Rolling avg loss (w=20) : 1.5489
==============================================================================

===== ue30 / seed72 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue30\seed72\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.6330
Eval std   : 0.0241
Best train : 0.6457
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.0526 -> 0.7506
Loss (first -> last)       : 2.7224 -> 0.9294
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6457
Best avg cost episode      : 432
Best loss                  : 0.6332
Best loss episode          : 449
Cost improvement (%)       : 81.48
Rolling avg cost (w=20) : 0.8491
Rolling avg loss (w=20) : 0.8624
==============================================================================

===== ue30 / seed82 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue30\seed82\standard\il_results.json
Method     : IL
Eval mean  : 1.3061
Eval std   : 0.0770
Best train : 0.6114
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.0328 -> 1.6886
Loss (first -> last)       : 10.6822 -> 1.5317
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6114
Best avg cost episode      : 415
Best loss                  : 0.9495
Best loss episode          : 11
Cost improvement (%)       : 58.13
Rolling avg cost (w=20) : 1.5666
Rolling avg loss (w=20) : 1.4978
==============================================================================

===== ue30 / seed82 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue30\seed82\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 1.1931
Eval std   : 0.0553
Best train : 1.0578
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.0388 -> 1.3659
Loss (first -> last)       : 2.6413 -> 1.5940
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 1.0578
Best avg cost episode      : 426
Best loss                  : 1.0156
Best loss episode          : 59
Cost improvement (%)       : 66.18
Rolling avg cost (w=20) : 1.3069
Rolling avg loss (w=20) : 1.5264
==============================================================================

===== ue40 / seed42 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue40\seed42\standard\il_results.json
Method     : IL
Eval mean  : 1.0866
Eval std   : 0.0757
Best train : 0.6242
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.1362 -> 1.0451
Loss (first -> last)       : 10.8757 -> 1.5531
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6242
Best avg cost episode      : 449
Best loss                  : 0.7095
Best loss episode          : 15
Cost improvement (%)       : 74.73
Rolling avg cost (w=20) : 1.0635
Rolling avg loss (w=20) : 1.6183
==============================================================================

===== ue40 / seed42 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue40\seed42\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.8869
Eval std   : 0.0325
Best train : 0.6407
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.2727 -> 0.9071
Loss (first -> last)       : 2.3601 -> 0.9688
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6407
Best avg cost episode      : 431
Best loss                  : 0.8027
Best loss episode          : 459
Cost improvement (%)       : 78.77
Rolling avg cost (w=20) : 0.9290
Rolling avg loss (w=20) : 0.9384
==============================================================================

===== ue40 / seed52 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue40\seed52\standard\il_results.json
Method     : IL
Eval mean  : 0.9377
Eval std   : 0.1003
Best train : 0.6121
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.0842 -> 1.1406
Loss (first -> last)       : 11.0310 -> 1.6280
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6121
Best avg cost episode      : 429
Best loss                  : 0.6778
Best loss episode          : 15
Cost improvement (%)       : 72.07
Rolling avg cost (w=20) : 1.0613
Rolling avg loss (w=20) : 1.6679
==============================================================================

===== ue40 / seed52 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue40\seed52\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.7812
Eval std   : 0.0334
Best train : 0.6517
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.0946 -> 0.7725
Loss (first -> last)       : 2.4896 -> 0.7413
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6517
Best avg cost episode      : 460
Best loss                  : 0.6672
Best loss episode          : 494
Cost improvement (%)       : 81.13
Rolling avg cost (w=20) : 0.7999
Rolling avg loss (w=20) : 0.7132
==============================================================================

===== ue40 / seed62 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue40\seed62\standard\il_results.json
Method     : IL
Eval mean  : 1.0552
Eval std   : 0.0783
Best train : 0.6307
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.2582 -> 1.1186
Loss (first -> last)       : 10.6846 -> 1.6214
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6307
Best avg cost episode      : 409
Best loss                  : 0.6952
Best loss episode          : 20
Cost improvement (%)       : 73.73
Rolling avg cost (w=20) : 1.1537
Rolling avg loss (w=20) : 1.6444
==============================================================================

===== ue40 / seed62 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue40\seed62\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.8000
Eval std   : 0.0332
Best train : 0.5948
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.2854 -> 0.9017
Loss (first -> last)       : 2.6392 -> 0.4086
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5948
Best avg cost episode      : 477
Best loss                  : 0.3404
Best loss episode          : 497
Cost improvement (%)       : 78.96
Rolling avg cost (w=20) : 0.7173
Rolling avg loss (w=20) : 0.4165
==============================================================================

===== ue40 / seed72 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue40\seed72\standard\il_results.json
Method     : IL
Eval mean  : 1.0631
Eval std   : 0.0772
Best train : 0.6308
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.9381 -> 1.2888
Loss (first -> last)       : 11.1839 -> 1.5922
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6308
Best avg cost episode      : 456
Best loss                  : 0.6955
Best loss episode          : 16
Cost improvement (%)       : 67.27
Rolling avg cost (w=20) : 1.0246
Rolling avg loss (w=20) : 1.6415
==============================================================================

===== ue40 / seed72 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue40\seed72\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.6385
Eval std   : 0.0198
Best train : 0.6339
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.9340 -> 0.7600
Loss (first -> last)       : 2.5071 -> 0.7818
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6339
Best avg cost episode      : 477
Best loss                  : 0.7058
Best loss episode          : 487
Cost improvement (%)       : 80.68
Rolling avg cost (w=20) : 0.7084
Rolling avg loss (w=20) : 0.7934
==============================================================================

===== ue40 / seed82 / standard =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue40\seed82\standard\il_results.json
Method     : IL
Eval mean  : 1.0325
Eval std   : 0.1270
Best train : 0.6256
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.9784 -> 1.1997
Loss (first -> last)       : 9.7361 -> 1.6199
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6256
Best avg cost episode      : 454
Best loss                  : 0.7228
Best loss episode          : 22
Cost improvement (%)       : 69.84
Rolling avg cost (w=20) : 1.1127
Rolling avg loss (w=20) : 1.6464
==============================================================================

===== ue40 / seed82 / gnn =====
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : Seed Results\NTN_MEC_Ablation\ue40\seed82\gnn\gnn_il_results.json
Method     : GNN-IL
Eval mean  : 0.6903
Eval std   : 0.0354
Best train : 0.6501
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 4.0140 -> 0.8123
Loss (first -> last)       : 2.7032 -> 0.7960
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.6501
Best avg cost episode      : 474
Best loss                  : 0.7689
Best loss episode          : 54
Cost improvement (%)       : 79.76
Rolling avg cost (w=20) : 0.7816
Rolling avg loss (w=20) : 0.8359
==============================================================================

(rlProject) PS D:\Projects\RL Project> python trainGnnGat.py --episodes 500 --log_every 50 --n_ues 10 --save_dir checkpoints/ue10GnnGatect> 
Device: cuda

Training GAT-IL — 10 UEs, 2 UAVs, 500 episodes
GAT: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
      50      1.5966      1.0907   0.782
     100      0.9899      0.8740   0.609
     150      1.0717      0.9690   0.474
     200      0.6248      0.7094   0.369
     250      0.5917      0.5276   0.287
     300      0.6071      0.3407   0.223
     350      0.6399      0.3207   0.174
     400      0.5527      0.3014   0.135
     450      0.6003      0.3548   0.105
     500      0.5367      0.3116   0.082

Eval (20 eps): mean=0.5617  std=0.0255
Saved to checkpoints/ue10GnnGat/

GAT-IL eval  —  0.5617 ± 0.0255
(rlProject) PS D:\Projects\RL Project> python .\view_results.py checkpoints\ue10GnnGat\gat_il_results.json  --detail standard --show-config  
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : checkpoints\ue10GnnGat\gat_il_results.json
Method     : GAT-IL
Eval mean  : 0.5617
Eval std   : 0.0255
Best train : 0.4713
------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------
attn_dim      : 32
batch_size    : 32
buffer_cap    : 5000
cpu           : False
dqn_hidden    : 128
episodes      : 500
eps_decay     : 0.995
eval_episodes : 20
gamma         : 0.9
gnn_hidden    : 64
gnn_out       : 32
log_every     : 50
lr            : 0.001
n_uavs        : 2
n_ues         : 10
save_dir      : checkpoints/ue10GnnGat
seed          : 42
steps_per_ep  : 100
target_update : 20
task_prob     : 0.3
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 1.6355 -> 0.5367
Loss (first -> last)       : 2.1286 -> 0.3116
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.4713
Best avg cost episode      : 430
Best loss                  : 0.2295
Best loss episode          : 498
Cost improvement (%)       : 67.18
Rolling avg cost (w=20) : 0.5769
Rolling avg loss (w=20) : 0.2909
------------------------------------------------------------------------------
CHECKPOINT SNAPSHOT
------------------------------------------------------------------------------
Ep      AvgCost     Loss        Eps
------  ----------  ----------  ----------
     1      1.6355      2.1286      1.0000
    56      0.6604      0.9973      0.7590
   112      0.6620      0.8372      0.5733
   167      0.9540      0.9484      0.4351
   223      0.7094      0.5945      0.3286
   278      0.5802      0.4175      0.2495
   334      0.5151      0.3108      0.1884
   389      0.5709      0.3183      0.1430
   445      0.5779      0.3386      0.1080
   500      0.5367      0.3116      0.0820
==============================================================================

(rlProject) PS D:\Projects\RL Project> python trainGnnGatPlus.py --episodes 500 --log_every 50 --n_ues 10 --save_dir checkpoints/ue10GnnGatPlus
Device: cuda

Training GATPlus-IL — 10 UEs, 2 UAVs, 500 episodes
GATPlus: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
      50      1.1361     13.4443   0.782
     100      1.0655     10.5989   0.609
     150      1.4841     14.2829   0.474
     200      0.8632     10.2326   0.369
     250      0.6889      5.6168   0.287
     300      0.7058      2.8300   0.223
     350      0.6044      0.6453   0.174
     400      0.5659      0.2524   0.135
     450      0.6297      0.4360   0.105
     500      0.5532      0.5485   0.082

Eval (20 eps): mean=0.5615  std=0.0261
Saved to checkpoints/ue10GnnGatPlus/

GATPlus-IL eval  —  0.5615 ± 0.0261
(rlProject) PS D:\Projects\RL Project> python .\view_results.py checkpoints\ue10GnnGatPlus\gatplus_il_results.json  --detail standard --show-config
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : checkpoints\ue10GnnGatPlus\gatplus_il_results.json
Method     : GATPlus-IL
Eval mean  : 0.5615
Eval std   : 0.0261
Best train : 0.4906
------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------
attn_dim       : 32
batch_size     : 32
buffer_cap     : 5000
chi            : 0.5
cpu            : False
dqn_hidden     : 128
episodes       : 500
eps_decay      : 0.995
eval_episodes  : 20
gamma          : 0.9
gnn_hidden     : 64
gnn_out        : 32
log_every      : 50
lr             : 0.001
n_uavs         : 2
n_ues          : 10
per_alpha      : 0.6
per_beta       : 0.4
per_beta_steps : 15000
save_dir       : checkpoints/ue10GnnGatPlus
seed           : 42
steps_per_ep   : 100
target_update  : 20
task_prob      : 0.3
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 1.5182 -> 0.5532
Loss (first -> last)       : 24.8740 -> 0.5485
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.4906
Best avg cost episode      : 352
Best loss                  : 0.1750
Best loss episode          : 389
Cost improvement (%)       : 63.56
Rolling avg cost (w=20) : 0.6067
Rolling avg loss (w=20) : 0.5993
------------------------------------------------------------------------------
CHECKPOINT SNAPSHOT
------------------------------------------------------------------------------
Ep      AvgCost     Loss        Eps
------  ----------  ----------  ----------
     1      1.5182     24.8740      1.0000
    56      0.7200     11.7408      0.7590
   112      1.0926     12.3746      0.5733
   167      1.5177     14.0696      0.4351
   223      0.8329      8.0149      0.3286
   278      0.7237      3.1004      0.2495
   334      0.5157      1.7689      0.1884
   389      0.5871      0.1750      0.1430
   445      0.6162      0.5250      0.1080
   500      0.5532      0.5485      0.0820
==============================================================================

(rlProject) PS D:\Projects\RL Project> python trainGnnGatPlus.py --episodes 500 --log_every 50 --n_ues 20 --save_dir checkpoints/ue20GnnGatPlus    
Device: cuda

Training GATPlus-IL — 20 UEs, 2 UAVs, 500 episodes
GATPlus: hidden=64, out=32  | enriched_dim=43

 Episode     AvgCost        Loss     Eps
------------------------------------------
      50      2.3688     34.7597   0.782
     100      2.1984     31.0899   0.609
     150      1.7873     27.3919   0.474
     200      1.4830     22.1421   0.369
     250      0.7253     13.6431   0.287
     300      0.6988      6.8731   0.223
     350      0.5879      3.0622   0.174
     400      0.5539      0.6816   0.135
     450      0.6132      1.1709   0.105
     500      0.6121      2.8222   0.082

Eval (20 eps): mean=0.6045  std=0.0234
Saved to checkpoints/ue20GnnGatPlus/

GATPlus-IL eval  —  0.6045 ± 0.0234
(rlProject) PS D:\Projects\RL Project> python .\view_results.py checkpoints\ue20GnnGatPlus\gatplus_il_results.json  --detail standard --show-config
==============================================================================
RESULT FILE
------------------------------------------------------------------------------
Path       : checkpoints\ue20GnnGatPlus\gatplus_il_results.json
Method     : GATPlus-IL
Eval mean  : 0.6045
Eval std   : 0.0234
Best train : 0.5485
------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------
attn_dim       : 32
batch_size     : 32
buffer_cap     : 5000
chi            : 0.5
cpu            : False
dqn_hidden     : 128
episodes       : 500
eps_decay      : 0.995
eval_episodes  : 20
gamma          : 0.9
gnn_hidden     : 64
gnn_out        : 32
log_every      : 50
lr             : 0.001
n_uavs         : 2
n_ues          : 20
per_alpha      : 0.6
per_beta       : 0.4
per_beta_steps : 15000
save_dir       : checkpoints/ue20GnnGatPlus
seed           : 42
steps_per_ep   : 100
target_update  : 20
task_prob      : 0.3
------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
Episodes                   : 500
Avg cost (first -> last)   : 3.5895 -> 0.6121
Loss (first -> last)       : 46.3181 -> 2.8222
Epsilon (first -> last)    : 1.0000 -> 0.0820
Best avg cost              : 0.5485
Best avg cost episode      : 373
Best loss                  : 0.4751
Best loss episode          : 398
Cost improvement (%)       : 82.95
Rolling avg cost (w=20) : 0.6499
Rolling avg loss (w=20) : 2.9452
------------------------------------------------------------------------------
CHECKPOINT SNAPSHOT
------------------------------------------------------------------------------
Ep      AvgCost     Loss        Eps
------  ----------  ----------  ----------
     1      3.5895     46.3181      1.0000
    56      2.8639     34.9780      0.7590
   112      2.3238     30.1380      0.5733
   167      1.6706     25.7618      0.4351
   223      1.3591     18.4941      0.3286
   278      1.1685      9.6665      0.2495
   334      0.7029      3.7403      0.1884
   389      0.6277      0.6131      0.1430
   445      0.6907      1.2937      0.1080
   500      0.6121      2.8222      0.0820
==============================================================================
(rlProject) PS D:\Projects\RL Project> 