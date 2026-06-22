1) No LEO satellite right now
2) Smaller scale for now
3) Same underlying physics for channels, queues, delay, energy loss from paper

IL baseline (train_il.py) - M independent Dueling DQN agents, each with its own replay buffer and weights. Agents are completely blind to each other. No inter-agent communication.

GNN-IL (train_gnn_il.py) - One shared BipartiteGNN encoder + shared Dueling DQN across all UEs. The GNN builds a bipartite UE↔UAV graph where UAV nodes carry a congestion feature (fraction of UEs with active queues) that IL agents cannot see. Message passing propagates this global signal back to UE node embeddings before action selection.

Framestack ablation (framestack.py, trainStack.py) — Attempted to add temporal memory to IL via K=4 frame concatenation as a cheaper alternative to GRU. Implemented cleanly as a drop-in wrapper.

GRU — Implemented sequence replay buffer with chunk_len=8. Training diverged completely — loss climbed from 1.29 to 3.42 over 500 episodes, never descended. Eval cost 0.9574 vs vanilla IL's 0.6626.

(single seed)
UEs,  Random,     Local,      IL,               GNN-IL,           IL gain
5,    0.6305,     0.9176,     0.6037 ± 0.031,   0.5719 ± 0.025,   5.3%
10,   1.4585,     0.9513,     0.6626 ± 0.068,   0.5650 ± 0.028,   14.7%
20,   3.6671,     0.9139,     0.8120 ± 0.067,   0.5833 ± 0.021,   28.2%


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

GNN-IL's cost is relatively stable as M grows while IL degrades significantly. The gain widens from ~5% at M=5 to ~28% at M=20 to a dramatic gap at M=30 (IL: 1.338, GNN-IL: 0.660). The interpretation: as more UEs compete for UAVs, coordination becomes more valuable. IL agents see congestion only through their own queue state. GNN agents see the global congestion signal proactively.
at low UE count, UAVs are underloaded so offloading is almost always beneficial - random policy offloads ~66% of the time and gets rewarded. At high UE count, UAVs congest, random offloading often hits an overloaded server and incurs penalty, making local safer

Framestack performing worse than vanilla IL (0.6847 vs 0.6626). The queue state is already encoded in the current observation (local_wait, tran_wait, edge_q). Stacking 4 frames mostly quadruples input dimensionality without adding new information, making the DQN harder to train. This rules out "IL was disadvantaged by lacking temporal memory" as an explanation for GNN-IL's gains. The bottleneck is spatial coordination awareness, not temporal memory.

GNN-IL uses full parameter sharing - one network for all agents. When UE 1 takes a bad action and receives a penalty, the gradient flows backward through the DQN and into the GNN. Because the GNN used message passing to build UE 1's embedding through UAV nodes, the gradient physically flows from UE 1 through the UAV node and updates the weights governing how UAV nodes aggregate information from all UEs including UE 2. UE 2's representation is immediately upgraded by UE 1's mistake. Vanilla IL has M completely separate networks - UE 2 learns absolutely nothing from UE 1's experience.
This means GNN-IL's advantage comes from two separable mechanisms:
    Richer observations (congestion signal propagated via message passing)
    Implicit gradient coupling through shared weights and graph structure

GNN-IL is not CTDE
UEs are the ones actually choosing: local / UAV / satellite.
Central controller is the one that collects experience from all UEs, trains the neural network, and sends updated weights back.
Each UE chooses one action, and produces a sequence
This sequence is sent to the CTDE and stored in the replay buffer
The controller samples mini-batches from the replay buffer and updates the primary dueling DQN weights. It also periodically updates the target network.
After training, the updated weights are copied to the UEs, so each UE can do decentralized execution using the learned policy.
UE 2 affects UE 1 indirectly through:
    the load history (GRU),
    the training data the central controller sees,
    the shared weights after training.

In GNN, UE 1’s input is not just its own features. The GNN does message passing:
    UE features go to UAVs,
    UAV representations go back to UEs,
    then each UE gets an enriched embedding before the DQN chooses an action.

UE 1’s embedding can change right away, because UE 2’s information is part of the graph message passing.
Training is also faster, and converges faster than a GRU
GNN-IL requires the full graph at execution time too, while CTDE's centralized component is training-only


Next
    Full 5-seed sweep at M=5,10,20,30,40 (running on Colab now)
    CTDE implementation (vanilla, then GNN-CTDE)
    Transfer experiment: train on M=10, eval on M=15/20 without retraining
    Parameter sharing ablation: GNN-IL with separate per-agent weights (no parameter sharing), to isolate gradient coupling contribution from richer observations (each agent sees other's state during)
    LEO satellite

(rlProject) PS D:\Projects\RL Project> python trainGnnStack.py --episodes 500 --log_every 50     
Device: cuda

Training GNN-IL+Stack — 10 UEs, 2 UAVs, 500 episodes  |  K=4  stacked_dim=172

 Episode     AvgCost        Loss     Eps
------------------------------------------
      50      0.9216      0.9144   0.782
     100      0.7330      0.7101   0.609
     150      0.8873      0.5955   0.474
     200      0.6091      0.4633   0.369
     250      0.6995      0.4732   0.287
     300      0.6088      0.4463   0.223
     350      0.6765      0.4405   0.174
     400      0.5804      0.4117   0.135
     450      0.6156      0.3445   0.105
     500      0.5426      0.3412   0.082

Eval (20 eps): mean=0.5464  std=0.0260
Saved to checkpoints/

GNN-IL+Stack eval  —  0.5464 ± 0.0260
