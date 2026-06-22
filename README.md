# Thinking Globally, Acting Locally: Graph Neural Networks for Decentralised MEC Offloading

**Authors:** Anirudh M  
**Institution:** BITS Pilani, Hyderabad Campus  
📄 [Read the paper](https://github.com/CosmoProgrammer/GNN-IL-NTN-MEC/blob/main/RLPaperFinalDraft.pdf)

---

## Overview

Multi-agent computation offloading in Non-Terrestrial Network (NTN) empowered Multi-access Edge Computing (MEC) systems requires agents to coordinate shared UAV resources under partial observability. Standard decentralised approaches (Independent Learning) struggle with this coordination; centralised methods (CTDE) impose heavy infrastructure overhead.

We propose **GNN-IL**: a bipartite weighted GraphSAGE encoder combined with a shared Dueling DQN, trained in a fully decentralised manner via parameter sharing across all agents - no central controller at any stage.

---

## Architecture

At each time slot, the system state is represented as a bipartite graph G = (V_UE ∪ V_UAV, E):

- **UE nodes** carry local observation vectors (position, queue waits, task size)
- **UAV nodes** carry position and normalised active queue count B_n(i)/M
- **Edges** are weighted by normalised channel gain

A two-layer weighted GraphSAGE propagates UAV congestion signals back to UE embeddings. Each UE's enriched embedding is concatenated with its raw observation via a skip connection and fed to a shared Dueling DQN. All parameters - GNN encoder, DQN, and replay buffer - are shared across all M agents.

---

## Results

### Scaling Performance (5 seeds: 42, 52, 62, 72, 82)

| UEs (M) | IL mean ± std | GNN-IL mean ± std | Gain | GNN-IL wins |
|:-------:|:-------------:|:-----------------:|:----:|:-----------:|
| 5       | 0.6035 ± 0.025 | 0.5888 ± 0.014   | 2.4% | 3/5         |
| 10      | 0.8007 ± 0.089 | 0.7569 ± 0.232   | 5.5% | 3/5         |
| 20      | 0.9192 ± 0.052 | **0.5998** (median) | 34.8%* | 3/5    |
| 30      | 1.3088 ± 0.057 | 0.7713 ± 0.219   | **41.1%** | 5/5  |
| 40      | 1.0350 ± 0.052 | 0.7594 ± 0.087   | 26.6% | 5/5        |

*M=20 reports median due to bimodal convergence; see paper Section 5.2.*

### Baseline Comparison (M=10, seed 42)

| Method | Eval Mean |
|:-------|:---------:|
| Random | 1.4585 |
| Local | 0.9513 |
| CTDE (w/o B_n) | 0.9335 |
| IL | 0.6626 |
| CTDE (w/ B_n) | 0.6286 |
| **GNN-IL** | **0.5650** |

GNN-IL outperforms CTDE by **10.1%** at M=10 without requiring a central controller.

---

## Key Findings

**1. The bottleneck in IL is spatial, not temporal.**  
Frame stacking (K=4) *hurts* IL by 3.3%, and a GRU encoder diverges entirely. Temporal memory does not address IL's fundamental limitation.

**2. Graph structure and parameter sharing are both necessary.**  
GNN-IL-NoShare - which provides GNN-enriched observations but removes parameter sharing - performs *worse than vanilla IL* at M=10 and M=20. Richer observations alone are not enough; gradient coupling through shared weights is the operative mechanism.

**3. The same information has different utility depending on architecture.**  
Adding B_n(i) to CTDE's observation recovers 32.7% in performance; the same feature marginally *hurts* IL (+3.8%); GNN-IL integrates it naturally through message passing. Information availability ≠ information utility.

**4. GNN-IL becomes more reliable as the coordination problem grows harder.**  
GNN-IL exhibits bimodal convergence at M=20 but converges on 4/5 seeds at M=30 and M=40, with tightening standard deviation (0.087 at M=40). IL degrades consistently with M.

---

## Repository Structure

```
├── agent.py              # Dueling DQN agent
├── gnnAgent.py           # GNN-IL agent (GraphSAGE + shared DQN)
├── gnnGatAgent.py        # GAT-IL variant
├── gnnGatPlusAgent.py    # GATPlus-IL variant (PER + auxiliary loss)
├── ctdeAgent.py          # CTDE baseline
├── ctdeGnnAgent.py       # GNN-CTDE variant
├── env.py                # NTN-MEC environment
├── framestack.py         # Frame stacking wrapper (ablation)
├── train_*.py            # Training scripts for each method
├── view_results.py       # Results visualisation
├── figures/              # Paper figures
├── Seed Results/         # Raw per-seed evaluation data
└── RLPaperFinalDraft.pdf # Full paper
```

---

## Conscious Simplifications vs. Base Paper

This work extends Fatima et al. (GLOBECOM 2024) with the following deliberate simplifications:
- N=2 UAVs (base paper: N=4); LEO satellite omitted to isolate UAV coordination dynamics
- 500 training episodes (base paper: 1000)
- No GRU module in CTDE (removed for a fair IL vs. CTDE comparison)

These choices are analysed in Section 6.4 of the paper.

---

## Citation

```
Anirudh M "Thinking Globally, Acting Locally: Graph Neural Networks for Decentralised MEC Offloading." Course project report, BITS Pilani Hyderabad Campus, 2026.
```
