@echo off
python run_zero_shot_ablation.py --checkpoints_root "Seed Results/NTN_MEC_Ablation" --use_seed_results --seed_id 42 --eval_ms 10 20 30 --eval_episodes 20 --steps_per_ep 100 --il_cycle_weights
pause
