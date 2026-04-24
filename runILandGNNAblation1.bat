@echo off
python train.py --episodes 500 --log_every 500 --n_ues 5 --save_dir checkpoints/ue5
python trainGnn.py --episodes 500 --log_every 500 --n_ues 5 --save_dir checkpoints/ue5
python train.py --episodes 500 --log_every 500 --n_ues 10 --save_dir checkpoints/ue10
python trainGnn.py --episodes 500 --log_every 500 --n_ues 10 --save_dir checkpoints/ue10
python train.py --episodes 500 --log_every 500 --n_ues 20 --save_dir checkpoints/ue20
python trainGnn.py --episodes 500 --log_every 500 --n_ues 20 --save_dir checkpoints/ue20
pause