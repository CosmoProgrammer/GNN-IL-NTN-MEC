@echo off
python trainGnnNoShare.py --n_ues 10 --episodes 500 --seed 42 --save_dir checkpoints/ue10NoShare
python trainGnnNoShare.py --n_ues 20 --episodes 500 --seed 42 --save_dir checkpoints/ue20NoShare
python trainGnnNoShare.py --n_ues 30 --episodes 500 --seed 42 --save_dir checkpoints/ue30NoShare
python trainCtdeGNN.py --episodes 500 --log_every 50 --n_ues 20 --save_dir checkpoints/ue20CTDEGNN
python trainCtdeGNN.py --episodes 500 --log_every 50 --n_ues 30 --save_dir checkpoints/ue30CTDEGNN
python trainCtde.py --episodes 1000 --log_every 100 --n_ues 20 --save_dir checkpoints/ue20CTDEWithB1000E  
pause