@echo off
setlocal enabledelayedexpansion

:: List of UE counts (M values)
set ues=40

:: List of Seeds
set seeds=42 52 62

for %%U in (%ues%) do (
    for %%S in (%seeds%) do (
        
        :: This updates the Window Title so you can track progress from the Taskbar
        title UEs: %%U - Seed: %%S

        echo.
        echo ====================================================
        echo RUNNING: UEs=%%U ^| SEED=%%S
        echo ====================================================

        :: 1. Run Standard Train (only if folder doesn't exist)
        if not exist "checkpoints\ue%%U\seed%%S\standard\" (
            echo Training Standard IL...
            python train.py --episodes 500 --log_every 500 --n_ues %%U --seed %%S --save_dir checkpoints/ue%%U/seed%%S/standard
        ) else (
            echo Skipping Standard: Folder already exists.
        )
        
        :: 2. Run GNN Train (only if folder doesn't exist)
        if not exist "checkpoints\ue%%U\seed%%S\gnn\" (
            echo Training GNN IL...
            python trainGnn.py --episodes 500 --log_every 500 --n_ues %%U --seed %%S --save_dir checkpoints/ue%%U/seed%%S/gnn
        ) else (
            echo Skipping GNN: Folder already exists.
        )

    )
)

echo.
echo All experiments completed successfully!
pause
