@echo off
python .\view_results.py checkpoints\ue10NoShare\gnn_noshare_results.json --detail standard --show-config
python .\view_results.py checkpoints\ue20NoShare\gnn_noshare_results.json --detail standard --show-config
python .\view_results.py checkpoints\ue30NoShare\gnn_noshare_results.json --detail standard --show-config
python .\view_results.py checkpoints\ue20CTDEGNN\gnn_ctde_results.json --detail standard --show-config
python .\view_results.py checkpoints\ue30CTDEGNN\gnn_ctde_results.json --detail standard --show-config
python .\view_results.py checkpoints\ue20CTDEWithB1000E\ctde_results.json --detail standard --show-config
pause