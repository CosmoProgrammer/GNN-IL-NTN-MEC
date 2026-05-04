@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

set "SCRIPT=view_resultts.py"
if not exist ".\%SCRIPT%" (
    echo [WARN] .\view_resultts.py not found. Falling back to .\view_results.py
    set "SCRIPT=view_results.py"
)

if not exist ".\%SCRIPT%" (
    echo [ERROR] Neither .\view_resultts.py nor .\view_results.py was found in this folder.
    exit /b 1
)

set "BASE=Seed Results\NTN_MEC_Ablation"
set "FOUND=0"
set "MISSING=0"

for %%U in (ue5 ue10 ue20 ue30 ue40) do (
    for %%S in (seed42 seed52 seed62 seed72 seed82) do (
        for %%M in (standard gnn) do (
            if /i "%%M"=="gnn" (
                set "FILE=gnn_il_results.json"
            ) else (
                set "FILE=il_results.json"
            )

            set "JSON=!BASE!\%%U\%%S\%%M\!FILE!"

            if exist "!JSON!" (
                echo.
                echo ===== %%U / %%S / %%M =====
                python ".\%SCRIPT%" "!JSON!" --detail brief
                set /a FOUND+=1
            ) else (
                echo [MISSING] !JSON!
                set /a MISSING+=1
            )
        )
    )
)

echo.
echo Completed. Found !FOUND! JSON file(s).
if "!MISSING!"=="0" echo No missing expected JSON files.
if not "!MISSING!"=="0" echo Missing !MISSING! expected JSON file(s).
endlocal
