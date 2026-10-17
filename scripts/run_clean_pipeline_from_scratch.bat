@echo off
REM ==============================================================================
REM PE-CIR: End-to-End Clean Pipeline from Scratch (B1-clean -> B4-new -> B5-new)
REM All models trained with seed 42 on clean reference-disjoint split.
REM ==============================================================================

set IMAGE_ROOT=%1
if "%IMAGE_ROOT%"=="" set IMAGE_ROOT=data\images

set OUTPUT_DIR=%2
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=runs\clean_pipeline_from_scratch

echo ==============================================================================
echo PE-CIR: MASTER CLEAN EXPERIMENT PIPELINE (SEED: 42)
echo Image root : %IMAGE_ROOT%
echo Output dir : %OUTPUT_DIR%
echo ==============================================================================

mkdir %OUTPUT_DIR% 2>nul

echo.
echo >>> STEP 1/2: Running Controlled Training Pipeline (B1 -^> B4 -^> Cluster CI Gate -^> B5)...
python scripts\controlled_repair_b4_to_b5.py ^
    --train-manifest output\BAN_GIAO_CHO_CO_20261010\1_train_manifest\train.csv ^
    --image-root %IMAGE_ROOT% ^
    --output-dir %OUTPUT_DIR% ^
    --train-b1 ^
    --b1-epochs 3 ^
    --b4-epochs 3 ^
    --b5-epochs 3 ^
    --batch-size 16 ^
    --seed 42 ^
    --cf-margin 0.15 ^
    --lambda-cf 0.8 ^
    --n-boot 10000 ^
    --device auto
if errorlevel 1 goto error

echo.
echo >>> STEP 2/2: Running Full Triplet + Gallery Comparison with 10,000 Cluster Bootstrap iterations...
python scripts\compare_b1_b4_clean_val.py ^
    --manifest %OUTPUT_DIR%\clean_data\clean_val.csv ^
    --image-root %IMAGE_ROOT% ^
    --b1-checkpoint %OUTPUT_DIR%\B1_clean_train\best_checkpoint.pt ^
    --b4-checkpoint %OUTPUT_DIR%\B4_controlled_repair\best_checkpoint.pt ^
    --output-dir %OUTPUT_DIR%\comparison_b1_b4 ^
    --n-boot 10000 ^
    --device auto
if errorlevel 1 goto error

echo.
echo ==============================================================================
echo ALL EXPERIMENTS COMPLETED SUCCESSFULLY!
echo ==============================================================================
exit /b 0

:error
echo.
echo [ERROR] Pipeline failed or was halted by Scientific GO Gate.
exit /b 1
