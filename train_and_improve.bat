@echo off
echo ================================================================================
echo QUOTRADING BOT - ADAPTIVE LEARNING PIPELINE
echo ================================================================================
echo.
echo This script will:
echo   1. Train exit neural network on backtest data (all 131 parameters)
echo   2. Bot will learn optimal exit params from winning vs losing trades
echo   3. Next backtest will use learned params instead of defaults
echo.
echo ================================================================================
echo.

echo [Step 1/1] Training exit neural network...
python dev-tools\train_exit_model.py

echo.
echo ================================================================================
echo TRAINING COMPLETE!
echo ================================================================================
echo.
echo The bot now has a trained exit model that predicts all 131 parameters.
echo.
echo Next steps:
echo   1. Run another backtest - it will use the trained model
echo   2. Bot will adapt exits based on market conditions
echo   3. Performance should improve as it learns from experience
echo.
echo Run: python dev-tools\full_backtest.py --days 15 --max-contracts 3 --confidence 70
echo.
pause
