@echo off
echo ========================================
echo QUOTRADING 75-DAY BACKTEST
echo ========================================
echo Starting backtest outside VS Code for maximum performance...
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
python dev-tools\full_backtest.py --days 75

echo.
echo ========================================
echo BACKTEST COMPLETE
echo ========================================
pause
