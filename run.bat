@echo off
title Laptop Price Predictor
cd /d "%~dp0"

echo Checking required packages...
python -m pip install -r requirements.txt --quiet

echo.
echo Starting Laptop Price Predictor...
echo Chrome will open automatically in a moment.
echo (Keep this window open while using the app. Close it to stop the server.)
echo.

python app.py

pause
