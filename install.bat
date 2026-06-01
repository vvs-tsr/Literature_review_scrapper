@echo off
echo Setting up virtual environment...
python -m venv venv

echo Installing dependencies...
venv\Scripts\pip install -r requirements.txt

echo.
echo Done! Activate the environment with: venv\Scripts\activate
pause
