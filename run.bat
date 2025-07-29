@echo off
REM Change to the directory where this script is located (project root)
cd /d %~dp0

REM Create virtual environment if it doesn't exist
IF NOT EXIST venv (
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies
pip install -r requirements.txt

REM Set PYTHONPATH so app.py can find src/data_pipeline modules
set PYTHONPATH=%cd%\src

REM Set Flask environment variables
set FLASK_APP=%cd%\chest_scan_app\app.py
set FLASK_ENV=development

REM Run the app
flask run

REM Pause to keep the window open
pause