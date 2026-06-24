@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  set PYTHON=py -3
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    set PYTHON=python
  ) else (
    echo Python 3 is required. Install Python 3.10 or newer from https://www.python.org/downloads/
    pause
    exit /b 1
  )
)

if not exist ".venv\Scripts\python.exe" (
  %PYTHON% -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r server\requirements.txt

echo.
echo DoubleDeep AI is starting at http://127.0.0.1:8787/
echo Keep this window open while using the website.
echo.
start "" "http://127.0.0.1:8787/"
python server\app.py

pause
