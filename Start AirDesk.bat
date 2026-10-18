@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto source
if exist "dist\AirDesk\AirDesk.exe" goto packaged
echo Chưa cài AirDesk. Hãy chạy "Setup AirDesk.bat" trước.
if "%~1"=="" pause
exit /b 1
:source
".venv\Scripts\python.exe" -m airdesk %*
goto finished
:packaged
"dist\AirDesk\AirDesk.exe" %*
:finished
set "airdesk_exit=%errorlevel%"
if "%airdesk_exit%"=="0" exit /b 0
echo AirDesk không khởi chạy thành công. Xem lỗi ở trên và phần Khắc phục lỗi trong README.md.
if "%~1"=="" pause
exit /b %airdesk_exit%
