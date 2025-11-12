@echo off
title Math Professor AI - Full Stack Launcher
echo.
echo ===========================================
echo 🚀 Starting Math Professor AI Application
echo ===========================================
echo.

REM === Activate your conda environment ===
call conda activate mathprofessor

REM === Start backend (FastAPI) ===
echo 🧠 Launching FastAPI backend...
cd backend
start "FastAPI Backend" cmd /k "uvicorn main:app --reload"

REM === Give backend time to initialize ===
timeout /t 5 /nobreak >nul

REM === Start frontend (React + Vite) ===
echo 💻 Launching React frontend...
cd ../frontend
start "React Frontend" cmd /k "npm run dev"

REM === Confirmation message ===
echo.
echo ✅ Both backend and frontend are now running!
echo 🔗 Frontend: http://localhost:5173
echo 🔗 Backend:  http://127.0.0.1:8000
echo -------------------------------------------
echo.

REM === Wait for user to close ===
echo Press any key to close all...
pause >nul

REM === Kill both running windows cleanly ===
taskkill /FI "WINDOWTITLE eq FastAPI Backend" /F >nul
taskkill /FI "WINDOWTITLE eq React Frontend" /F >nul

echo.
echo 🛑 All processes stopped. Goodbye!
exit
