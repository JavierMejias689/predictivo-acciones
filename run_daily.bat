@echo off
cd /d C:\Proyectos\gm3\predictivo-acciones
call .venv\Scripts\activate.bat
python run_daily.py --threshold 0.60
