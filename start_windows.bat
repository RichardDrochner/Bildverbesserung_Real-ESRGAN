@echo off
setlocal

cd /d %~dp0

if not exist "Real-ESRGAN\" (
  echo FEHLER: Ordner "Real-ESRGAN" fehlt.
  echo Bitte Repository neu klonen oder ZIP korrekt entpacken.
  pause
  exit /b 1
)

if not exist "weights\RealESRGAN_x4plus.pth" (
  echo FEHLER: Model-Weights fehlen: weights\RealESRGAN_x4plus.pth
  echo Lege die Datei dort ab oder entpacke das bereitgestellte Model-ZIP in den Projektordner.
  pause
  exit /b 1
)

if not exist .venv (
  py -3.11 -m venv .venv
)

call .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

python app.py

pause