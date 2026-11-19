
@echo off
title Push Aplikasi Prediksi Keterlambatan

cd /d "%~dp0"

echo ==========================================
echo   PUSH PEMBARUAN KE GITHUB
echo   PREDIKSI KETERLAMBATAN PAKET
echo ==========================================
echo.

echo [1/3] Menambahkan perubahan...
git add .

echo.
echo [2/3] Memeriksa perubahan...

git diff --cached --quiet
if not errorlevel 1 goto push

set /p pesan="Masukkan pesan commit: "
if not defined pesan set "pesan=Update aplikasi prediksi keterlambatan"

git commit -m "%pesan%"
if errorlevel 1 goto gagal

:push
echo.
echo [3/3] Mengirim perubahan ke GitHub...
git push origin main

if errorlevel 1 goto gagal

echo.
echo ==========================================
echo   PUSH BERHASIL!
echo ==========================================
pause
exit /b 0

:gagal
echo.
echo ==========================================
echo   PUSH GAGAL! Periksa pesan error.
echo ==========================================
pause
exit /b 1
