@echo off
echo ========================================================
echo Push Pembaruan Prediksi Keterlambatan Paket ke GitHub
echo ========================================================
git init
if exist model_terlambat.joblib del /f /q model_terlambat.joblib
if exist meta.json del /f /q meta.json
git add -A
git commit -m "Update proyek sesuai 08_keterlambatan_paket_revisi_3 (src/features.py, artifacts/model.joblib, metadata.json, threshold 0.11)"
git branch -M main
git remote remove origin 2>nul
git remote add origin https://github.com/Felixriq/prediksi-keterlambatan-paket.git
git push -u origin main --force
pause
