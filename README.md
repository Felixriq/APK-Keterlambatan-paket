# Prediksi Keterlambatan Paket · TIM Asoy [Hub Bandung]
Kasus: 08, Prediksi Keterlambatan Paket Hub Bandung (klasifikasi biner & estimasi probabilitas, metrik utama PR-AUC) · Aplikasi: [Isi Tautan Streamlit] · Video: [Isi Tautan Video Demo]

AnggotaNIMBagian yang dikerjakanMuhammad Thoriq Salahudin Haryoto (Felixriq)[Isi NIM][Isi bagian yang dikerjakan]Sekar Ayu Kinanti (sekar321)[Isi NIM][Isi bagian yang dikerjakan]

Aplikasi prediksi probabilitas keterlambatan pengiriman paket saat *pickup* di Hub Bandung beserta rekomendasi tindakan operasional (**AMAN**, **KABARI PELANGGAN**, atau **PRIORITASKAN + KABARI PELANGGAN**). Proyek ini telah disesuaikan sepenuhnya dengan notebook **`08_keterlambatan_paket_revisi_3.ipynb`**.

## Struktur Direktori & File
- `src/features.py` : Modul pembersihan data, normalisasi kota/layanan, rekayasa 17 fitur (`tambah_fitur`), serta validasi masukan (`periksa_request`).
- `artifacts/model.joblib` : Pipeline model final **Logistic Regression + fitur** (`StandardScaler` + `OneHotEncoder` + `LogisticRegression`).
- `artifacts/metadata.json` — Metadata model (`versi_model: 1.0.0`, `threshold: 0.11`, `batas_prioritas: 0.50`, daftar 11 `fitur_request`, metrik validasi & test, serta asumsi biaya `FN: Rp50.000` dan `FP: Rp5.000`).
- `core.py` : Modul integrasi model, `rekomendasi()`, dan `prediksi(request)`.
- `streamlit_app.py` : Antarmuka web interaktif berbasis **Streamlit** (siap di-deploy ke **Streamlit Community Cloud**).
- `app.py` : Antarmuka web lokal berbasis **Flask** (`http://127.0.0.1:5000` dan endpoint `POST /api/predict`).
- `requirements.txt` : Daftar pustaka Python untuk deployment lokal maupun Streamlit Cloud.
- `push_ke_github.bat` : Skrip otomatis untuk melakukan commit dan push seluruh pembaruan ke repositori GitHub `Felixriq/prediksi-keterlambatan-paket`.

## Ringkasan Evaluasi Model
- **Model Terpilih**: `Logistic Regression + fitur`
- **PR-AUC Validation**: `0.662` (Baseline Regresi Logistik: `0.646`)
- **PR-AUC Test**: `0.378` (Recall: `0.772`, Precision: `0.238`)
- **Threshold Terbaik**: `0.11` (Biaya minimum validasi Rp19.465.000; biaya test Rp22.290.000 — turun 52% dibanding tanpa model Rp46.950.000)
