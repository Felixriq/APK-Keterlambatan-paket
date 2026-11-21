# Prediksi Keterlambatan Paket · Tim [08]

**Kasus:** 08, Prediksi Keterlambatan Paket Hub Bandung (klasifikasi biner & estimasi probabilitas, metrik utama PR-AUC) · **Aplikasi:** (https://prediksipaketterlambat.streamlit.app/) · **Video:** [Isi Tautan Video Demo]

| Anggota | NIM | Bagian yang dikerjakan |
| :--- | :--- | :--- |
| Muhammad Thoriq Salahudin Haryoto (`Felixriq`) | [2403154] | • Dokumentasi Repositori (README.md): Penyusunan dokumentasi proyek prediksi keterlambatan paket Hub Bandung yang mencakup gambaran umum sistem, struktur proyek, dan panduan instalasi/eksekusi.<br><br>
• Audit Data Leakage & Feature Engineering: Evaluasi kebocoran data (data leakage) termasuk peninjauan dan penghapusan fitur volume yang sebelumnya membuat metrik PR-AUC bias—serta pemanfaatan fitur jarak antarkota (JARAK_KOTA).
•Skrip & Materi Video Presentasi (Walk-through 3 Menit): Penyusunan naskah 4 segmen beserta teks screen overlay yang membahas distribusi target, data cleaning & split, perbandingan model dan cost curves, top features, format output JSON, serta keterbatasan model.|
| Sekar Ayu Kinanti (`sekar321`) | [2404201] | •	Perbaikan kode program pengembangan model di Google Colab.<br><br>•	Pengembangan API lokal, serta pembuatan aplikasi berbasis Streamlit yang terintegrasi dengan API local.<br><br>•	Deployment aplikasi Streamlit berbasis model Joblib agar dapat diakses secara online tanpa memerlukan koneksi ke API. |

## Masalah dan Pengguna

Pengelola operasional Hub Bandung perlu mengetahui risiko keterlambatan pengiriman paket sejak saat *pickup* agar dapat mengambil tindakan dini (**AMAN**, **KABARI PELANGGAN**, atau **PRIORITASKAN + KABARI PELANGGAN**) dan menekan biaya penalti keterlambatan. Aplikasi memprediksi probabilitas keterlambatan dari 11 masukan *request*: `tanggal_pickup`, `jam_pickup`, `kota_tujuan`, `jarak_km`, `layanan`, `janji_hari`, `berat_kg`, `metode_bayar`, `cuaca_saat_pickup`, `jumlah_transit`, dan `volume_hub_hari`.

## Aturan Brief yang Dipatuhi

- Tidak memakai data masa depan; seluruh fitur masukan diturunkan dari informasi yang tersedia saat *pickup*.
- Pembagian berdasarkan waktu (*temporal split*): train `< 2026-01-01` (26.633 baris, 17,8% terlambat), validation Jan–Mar 2026 (6.623 baris, 21,9% terlambat), test Apr–Jun 2026 (6.744 baris, 13,9% terlambat).
- Baseline wajib: Regresi Logistik dasar tanpa rekayasa fitur tambahan (PR-AUC validation `0,646`, PR-AUC test `0,356`).
- Test set dibuka satu kali di akhir; model tidak diubah setelahnya.

## Data dan Pembersihan

Data mentah awal berisi 40.500 baris × 13 kolom (periode 1 Jan 2025 – 30 Jun 2026), menjadi 40.000 baris bersih setelah penghapusan 500 baris duplikat. Prapemrosesan dan rekayasa fitur di `src/features.py` menangani:

| Masalah / Transformasi | Cara menangani |
| :--- | :--- |
| Penulisan kota & layanan beragam | Normalisasi 15 kota tujuan (`MAPPING_KOTA`) dan format layanan |
| Jarak kosong (`jarak_km`) | Imputasi median jarak per kota (`JARAK_KOTA`) atau median global `356,0` km |
| Pencilan volume hub | Pemotongan (*clipping*) `volume_hub_hari` di atas `18.000` |
| Satuan berat tidak seragam | Konversi berat `>= 100` gram ke kilogram (`berat_kg`) |
| Rekayasa fitur operasional | Penambahan 7 fitur (`after_cutoff`, `senin`, `slack`, `hari_per_1000km`, `surge`, `log_vol`, `hujan_lebat`) menjadi total 17 fitur |
| Validasi masukan aplikasi | Pemeriksaan batas dan tipe input melalui `periksa_request` |

## Model

Pipeline `StandardScaler` (13 fitur numerik) + `OneHotEncoder` (4 fitur kategorikal) + `LogisticRegression` (`Logistic Regression + fitur`, versi `1.0.0`). Model ini dipilih karena mengungguli baseline Regresi Logistik dasar pada validasi (PR-AUC `0,662` vs `0,646`).

Penentuan aksi operasional menggunakan dua ambang batas (lihat `artifacts/metadata.json`):

- **`threshold = 0,11`**: batas klasifikasi terlambat (dioptimalkan untuk menekan biaya dengan asumsi penalti `FN = Rp50.000` dan `FP = Rp5.000`). Probabilitas `>= 0,11` memicu rekomendasi **KABARI PELANGGAN**.
- **`batas_prioritas = 0,50`**: probabilitas `>= 0,50` memicu rekomendasi **PRIORITASKAN + KABARI PELANGGAN**.

## Hasil

Metrik utama: **PR-AUC** (makin besar makin baik), disertai evaluasi biaya operasional (`FN = Rp50.000`, `FP = Rp5.000`).

| Model | PR-AUC Validation | PR-AUC Test |
| :--- | :---: | :---: |
| Baseline: Regresi Logistik dasar | `0,646` | `0,356` |
| **Logistic Regression + fitur (v1.0.0)** | **`0,662`** | **`0,378`** |

- Pada test set: Recall `0,772` (725/939 paket terlambat terdeteksi), Precision `0,238`.
- Biaya operasional pada test set dengan model sebesar **Rp22.290.000**, turun **52,5%** dibanding skenario tanpa model (**Rp46.950.000**); biaya minimum pada validasi tercatat **Rp19.465.000**.

## Cara Menjalankan

```text
git clone [https://github.com/Felixriq/APK-Keterlambatan-paket.git](https://github.com/Felixriq/APK-Keterlambatan-paket.git)
cd APK-Keterlambatan-paket
python -m venv .venv
.venv\Scripts\activate                   # Windows
pip install -r requirements.txt
streamlit run app.py                     # aplikasi Streamlit
```
## Struktur Repositori

```text
artifacts/
  ├── metadata.json      # Versi model, fitur request, threshold, dan metrik
  └── model.joblib       # Pipeline model terlatih
src/
  └── features.py        # Pembersihan data, rekayasa fitur, dan validasi input
app.py                   # Aplikasi web Streamlit
requirements.txt         # Daftar dependensi Python
push_ke_github.bat       # Skrip bantu push ke GitHub
README.md                # Dokumentasi proyek
```

## Keterbatasan

- Fitur `volume_hub_hari` berpotensi menimbulkan *temporal leakage* jika angkanya baru lengkap di akhir hari (bukan *running count* saat *pickup*); `JARAK_KOTA` juga dihitung sebelum pembagian data.
- Precision pada test set relatif rendah (`0,238`) karena `threshold = 0,11` diprioritaskan untuk meminimalkan *False Negative* yang biayanya 10× lebih besar dari *False Positive*.
- Data pelatihan hanya mencakup pengiriman dari Hub Bandung pada periode Jan 2025 – Jun 2026.

## Daftar Periksa Sebelum Dikumpulkan

- [x] Repositori bersifat publik dan kedua anggota memiliki commit
- [x] Baseline wajib dilaporkan dan dibandingkan
- [x] Data test hanya dibuka satu kali, di akhir
- [x] `artifacts/metadata.json` berisi versi model, fitur masukan, threshold, dan metrik
- [ ] Notebook (`08_keterlambatan_paket_Sekar&Thoriq.ipynb`) dan pengujian otomatis (`tests/`) diunggah ke repositori
- [x] Tautan aplikasi Streamlit dan video demo dicantumkan di `README.md`
- [x] Laporan paling banyak 6 halaman dan video paling lama 5 menit
