
import json
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.features import tambah_fitur, periksa_request


# =========================================================
# 1. KONFIGURASI
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
METADATA_PATH = BASE_DIR / "artifacts" / "metadata.json"

app = FastAPI(
    title="API Prediksi Keterlambatan Paket",
    description="API untuk memprediksi risiko keterlambatan paket.",
    version="1.0.0"
)


# =========================================================
# 2. LOAD MODEL DAN METADATA
# =========================================================

try:
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

except Exception as e:
    raise RuntimeError(
        f"Gagal memuat model atau metadata: {e}"
    )


THRESHOLD = float(metadata["threshold"])
BATAS_PRIORITAS = float(metadata["batas_prioritas"])

MEDIAN_JARAK_KOTA = metadata["median_jarak_kota"]
MEDIAN_JARAK_GLOBAL = metadata["median_jarak_global"]

FITUR_REQUEST = metadata["fitur_request"]


# =========================================================
# 3. FORMAT INPUT API
# =========================================================

class DataPaket(BaseModel):

    model_config = ConfigDict(extra="forbid")

    tanggal_pickup: str
    jam_pickup: float
    kota_tujuan: str
    jarak_km: float | None = None
    layanan: str
    janji_hari: float
    berat_kg: float | None = None
    metode_bayar: str
    cuaca_saat_pickup: str
    jumlah_transit: float | None = None
    volume_hub_hari: float | None = None


# =========================================================
# 4. ENDPOINT UTAMA
# =========================================================

@app.get("/")
def halaman_utama():

    return {
        "pesan": "API Prediksi Keterlambatan Paket aktif",
        "versi_model": metadata["versi_model"]
    }


# =========================================================
# 5. ENDPOINT HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "nama_model": metadata["nama_model"],
        "versi_model": metadata["versi_model"]
    }


# =========================================================
# 6. ENDPOINT PREDIKSI
# =========================================================

@app.post("/predict")
def prediksi_paket(data: DataPaket):

    try:

        # Mengubah input menjadi dictionary
        request = data.model_dump()

        # Membuat DataFrame sesuai metadata
        df = pd.DataFrame(
            [request],
            columns=FITUR_REQUEST
        )

        # Validasi input
        masalah, peringatan = periksa_request(request)

        if masalah:
            raise HTTPException(
                status_code=400,
                detail={
                    "pesan": "Data input tidak valid",
                    "masalah": masalah,
                    "peringatan": peringatan
                }
            )

        # Feature engineering
        df_fitur = tambah_fitur(df)

        # Mengambil kota yang sudah dinormalisasi
        kota = df_fitur["kota_tujuan"].iloc[0]

        # Menentukan median jarak berdasarkan kota
        if kota in MEDIAN_JARAK_KOTA:

            median_jarak = MEDIAN_JARAK_KOTA[kota]
            sumber_jarak = f"Median kota {kota}"

        else:

            median_jarak = MEDIAN_JARAK_GLOBAL
            sumber_jarak = "Median jarak global"

        # Imputasi jarak
        jarak_diimputasi = pd.isna(request["jarak_km"])

        df_fitur["jarak_km"] = (
            df_fitur["jarak_km"].fillna(median_jarak)
        )

        # Menghitung ulang fitur turunan setelah imputasi
        df_fitur["hari_per_1000km"] = (
            df_fitur["janji_hari"]
            / (df_fitur["jarak_km"] / 1000)
        )

        # Mengambil fitur dari pipeline model
        preprocessor = model.named_steps["preprocessor"]

        fitur_numerik = list(
            preprocessor.transformers_[0][2]
        )

        fitur_kategorikal = list(
            preprocessor.transformers_[1][2]
        )

        fitur_model = fitur_numerik + fitur_kategorikal

        X = df_fitur[fitur_model]

        # Mengambil classifier
        classifier = model.named_steps["classifier"]

        kelas = list(classifier.classes_)

        if 1 not in kelas:
            raise ValueError(
                "Model tidak memiliki kelas positif 1 (terlambat)."
            )

        indeks_terlambat = kelas.index(1)

        # Menghitung probabilitas keterlambatan
        probabilitas = float(
            model.predict_proba(X)[0][indeks_terlambat]
        )

        # Keputusan berdasarkan threshold
        prediksi = int(probabilitas >= THRESHOLD)

        # Menentukan rekomendasi
        if probabilitas >= BATAS_PRIORITAS:

            rekomendasi = "PRIORITAS TINGGI"

            keterangan = (
                "Paket memiliki risiko keterlambatan tinggi. "
                "Segera lakukan tindak lanjut."
            )

        elif prediksi == 1:

            rekomendasi = "KABARI PELANGGAN"

            keterangan = (
                "Paket terindikasi berisiko terlambat. "
                "Informasikan potensi keterlambatan kepada pelanggan."
            )

        else:

            rekomendasi = "PEMANTAUAN NORMAL"

            keterangan = (
                "Risiko keterlambatan berada di bawah threshold. "
                "Paket dapat dipantau melalui proses operasional normal."
            )

        # Hasil prediksi
        return {
            "berhasil": True,
            "probabilitas": probabilitas,
            "prediksi": prediksi,
            "status": (
                "Berpotensi Terlambat"
                if prediksi == 1
                else "Tidak Terindikasi Terlambat"
            ),
            "threshold": THRESHOLD,
            "rekomendasi": rekomendasi,
            "keterangan": keterangan,
            "jarak_digunakan": float(
                df_fitur["jarak_km"].iloc[0]
            ),
            "jarak_diimputasi": bool(jarak_diimputasi),
            "sumber_jarak": sumber_jarak,
            "peringatan": peringatan
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat prediksi: {str(e)}"
        )
