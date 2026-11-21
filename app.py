
import json
import os
import requests
import joblib
import pandas as pd
import streamlit as st

from pathlib import Path
from src.features import tambah_fitur, periksa_request


# =========================================================
# 1. KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Prediksi Keterlambatan Paket",
    page_icon="📦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

METADATA_PATH = BASE_DIR / "artifacts" / "metadata.json"
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"


# =========================================================
# 2. KONFIGURASI API
# =========================================================

# Prioritas URL:
# 1. Streamlit Secrets
# 2. Environment Variable
# 3. Jika tidak ada, gunakan model lokal

try:
    API_URL = st.secrets.get("API_URL", "")
except Exception:
    API_URL = ""

API_URL = (
    API_URL or os.getenv("API_URL", "")
).strip().rstrip("/")


# =========================================================
# 3. LOAD METADATA DAN MODEL
# =========================================================

@st.cache_data
def load_metadata():

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


try:

    metadata = load_metadata()

except Exception as e:

    st.error(f"Gagal memuat metadata: {e}")
    st.stop()


THRESHOLD = float(metadata["threshold"])


# =========================================================
# 4. FUNGSI PREDIKSI LOKAL
# =========================================================

def prediksi_lokal(request):

    # Validasi input
    masalah, peringatan = periksa_request(request)

    if masalah:
        return None, masalah, peringatan

    # Load model
    model = load_model()

    # Membuat DataFrame
    fitur_request = metadata["fitur_request"]

    df = pd.DataFrame(
        [request],
        columns=fitur_request
    )

    # Feature engineering
    df_fitur = tambah_fitur(df)

    # Menentukan kota
    kota = df_fitur["kota_tujuan"].iloc[0]

    # Mengambil median jarak dari metadata
    median_jarak_kota = metadata["median_jarak_kota"]
    median_jarak_global = metadata["median_jarak_global"]

    if kota in median_jarak_kota:

        median_jarak = median_jarak_kota[kota]
        sumber_jarak = f"Median kota {kota}"

    else:

        median_jarak = median_jarak_global
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
    batas_prioritas = float(metadata["batas_prioritas"])

    if probabilitas >= batas_prioritas:

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

    # Mengembalikan hasil dengan format yang sama seperti API
    hasil = {
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

    return hasil, [], peringatan


# =========================================================
# 5. FUNGSI MENAMPILKAN HASIL
# =========================================================

def tampilkan_hasil(hasil, mode):

    if mode == "API":

        st.success("Mode prediksi: FastAPI")

    else:

        st.info("Mode prediksi: Model Lokal (joblib)")

    # Peringatan
    for item in hasil.get("peringatan", []):
        st.warning(item)

    st.divider()

    # Hasil prediksi
    st.subheader("Hasil Prediksi")

    probabilitas = float(hasil["probabilitas"])

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Probabilitas Terlambat",
            f"{probabilitas:.2%}"
        )

    with col2:

        st.metric(
            "Threshold Model",
            f"{hasil['threshold']:.0%}"
        )

    with col3:

        st.metric(
            "Jarak yang Digunakan",
            f"{hasil['jarak_digunakan']:.1f} km"
        )

    st.progress(
        min(max(probabilitas, 0.0), 1.0),
        text="Tingkat Risiko Keterlambatan"
    )

    # Rekomendasi
    if hasil["rekomendasi"] == "PRIORITAS TINGGI":

        st.error(
            f"🚨 {hasil['rekomendasi']}"
        )

    elif hasil["rekomendasi"] == "KABARI PELANGGAN":

        st.warning(
            f"⚠️ {hasil['rekomendasi']}"
        )

    else:

        st.success(
            f"✅ {hasil['rekomendasi']}"
        )

    st.write(hasil["keterangan"])

    # Informasi imputasi jarak
    if hasil["jarak_diimputasi"]:

        st.info(
            f"Jarak tidak diketahui. Sistem menggunakan "
            f"{hasil['sumber_jarak']} sebesar "
            f"{hasil['jarak_digunakan']:.1f} km."
        )

    st.caption(
        "Hasil prediksi merupakan estimasi berdasarkan "
        "model machine learning, bukan kepastian bahwa "
        "paket akan mengalami keterlambatan."
    )


# =========================================================
# 6. FUNGSI MENANGANI ERROR API
# =========================================================

def tampilkan_error_api(response):

    try:
        hasil_error = response.json()

    except ValueError:
        hasil_error = {}

    detail = hasil_error.get(
        "detail",
        response.text
    )

    if isinstance(detail, dict):

        st.error(
            detail.get(
                "pesan",
                "Data tidak dapat diproses."
            )
        )

        for item in detail.get("masalah", []):
            st.write(f"- {item}")

        for item in detail.get("peringatan", []):
            st.warning(item)

    else:

        st.error(f"Prediksi gagal: {detail}")


# =========================================================
# 7. HEADER APLIKASI
# =========================================================

st.title("📦 Sistem Prediksi Keterlambatan Paket")

st.write(
    "Aplikasi untuk membantu staf operasional hub "
    "mengidentifikasi paket yang berisiko mengalami "
    "keterlambatan berdasarkan informasi pada saat pickup."
)

st.caption(
    f"Model: {metadata['nama_model']} | "
    f"Versi: {metadata['versi_model']} | "
    f"Threshold: {THRESHOLD:.0%}"
)

st.divider()


# =========================================================
# 8. INPUT DATA PAKET
# =========================================================

st.subheader("Input Data Paket")

# INFORMASI PICKUP

st.markdown("#### Informasi Pickup")

col1, col2 = st.columns(2)

with col1:

    tanggal_pickup = st.date_input(
        "Tanggal Pickup"
    )

    jam_pickup = st.number_input(
        "Jam Pickup",
        min_value=7,
        max_value=22,
        value=10,
        step=1
    )

    kota_tujuan = st.selectbox(
        "Kota Tujuan",
        [
            "Bekasi",
            "Bogor",
            "Jakarta",
            "Yogyakarta",
            "Surabaya",
            "Medan",
            "Denpasar",
            "Bali",
            "Balikpapan",
            "Cirebon",
            "Makassar",
            "Malang",
            "Palembang",
            "Pontianak",
            "Semarang",
            "Tasikmalaya"
        ]
    )

with col2:

    layanan = st.selectbox(
        "Jenis Layanan",
        ["REG", "EXPRESS", "HEMAT"]
    )

    janji_hari = st.number_input(
        "Estimasi Pengiriman (Hari)",
        min_value=1,
        max_value=6,
        value=3,
        step=1
    )

    metode_bayar = st.selectbox(
        "Metode Pembayaran",
        ["COD", "non_COD"]
    )


# INFORMASI PAKET

st.markdown("#### Informasi Paket")

col3, col4 = st.columns(2)

with col3:

    tidak_tahu_jarak = st.checkbox(
        "Jarak pengiriman tidak diketahui"
    )

    if tidak_tahu_jarak:

        jarak_km = None

        st.text_input(
            "Jarak Pengiriman (km)",
            value="Tidak diketahui",
            disabled=True
        )

    else:

        jarak_km = st.number_input(
            "Jarak Pengiriman (km)",
            min_value=0.1,
            max_value=2500.0,
            value=100.0,
            step=10.0
        )

    berat_kg = st.number_input(
        "Berat Paket (kg)",
        min_value=0.0,
        value=1.0,
        step=0.1
    )

with col4:

    cuaca = st.selectbox(
        "Cuaca Saat Pickup",
        [
            "cerah",
            "berawan",
            "hujan_ringan",
            "hujan_lebat",
            "tidak_diketahui"
        ]
    )


# INFORMASI OPERASIONAL HUB

st.markdown("#### Informasi Operasional Hub")

col5, col6 = st.columns(2)

with col5:

    jumlah_transit = st.number_input(
        "Jumlah Transit",
        min_value=0,
        value=2,
        step=1
    )

with col6:

    volume_hub_hari = st.number_input(
        "Volume Paket Hub per Hari",
        min_value=2800,
        max_value=18000,
        value=8000,
        step=100
    )

st.caption(
    "Jumlah transit merupakan perkiraan jumlah titik transit "
    "yang akan dilalui paket. Volume hub merupakan jumlah "
    "paket yang diperkirakan ditangani saat pickup, bukan "
    "volume aktual pada akhir hari."
)

st.divider()


# =========================================================
# 9. TOMBOL PREDIKSI
# =========================================================

tombol_prediksi = st.button(
    "🔍 Prediksi Keterlambatan",
    use_container_width=True
)


# =========================================================
# 10. PROSES PREDIKSI OTOMATIS
# =========================================================

if tombol_prediksi:

    request_data = {
        "tanggal_pickup": str(tanggal_pickup),
        "jam_pickup": jam_pickup,
        "kota_tujuan": kota_tujuan,
        "jarak_km": jarak_km,
        "layanan": layanan,
        "janji_hari": janji_hari,
        "berat_kg": berat_kg,
        "metode_bayar": metode_bayar,
        "cuaca_saat_pickup": cuaca,
        "jumlah_transit": jumlah_transit,
        "volume_hub_hari": volume_hub_hari
    }

    hasil = None
    mode = None

    # -----------------------------------------------------
    # MODE 1: API TERSEDIA
    # -----------------------------------------------------

    if API_URL:

        try:

            with st.spinner("Menghubungi API..."):

                response = requests.post(
                    f"{API_URL}/predict",
                    json=request_data,
                    timeout=15
                )

            if response.status_code == 200:

                hasil = response.json()
                mode = "API"

            else:

                tampilkan_error_api(response)

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout
        ):

            st.warning(
                "API tidak dapat dihubungi. "
                "Sistem akan beralih ke model lokal."
            )

        except requests.exceptions.RequestException as e:

            st.warning(
                f"Terjadi masalah komunikasi API: {e}. "
                "Sistem akan mencoba model lokal."
            )

        except ValueError as e:

            st.error(f"Respons API tidak valid: {e}")

    # -----------------------------------------------------
    # MODE 2: MODEL LOKAL
    # -----------------------------------------------------

    # Digunakan jika:
    # - API_URL tidak tersedia
    # - API gagal dihubungi
    # - API mengalami timeout

    if hasil is None and mode is None:

        try:

            with st.spinner(
                "Memproses prediksi menggunakan model lokal..."
            ):

                hasil, masalah, peringatan = prediksi_lokal(
                    request_data
                )

            if masalah:

                st.error("Data input tidak valid:")

                for item in masalah:
                    st.write(f"- {item}")

                for item in peringatan:
                    st.warning(item)

            else:

                mode = "LOKAL"

        except Exception as e:

            st.error(
                f"Gagal melakukan prediksi lokal: {e}"
            )

    # -----------------------------------------------------
    # MENAMPILKAN HASIL
    # -----------------------------------------------------

    if hasil is not None and mode is not None:

        tampilkan_hasil(hasil, mode)


# =========================================================
# 11. FOOTER
# =========================================================

st.divider()

st.caption(
    "Sistem Prediksi Keterlambatan Paket | "
    f"Model {metadata['versi_model']} | "
    f"Dibuat pada {metadata['dibuat']}"
)
