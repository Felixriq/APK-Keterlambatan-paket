
import json
import joblib
import streamlit as st
import pandas as pd

from pathlib import Path
from datetime import date

from src.features import tambah_fitur, periksa_request


# =====================================================
# KONFIGURASI HALAMAN
# =====================================================

st.set_page_config(
    page_title="Prediksi Keterlambatan Paket",
    page_icon="📦",
    layout="wide"
)


# =====================================================
# LOKASI FILE
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
METADATA_PATH = BASE_DIR / "artifacts" / "metadata.json"


# =====================================================
# LOAD MODEL DAN METADATA
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    meta = load_metadata()

except Exception as e:
    st.error(f"Gagal memuat model atau metadata: {e}")
    st.stop()


# =====================================================
# INFORMASI MODEL
# =====================================================

THRESHOLD = meta["threshold"]
BATAS_PRIORITAS = meta["batas_prioritas"]

MEDIAN_JARAK_KOTA = meta["median_jarak_kota"]
MEDIAN_JARAK_GLOBAL = meta["median_jarak_global"]


# =====================================================
# KATEGORI INPUT
# =====================================================

KOTA = [
    "Bali",
    "Balikpapan",
    "Bekasi",
    "Bogor",
    "Cirebon",
    "Denpasar",
    "Jakarta",
    "Makassar",
    "Malang",
    "Medan",
    "Palembang",
    "Pontianak",
    "Semarang",
    "Surabaya",
    "Tasikmalaya",
    "Yogyakarta"
]

LAYANAN = [
    "EXPRESS",
    "HEMAT",
    "REG"
]

METODE_BAYAR = [
    "COD",
    "non_COD"
]

CUACA = [
    "berawan",
    "cerah",
    "hujan",
    "hujan_lebat",
    "tidak_diketahui"
]


# =====================================================
# JUDUL APLIKASI
# =====================================================

st.title("📦 Prediksi Keterlambatan Pengiriman Paket")

st.write(
    "Aplikasi ini digunakan untuk memperkirakan kemungkinan "
    "paket mengalami keterlambatan berdasarkan informasi "
    "pengiriman yang dimasukkan."
)

st.divider()


# =====================================================
# INFORMASI MODEL
# =====================================================

with st.expander("Informasi Model"):
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Nama Model",
        meta["nama_model"]
    )

    col2.metric(
        "Threshold",
        f"{THRESHOLD:.2f}"
    )

    col3.metric(
        "Recall Test",
        f"{meta['metrik_test']['recall']:.3f}"
    )

    st.caption(
        f"Versi model: {meta['versi_model']} | "
        f"Tanggal dibuat: {meta['dibuat']}"
    )


# =====================================================
# INPUT PENGIRIMAN
# =====================================================

st.subheader("Masukkan Informasi Pengiriman")

# Checkbox harus di luar form agar perubahan status
# langsung memperbarui kolom jarak.

jarak_tidak_diketahui = st.checkbox(
    "Jarak pengiriman tidak diketahui"
)


with st.form("form_prediksi"):

    # ---------------------------------------------
    # INFORMASI WAKTU
    # ---------------------------------------------

    st.markdown("### 🕒 Informasi Waktu")

    col1, col2 = st.columns(2)

    with col1:
        tanggal_pickup = st.date_input(
            "Tanggal Pickup",
            value=date.today()
        )

    with col2:
        jam_pickup = st.number_input(
            "Jam Pickup",
            min_value=7.0,
            max_value=22.0,
            value=10.0,
            step=0.5
        )

    st.divider()

    # ---------------------------------------------
    # INFORMASI TUJUAN
    # ---------------------------------------------

    st.markdown("### 📍 Informasi Tujuan")

    col1, col2 = st.columns(2)

    with col1:
        kota_tujuan = st.selectbox(
            "Kota Tujuan",
            options=KOTA
        )

    with col2:
        jarak_km = st.number_input(
            "Jarak Pengiriman (km)",
            min_value=0.0,
            max_value=2500.0,
            value=100.0,
            step=1.0,
            disabled=jarak_tidak_diketahui
        )

    st.divider()

    # ---------------------------------------------
    # INFORMASI PAKET
    # ---------------------------------------------

    st.markdown("### 📦 Informasi Paket")

    col1, col2 = st.columns(2)

    with col1:
        layanan = st.selectbox(
            "Jenis Layanan",
            options=LAYANAN
        )

    with col2:
        janji_hari = st.number_input(
            "Estimasi Janji Pengiriman (hari)",
            min_value=1,
            max_value=6,
            value=3,
            step=1
        )

    berat_kg = st.number_input(
        "Berat Paket (kg)",
        min_value=0.0,
        value=2.5,
        step=0.1
    )

    st.divider()

    # ---------------------------------------------
    # INFORMASI TAMBAHAN
    # ---------------------------------------------

    st.markdown("### 🌦️ Informasi Tambahan")

    col1, col2 = st.columns(2)

    with col1:
        metode_bayar = st.selectbox(
            "Metode Pembayaran",
            options=METODE_BAYAR
        )

    with col2:
        cuaca = st.selectbox(
            "Cuaca Saat Pickup",
            options=CUACA
        )

    st.divider()

    tombol_prediksi = st.form_submit_button(
        "Prediksi Keterlambatan",
        use_container_width=True
    )


# =====================================================
# PROSES PREDIKSI
# =====================================================

if tombol_prediksi:

    # ---------------------------------------------
    # MEMBUAT REQUEST
    # ---------------------------------------------

    request = {
        "tanggal_pickup": tanggal_pickup.isoformat(),
        "jam_pickup": jam_pickup,
        "kota_tujuan": kota_tujuan,
        "jarak_km": None if jarak_tidak_diketahui else jarak_km,
        "layanan": layanan,
        "janji_hari": janji_hari,
        "berat_kg": berat_kg,
        "metode_bayar": metode_bayar,
        "cuaca_saat_pickup": cuaca
    }

    # ---------------------------------------------
    # VALIDASI INPUT
    # ---------------------------------------------

    masalah, peringatan = periksa_request(request)

    if masalah:

        st.error("Terdapat kesalahan pada input:")

        for item in masalah:
            st.write(f"- {item}")

        st.stop()

    # ---------------------------------------------
    # PERINGATAN INPUT
    # ---------------------------------------------

    for item in peringatan:
        st.warning(item)

    # ---------------------------------------------
    # IMPUTASI JARAK
    # ---------------------------------------------

    if (
        request["jarak_km"] is None
        or request["jarak_km"] <= 0
    ):

        rata_rata_kota = MEDIAN_JARAK_KOTA.get(
            kota_tujuan,
            MEDIAN_JARAK_GLOBAL
        )

        request["jarak_km"] = rata_rata_kota

        st.info(
            f"Jarak tidak diketahui. Sistem menggunakan "
            f"median jarak kota {kota_tujuan}: "
            f"{rata_rata_kota:.2f} km."
        )

    # ---------------------------------------------
    # MEMBUAT DATAFRAME
    # ---------------------------------------------

    df_input = pd.DataFrame([request])

    # ---------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------

    data_input = tambah_fitur(df_input)

    # ---------------------------------------------
    # PREDIKSI PROBABILITAS
    # ---------------------------------------------

    try:

        probabilitas = model.predict_proba(data_input)[0]

        kelas_model = list(model.classes_)

        indeks_terlambat = kelas_model.index(1)

        probabilitas_terlambat = probabilitas[
            indeks_terlambat
        ]

        # -----------------------------------------
        # KLASIFIKASI BERDASARKAN THRESHOLD
        # -----------------------------------------

        if probabilitas_terlambat >= THRESHOLD:

            status = "BERPOTENSI TERLAMBAT"
            warna = "error"

        else:

            status = "RISIKO KETERLAMBATAN RENDAH"
            warna = "success"

        # -----------------------------------------
        # PRIORITAS PENANGANAN
        # -----------------------------------------

        if probabilitas_terlambat >= BATAS_PRIORITAS:

            prioritas = "PRIORITAS TINGGI"

        else:

            prioritas = "PRIORITAS NORMAL"

        # -----------------------------------------
        # HASIL PREDIKSI
        # -----------------------------------------

        st.divider()

        st.subheader("📊 Hasil Prediksi")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Probabilitas Keterlambatan",
                f"{probabilitas_terlambat * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Threshold Klasifikasi",
                f"{THRESHOLD * 100:.0f}%"
            )

        st.progress(
            float(probabilitas_terlambat)
        )

        if warna == "error":

            st.error(f"⚠️ {status}")

        else:

            st.success(f"✅ {status}")

        # -----------------------------------------
        # PRIORITAS
        # -----------------------------------------

        st.markdown("### Tingkat Prioritas Penanganan")

        if prioritas == "PRIORITAS TINGGI":

            st.warning(
                "🔴 PRIORITAS TINGGI — Paket perlu "
                "mendapatkan perhatian lebih."
            )

        else:

            st.info(
                "🟢 PRIORITAS NORMAL — Paket belum "
                "memenuhi batas prioritas tinggi."
            )

        # -----------------------------------------
        # INFORMASI INPUT
        # -----------------------------------------

        with st.expander("Lihat Detail Data Pengiriman"):

            st.write({
                "Tanggal Pickup": request["tanggal_pickup"],
                "Jam Pickup": request["jam_pickup"],
                "Kota Tujuan": request["kota_tujuan"],
                "Jarak (km)": request["jarak_km"],
                "Layanan": request["layanan"],
                "Janji Pengiriman (hari)": request["janji_hari"],
                "Berat (kg)": request["berat_kg"],
                "Metode Pembayaran": request["metode_bayar"],
                "Cuaca": request["cuaca_saat_pickup"]
            })

    except Exception as e:

        st.error(
            f"Terjadi kesalahan saat melakukan prediksi: {e}"
        )

        st.exception(e)
