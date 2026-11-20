
import json
import os
import requests
import streamlit as st

from pathlib import Path


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

# Alamat API lokal.
# Saat deployment, alamat ini akan diganti menggunakan
# environment variable API_URL.

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")


# =========================================================
# 2. LOAD METADATA
# =========================================================

@st.cache_data
def load_metadata():

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:

    metadata = load_metadata()

except Exception as e:

    st.error(f"Gagal memuat metadata: {e}")
    st.stop()


# =========================================================
# 3. KONFIGURASI MODEL
# =========================================================

THRESHOLD = float(metadata["threshold"])


# =========================================================
# 4. HEADER APLIKASI
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
# 5. INPUT DATA PAKET
# =========================================================

st.subheader("Input Data Paket")


# ---------------------------------------------------------
# INFORMASI PICKUP
# ---------------------------------------------------------

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
        [
            "REG",
            "EXPRESS",
            "HEMAT"
        ]
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
        [
            "COD",
            "non_COD"
        ]
    )


# ---------------------------------------------------------
# INFORMASI PAKET
# ---------------------------------------------------------

st.markdown("#### Informasi Paket")

col3, col4 = st.columns(2)

with col3:

    # Checkbox untuk jarak yang tidak diketahui
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


# ---------------------------------------------------------
# INFORMASI OPERASIONAL HUB
# ---------------------------------------------------------

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
# 6. TOMBOL PREDIKSI
# =========================================================

tombol_prediksi = st.button(
    "🔍 Prediksi Keterlambatan",
    use_container_width=True
)


# =========================================================
# 7. PROSES PREDIKSI MELALUI API
# =========================================================

if tombol_prediksi:

    # -----------------------------------------------------
    # MEMBUAT DATA REQUEST
    # -----------------------------------------------------

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

    try:

        # -------------------------------------------------
        # MENGIRIM DATA KE FASTAPI
        # -------------------------------------------------

        with st.spinner("Sedang memproses prediksi..."):

            response = requests.post(
                f"{API_URL}/predict",
                json=request_data,
                timeout=30
            )

        # Mengambil respons JSON
        try:
            hasil = response.json()

        except ValueError:
            hasil = {}

        # -------------------------------------------------
        # MENANGANI ERROR DARI API
        # -------------------------------------------------

        if response.status_code != 200:

            detail = hasil.get(
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

        else:

            # -------------------------------------------------
            # MENAMPILKAN PERINGATAN
            # -------------------------------------------------

            for item in hasil.get("peringatan", []):
                st.warning(item)

            st.divider()

            # -------------------------------------------------
            # HASIL PREDIKSI
            # -------------------------------------------------

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

            # Progress bar probabilitas
            st.progress(
                min(max(probabilitas, 0.0), 1.0),
                text="Tingkat Risiko Keterlambatan"
            )

            # -------------------------------------------------
            # REKOMENDASI TINDAKAN
            # -------------------------------------------------

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

            # -------------------------------------------------
            # INFORMASI IMPUTASI JARAK
            # -------------------------------------------------

            if hasil["jarak_diimputasi"]:

                st.info(
                    f"Jarak tidak diketahui. Sistem menggunakan "
                    f"{hasil['sumber_jarak']} sebesar "
                    f"{hasil['jarak_digunakan']:.1f} km."
                )

            # -------------------------------------------------
            # STATUS PREDIKSI
            # -------------------------------------------------

            st.caption(
                "Hasil prediksi merupakan estimasi berdasarkan "
                "model machine learning, bukan kepastian bahwa "
                "paket akan mengalami keterlambatan."
            )

    # -----------------------------------------------------
    # PENANGANAN KONEKSI API
    # -----------------------------------------------------

    except requests.exceptions.ConnectionError:

        st.error(
            "Tidak dapat terhubung ke API. "
            "Pastikan FastAPI sedang berjalan."
        )

    except requests.exceptions.Timeout:

        st.error(
            "Waktu permintaan API habis. Silakan coba kembali."
        )

    except requests.exceptions.RequestException as e:

        st.error(
            f"Terjadi kesalahan komunikasi dengan API: {e}"
        )

    except Exception as e:

        st.error(
            f"Terjadi kesalahan saat melakukan prediksi: {e}"
        )


# =========================================================
# 8. FOOTER
# =========================================================

st.divider()

st.caption(
    "Sistem Prediksi Keterlambatan Paket | "
    f"Model {metadata['versi_model']} | "
    f"Dibuat pada {metadata['dibuat']}"
)
