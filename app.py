
import json
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

MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
METADATA_PATH = BASE_DIR / "artifacts" / "metadata.json"


# =========================================================
# 2. LOAD MODEL DAN METADATA
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    metadata = load_metadata()

except Exception as e:
    st.error(f"Gagal memuat model atau metadata: {e}")
    st.stop()


# =========================================================
# 3. KONFIGURASI MODEL
# =========================================================

THRESHOLD = float(metadata["threshold"])
BATAS_PRIORITAS = float(metadata["batas_prioritas"])

MEDIAN_JARAK_KOTA = metadata["median_jarak_kota"]
MEDIAN_JARAK_GLOBAL = metadata["median_jarak_global"]

FITUR_REQUEST = metadata["fitur_request"]


# =========================================================
# 4. FUNGSI PREPROCESSING DAN PREDIKSI
# =========================================================

def proses_prediksi(request):

    # Membuat DataFrame sesuai fitur request
    df = pd.DataFrame(
        [request],
        columns=FITUR_REQUEST
    )

    # Validasi data input
    masalah, peringatan = periksa_request(request)

    if masalah:
        return None, masalah, peringatan

    # Feature engineering
    df_fitur = tambah_fitur(df)

    # Mengambil kota tujuan yang sudah dinormalisasi
    kota = df_fitur["kota_tujuan"].iloc[0]

    # Menentukan median jarak berdasarkan kota
    if kota in MEDIAN_JARAK_KOTA:
        median_jarak = MEDIAN_JARAK_KOTA[kota]
        sumber_jarak = f"Median kota {kota}"
    else:
        median_jarak = MEDIAN_JARAK_GLOBAL
        sumber_jarak = "Median jarak global"

    # Imputasi jarak jika tidak diketahui
    jarak_diimputasi = pd.isna(request["jarak_km"])

    df_fitur["jarak_km"] = df_fitur["jarak_km"].fillna(
        median_jarak
    )

    # Menghitung ulang fitur turunan setelah imputasi
    df_fitur["hari_per_1000km"] = (
        df_fitur["janji_hari"] /
        (df_fitur["jarak_km"] / 1000)
    )

    # Mengambil fitur sesuai Pipeline model
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

    # Memastikan kelas terlambat tersedia
    kelas = list(classifier.classes_)

    if 1 not in kelas:
        raise ValueError(
            "Model tidak memiliki kelas positif 1 (terlambat)."
        )

    indeks_terlambat = kelas.index(1)

    # Menghitung probabilitas keterlambatan
    probabilitas = model.predict_proba(X)[0][indeks_terlambat]

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

    # Menyimpan hasil prediksi
    hasil = {
        "probabilitas": probabilitas,
        "prediksi": prediksi,
        "rekomendasi": rekomendasi,
        "keterangan": keterangan,
        "jarak_digunakan": df_fitur["jarak_km"].iloc[0],
        "jarak_diimputasi": jarak_diimputasi,
        "sumber_jarak": sumber_jarak
    }

    return hasil, masalah, peringatan


# =========================================================
# 5. HEADER APLIKASI
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
# 6. INPUT DATA PAKET
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

    # Checkbox tepat di atas kolom jarak
    tidak_tahu_jarak = st.checkbox(
        "Jarak pengiriman tidak diketahui"
    )

    # Jika checkbox dicentang
    if tidak_tahu_jarak:

        jarak_km = None

        st.text_input(
            "Jarak Pengiriman (km)",
            value="Unable",
            disabled=True
        )

    # Jika checkbox tidak dicentang
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
# 7. TOMBOL PREDIKSI
# =========================================================

tombol_prediksi = st.button(
    "🔍 Prediksi Keterlambatan",
    use_container_width=True
)


# =========================================================
# 8. PROSES HASIL PREDIKSI
# =========================================================

if tombol_prediksi:

    # Membuat request
    request = {
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

        hasil, masalah, peringatan = proses_prediksi(request)

        # -------------------------------------------------
        # VALIDASI INPUT
        # -------------------------------------------------

        if masalah:

            st.error("Data belum dapat diproses.")

            for item in masalah:
                st.write(f"- {item}")

        else:

            # Menampilkan peringatan
            for item in peringatan:
                st.warning(item)

            st.divider()

            # -------------------------------------------------
            # HASIL PREDIKSI
            # -------------------------------------------------

            st.subheader("Hasil Prediksi")

            probabilitas = hasil["probabilitas"]

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Probabilitas Terlambat",
                    f"{probabilitas:.2%}"
                )

            with col2:

                st.metric(
                    "Threshold Model",
                    f"{THRESHOLD:.0%}"
                )

            with col3:

                st.metric(
                    "Jarak yang Digunakan",
                    f"{hasil['jarak_digunakan']:.1f} km"
                )

            # Progress bar probabilitas
            st.progress(
                float(probabilitas),
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
            # CATATAN PREDIKSI
            # -------------------------------------------------

            st.caption(
                "Hasil prediksi merupakan estimasi berdasarkan "
                "model machine learning, bukan kepastian bahwa "
                "paket akan mengalami keterlambatan."
            )

    except Exception as e:

        st.error(
            f"Terjadi kesalahan saat melakukan prediksi: {e}"
        )


# =========================================================
# 9. FOOTER
# =========================================================

st.divider()

st.caption(
    "Sistem Prediksi Keterlambatan Paket | "
    f"Model {metadata['versi_model']} | "
    f"Dibuat pada {metadata['dibuat']}"
)
