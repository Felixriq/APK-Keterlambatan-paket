"""
streamlit_app.py
Aplikasi Web Prediksi Keterlambatan Pengiriman Paket — Hub Bandung
Dirancang untuk deployment di Streamlit Community Cloud bersama core.py, meta.json, dan model_terlambat.joblib.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Konfigurasi Halaman
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Keterlambatan Paket — Hub Bandung",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Muat Modul Core & Metadata Model
# ---------------------------------------------------------------------------
CORE_ERROR: str | None = None
try:
    from core import prediksi  # type: ignore
except Exception as exc:
    prediksi = None
    CORE_ERROR = str(exc)


@st.cache_data
def load_meta() -> dict[str, Any]:
    meta_path = Path(__file__).parent / "meta.json"
    default_meta: dict[str, Any] = {
        "versi_model": "1.0.0",
        "model": "Logistic Regression + Fitur Rekayasa",
        "threshold": 0.11,
        "batas_prioritas": 0.50,
        "pr_auc_val": 0.6623,
        "pr_auc_test": 0.378,
        "recall_test": 0.772,
    }
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            default_meta.update(loaded)
        except Exception:
            pass
    return default_meta


META = load_meta()
THRESHOLD = float(META.get("threshold", 0.11))
BATAS_PRIORITAS = float(META.get("batas_prioritas", 0.50))

KOTA_TUJUAN_LIST = [
    "Bandung",
    "Bekasi",
    "Bogor",
    "Cirebon",
    "Denpasar",
    "Depok",
    "Jakarta",
    "Makassar",
    "Malang",
    "Medan",
    "Palembang",
    "Semarang",
    "Surabaya",
    "Tangerang",
    "Yogyakarta",
]

LAYANAN_LIST = ["Reguler", "Express", "Same Day", "Kargo", "Hemat"]
CUACA_LIST = ["Cerah", "Berawan", "Hujan Ringan", "Hujan Sedang", "Hujan Lebat"]


# ---------------------------------------------------------------------------
# Utilitas Normalisasi Output core.prediksi()
# ---------------------------------------------------------------------------
def klasifikasi_risiko(prob: float, threshold: float, batas_prioritas: float) -> tuple[str, str, str]:
    """Mengembalikan (label, warna_status, rekomendasi_aksi)."""
    if prob >= batas_prioritas:
        return (
            "PRIORITAS INTERVENSI",
            "error",
            "Segera jadwalkan penanganan prioritas, verifikasi kapasitas armada/transit, dan kirim notifikasi proaktif ke tim operasional.",
        )
    if prob >= threshold:
        return (
            "PANTAU (WASPADA)",
            "warning",
            "Tandai resi dalam daftar pantauan Hub Bandung; pastikan paket masuk kloter muat sebelum jam cutoff berikutnya.",
        )
    return (
        "AMAN (TEPAT WAKTU)",
        "success",
        "Proses sesuai alur operasional standar tanpa intervensi tambahan.",
    )


def jalankan_prediksi(payload: dict[str, Any]) -> dict[str, Any]:
    """Memanggil core.prediksi(payload) dan menormalkan struktur respons."""
    if prediksi is None:
        raise RuntimeError(
            f"Modul 'core.py' atau artefak model belum tersedia ({CORE_ERROR}). "
            "Pastikan core.py, meta.json, dan model_terlambat.joblib berada di folder yang sama dengan streamlit_app.py."
        )

    raw_res = prediksi(payload)
    if not isinstance(raw_res, dict):
        raw_res = {"probabilitas": float(raw_res)}

    prob = float(
        raw_res.get("probabilitas")
        or raw_res.get("prob")
        or raw_res.get("probability")
        or raw_res.get("skor_risiko")
        or 0.0
    )
    label_default, status_type, aksi_default = klasifikasi_risiko(prob, THRESHOLD, BATAS_PRIORITAS)

    return {
        "probabilitas": prob,
        "label": raw_res.get("kategori") or raw_res.get("label") or raw_res.get("status") or label_default,
        "status_type": status_type,
        "rekomendasi": raw_res.get("rekomendasi") or raw_res.get("aksi") or aksi_default,
        "peringatan": raw_res.get("peringatan") or raw_res.get("warnings") or [],
        "raw": raw_res,
    }


# ---------------------------------------------------------------------------
# Sidebar Informasi Model
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📦 Hub Bandung")
    st.caption("Sistem Peringatan Dini Keterlambatan Paket")
    st.divider()

    st.subheader("Spesifikasi Model")
    st.markdown(
        f"""
        - **Model:** {META.get('model', 'Logistic Regression + Fitur')}
        - **Versi:** `{META.get('versi_model', '1.0.0')}`
        - **Threshold Waspada:** `{THRESHOLD:.2f}`
        - **Batas Prioritas:** `{BATAS_PRIORITAS:.2f}`
        """
    )

    st.subheader("Performa Evaluasi")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("PR-AUC Val", f"{float(META.get('pr_auc_val', 0.6623)):.4f}")
    col_m2.metric("PR-AUC Test", f"{float(META.get('pr_auc_test', 0.378)):.3f}")
    st.metric("Recall Test", f"{float(META.get('recall_test', 0.772)) * 100:.1f}%")

    st.divider()
    st.info(
        "**Catatan Operasional:** Pastikan nilai `volume_hub_hari` yang dimasukkan merupakan "
        "*running count* saat paket dijemput (*pickup*), bukan total akhir hari.",
        icon="ℹ️",
    )


# ---------------------------------------------------------------------------
# Header Utama
# ---------------------------------------------------------------------------
st.title("📦 Prediksi Keterlambatan Pengiriman Paket — Hub Bandung")
st.markdown(
    "Aplikasi ini membantu tim operasional Hub Bandung mengidentifikasi paket berisiko terlambat "
    "lebih awal berdasarkan data penjemputan, rute, beban hub, dan kondisi cuaca."
)

if CORE_ERROR is not None:
    st.error(
        f"**Gagal memuat `core.py`:** `{CORE_ERROR}`\n\n"
        "Pastikan file `core.py`, `meta.json`, dan `model_terlambat.joblib` sudah diunggah ke "
        "root repositori GitHub Anda bersama `streamlit_app.py`."
    )

tab_single, tab_batch = st.tabs(["🔍 Prediksi Paket Tunggal", "📂 Prediksi Batch (Upload CSV)"])


# ---------------------------------------------------------------------------
# Tab 1: Prediksi Paket Tunggal
# ---------------------------------------------------------------------------
with tab_single:
    # Preset cepat untuk pengujian
    preset_col1, preset_col2, preset_col3 = st.columns([1, 1, 2])
    if preset_col1.button("Contoh Risiko Rendah", use_container_width=True):
        st.session_state["preset"] = "rendah"
    if preset_col2.button("Contoh Risiko Tinggi", use_container_width=True):
        st.session_state["preset"] = "tinggi"

    preset = st.session_state.get("preset", "rendah")
    if preset == "tinggi":
        def_jam, def_kota, def_layanan, def_berat = 19, "Makassar", "Hemat", 8.5
        def_jarak, def_transit, def_janji, def_vol, def_cuaca = 1650.0, 3, 2, 16500, "Hujan Lebat"
    else:
        def_jam, def_kota, def_layanan, def_berat = 10, "Jakarta", "Express", 1.2
        def_jarak, def_transit, def_janji, def_vol, def_cuaca = 150.0, 1, 2, 6500, "Cerah"

    with st.form("form_prediksi_tunggal"):
        st.subheader("Data Pengiriman Paket")
        c1, c2, c3 = st.columns(3)

        with c1:
            tanggal_pickup = st.date_input("Tanggal Pickup", value=dt.date.today())
            jam_pickup = st.number_input("Jam Pickup (0–23)", min_value=0, max_value=23, value=def_jam, step=1)
            kota_tujuan = st.selectbox(
                "Kota Tujuan",
                options=KOTA_TUJUAN_LIST,
                index=KOTA_TUJUAN_LIST.index(def_kota),
            )
            layanan = st.selectbox(
                "Jenis Layanan",
                options=LAYANAN_LIST,
                index=LAYANAN_LIST.index(def_layanan),
            )

        with c2:
            berat_kg = st.number_input("Berat Paket (kg)", min_value=0.01, max_value=50000.0, value=float(def_berat), step=0.1)
            jarak_km = st.number_input("Jarak Tempuh (km)", min_value=0.0, max_value=10000.0, value=float(def_jarak), step=10.0)
            jumlah_transit = st.number_input("Jumlah Transit", min_value=0, max_value=10, value=def_transit, step=1)
            janji_hari = st.number_input("Janji Estimasi (hari)", min_value=1, max_value=30, value=def_janji, step=1)

        with c3:
            volume_hub_hari = st.number_input(
                "Volume Hub Saat Ini (paket/hari)",
                min_value=0,
                max_value=50000,
                value=def_vol,
                step=100,
                help="Nilai di atas 18.000 akan otomatis dipotong (clip) oleh prapemrosesan model.",
            )
            cuaca = st.selectbox(
                "Kondisi Cuaca",
                options=CUACA_LIST,
                index=CUACA_LIST.index(def_cuaca),
            )
            jenis_paket = st.selectbox("Kategori Barang", options=["Umum", "Elektronik", "Dokumen", "Pecah Belah", "Lainnya"])
            submit_btn = st.form_submit_button("Prediksi Risiko Keterlambatan", type="primary", use_container_width=True)

    if submit_btn:
        payload = {
            "tanggal_pickup": tanggal_pickup.isoformat(),
            "jam_pickup": int(jam_pickup),
            "kota_tujuan": kota_tujuan,
            "layanan": layanan,
            "berat_kg": float(berat_kg),
            "jarak_km": float(jarak_km),
            "jumlah_transit": int(jumlah_transit),
            "janji_hari": int(janji_hari),
            "volume_hub_hari": int(volume_hub_hari),
            "cuaca": cuaca,
            "jenis_paket": jenis_paket,
        }

        try:
            hasil = jalankan_prediksi(payload)
            prob = hasil["probabilitas"]
            label = hasil["label"]
            status_type = hasil["status_type"]
            rekomendasi = hasil["rekomendasi"]
            peringatan = hasil["peringatan"]

            st.divider()
            st.subheader("Hasil Prediksi")

            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric("Probabilitas Terlambat", f"{prob * 100:.1f}%")
                st.progress(min(max(prob, 0.0), 1.0))

            with res_col2:
                if status_type == "error":
                    st.error(f"**Status: {label}**\n\n{rekomendasi}")
                elif status_type == "warning":
                    st.warning(f"**Status: {label}**\n\n{rekomendasi}")
                else:
                    st.success(f"**Status: {label}**\n\n{rekomendasi}")

            if peringatan:
                with st.expander("Peringatan Validasi Input", expanded=True):
                    if isinstance(peringatan, list):
                        for item in peringatan:
                            st.warning(str(item))
                    else:
                        st.warning(str(peringatan))

        except Exception as exc:
            st.error(f"Terjadi kesalahan saat menjalankan prediksi: {exc}")


# ---------------------------------------------------------------------------
# Tab 2: Prediksi Batch (Upload CSV)
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Unggah File CSV Pengiriman")
    st.markdown(
        "Unggah file CSV yang memuat kolom input pengiriman (misalnya `tanggal_pickup`, `jam_pickup`, "
        "`kota_tujuan`, `layanan`, `berat_kg`, `jarak_km`, `jumlah_transit`, `janji_hari`, `volume_hub_hari`, `cuaca`)."
    )

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.write(f"Pratinjau data ({len(df_input):,} baris):")
            st.dataframe(df_input.head(10), use_container_width=True)

            if st.button("Jalankan Prediksi Batch", type="primary"):
                if prediksi is None:
                    st.error("Modul `core.py` belum tersedia sehingga prediksi batch tidak dapat dijalankan.")
                else:
                    hasil_rows = []
                    progress_bar = st.progress(0)
                    total = len(df_input)

                    for idx, row in df_input.iterrows():
                        row_dict = row.where(pd.notnull(row), None).to_dict()
                        try:
                            res = jalankan_prediksi(row_dict)
                            hasil_rows.append(
                                {
                                    "probabilitas_terlambat": round(res["probabilitas"], 4),
                                    "status_risiko": res["label"],
                                    "rekomendasi_aksi": res["rekomendasi"],
                                }
                            )
                        except Exception as row_exc:
                            hasil_rows.append(
                                {
                                    "probabilitas_terlambat": None,
                                    "status_risiko": "ERROR",
                                    "rekomendasi_aksi": str(row_exc),
                                }
                            )
                        if total > 0:
                            progress_bar.progress(min(int(((idx + 1) / total) * 100), 100))

                    df_out = pd.concat([df_input.reset_index(drop=True), pd.DataFrame(hasil_rows)], axis=1)

                    jml_prioritas = int((df_out["status_risiko"] == "PRIORITAS INTERVENSI").sum())
                    jml_pantau = int((df_out["status_risiko"] == "PANTAU (WASPADA)").sum())
                    jml_aman = int((df_out["status_risiko"] == "AMAN (TEPAT WAKTU)").sum())

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Paket", f"{total:,}")
                    m2.metric("Prioritas Intervensi", f"{jml_prioritas:,}")
                    m3.metric("Pantau (Waspada)", f"{jml_pantau:,}")
                    m4.metric("Aman", f"{jml_aman:,}")

                    st.dataframe(df_out, use_container_width=True)

                    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Unduh Hasil Prediksi (CSV)",
                        data=csv_bytes,
                        file_name="hasil_prediksi_keterlambatan.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
        except Exception as exc:
            st.error(f"Gagal membaca atau memproses file CSV: {exc}")