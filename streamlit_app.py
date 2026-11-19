

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
import sys
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

APP_DIR = Path(__file__).resolve().parent

# Pastikan root proyek & subfolder umum masuk ke sys.path
for candidate_dir in [APP_DIR, APP_DIR / "src", APP_DIR / "app"]:
    if candidate_dir.exists() and str(candidate_dir) not in sys.path:
        sys.path.insert(0, str(candidate_dir))

# ---------------------------------------------------------------------------
# Konstanta Domain & Prapemrosesan (Hub Bandung)
# ---------------------------------------------------------------------------
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

MAPPING_KOTA = {
    "bali": "Denpasar",
    "denpasar": "Denpasar",
    "bandung": "Bandung",
    "bekasi": "Bekasi",
    "bogor": "Bogor",
    "cirebon": "Cirebon",
    "depok": "Depok",
    "jakarta": "Jakarta",
    "makassar": "Makassar",
    "malang": "Malang",
    "medan": "Medan",
    "palembang": "Palembang",
    "semarang": "Semarang",
    "surabaya": "Surabaya",
    "tangerang": "Tangerang",
    "yogyakarta": "Yogyakarta",
    "jogja": "Yogyakarta",
}

JARAK_KOTA_MEDIAN = {
    "Bandung": 15.0,
    "Bekasi": 135.0,
    "Bogor": 120.0,
    "Cirebon": 130.0,
    "Denpasar": 1070.0,
    "Depok": 140.0,
    "Jakarta": 150.0,
    "Makassar": 1650.0,
    "Malang": 780.0,
    "Medan": 1850.0,
    "Palembang": 630.0,
    "Semarang": 360.0,
    "Surabaya": 700.0,
    "Tangerang": 170.0,
    "Yogyakarta": 400.0,
}

LAYANAN_LIST = ["Reguler", "Express", "Same Day", "Kargo", "Hemat"]
CUACA_LIST = ["Cerah", "Berawan", "Hujan Ringan", "Hujan Sedang", "Hujan Lebat"]


# ---------------------------------------------------------------------------
# Deteksi Otomatis Backend: core.py -> *.joblib -> Mode Mandiri
# ---------------------------------------------------------------------------
def _muat_core_module():
    """Mencoba mengimpor fungsi prediksi() dari core.py di root atau subfolder."""
    try:
        from core import prediksi  # type: ignore

        return prediksi, "core.py"
    except Exception:
        pass

    for core_path in APP_DIR.rglob("core.py"):
        try:
            spec = importlib.util.spec_from_file_location("core", core_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["core"] = mod
                spec.loader.exec_module(mod)
                if hasattr(mod, "prediksi"):
                    return mod.prediksi, str(core_path.relative_to(APP_DIR))
        except Exception:
            continue
    return None, None


def _muat_joblib_model():
    """Mencoba memuat model .joblib langsung jika core.py tidak tersedia."""
    try:
        import joblib  # type: ignore
    except Exception:
        return None, None

    kandidat = [
        APP_DIR / "model_terlambat.joblib",
        APP_DIR / "artifacts" / "model_terlambat.joblib",
        APP_DIR / "model" / "model_terlambat.joblib",
    ]
    kandidat.extend(APP_DIR.rglob("*.joblib"))

    for path in kandidat:
        if path.exists():
            try:
                return joblib.load(path), str(path.relative_to(APP_DIR))
            except Exception:
                continue
    return None, None


CORE_PREDIKSI, CORE_SOURCE = _muat_core_module()
JOBLIB_MODEL, JOBLIB_SOURCE = (None, None) if CORE_PREDIKSI else _muat_joblib_model()

if CORE_PREDIKSI is not None:
    BACKEND_MODE = "core"
    BACKEND_DESC = f"Menggunakan `{CORE_SOURCE}`"
elif JOBLIB_MODEL is not None:
    BACKEND_MODE = "joblib"
    BACKEND_DESC = f"Menggunakan model langsung (`{JOBLIB_SOURCE}`)"
else:
    BACKEND_MODE = "mandiri"
    BACKEND_DESC = "Mode Mandiri (estimasi operasional tanpa file eksternal)"


@st.cache_data
def load_meta() -> dict[str, Any]:
    default_meta: dict[str, Any] = {
        "versi_model": "1.0.0",
        "model": "Logistic Regression + Fitur Rekayasa",
        "threshold": 0.11,
        "batas_prioritas": 0.50,
        "pr_auc_val": 0.6623,
        "pr_auc_test": 0.378,
        "recall_test": 0.772,
    }
    for meta_path in [APP_DIR / "meta.json", *APP_DIR.rglob("meta.json")]:
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                default_meta.update(loaded)
                break
            except Exception:
                pass
    return default_meta


META = load_meta()
THRESHOLD = float(META.get("threshold", 0.11))
BATAS_PRIORITAS = float(META.get("batas_prioritas", 0.50))


# ---------------------------------------------------------------------------
# Validasi Input & Rekayasa Fitur Mandiri
# ---------------------------------------------------------------------------
def periksa_dan_siapkan_fitur(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Membersihkan input, mendeteksi anomali, dan membentuk 7 fitur rekayasa."""
    peringatan: list[str] = []

    kota_raw = str(payload.get("kota_tujuan") or "Jakarta").strip()
    kota = MAPPING_KOTA.get(kota_raw.lower(), kota_raw.title())
    if kota not in JARAK_KOTA_MEDIAN:
        peringatan.append(f"Kota tujuan '{kota_raw}' di luar 15 kota utama; jarak diestimasi.")

    berat = float(payload.get("berat_kg") or 1.0)
    if berat >= 100:
        peringatan.append(f"Berat {berat:.1f} terdeteksi dalam satuan gram, otomatis dikonversi ke {berat / 1000:.3f} kg.")
        berat = berat / 1000.0
    elif berat <= 0:
        peringatan.append("Berat <= 0 tidak valid; otomatis disesuaikan ke 1.0 kg.")
        berat = 1.0

    jarak = payload.get("jarak_km")
    if jarak is None or float(jarak) <= 0:
        jarak = JARAK_KOTA_MEDIAN.get(kota, 300.0)
        peringatan.append(f"Jarak kosong/0; diimputasi menggunakan median kota {kota} ({jarak:.0f} km).")
    else:
        jarak = float(jarak)

    vol_raw = int(payload.get("volume_hub_hari") or 6000)
    vol_clipped = min(max(vol_raw, 0), 18000)
    if vol_raw > 18000:
        peringatan.append(f"Volume hub {vol_raw:,} melebihi batas atas, dipotong (clip) ke 18.000.")

    jam = int(payload.get("jam_pickup") if payload.get("jam_pickup") is not None else 12)
    transit = int(payload.get("jumlah_transit") if payload.get("jumlah_transit") is not None else 1)
    janji = max(int(payload.get("janji_hari") or 2), 1)
    cuaca = str(payload.get("cuaca") or "Cerah")
    layanan = str(payload.get("layanan") or "Reguler")

    tgl_str = str(payload.get("tanggal_pickup") or dt.date.today().isoformat())
    try:
        tgl = dt.date.fromisoformat(tgl_str[:10])
    except Exception:
        tgl = dt.date.today()

    after_cutoff = 1 if jam >= 17 else 0
    senin = 1 if tgl.weekday() == 0 else 0
    slack = janji - transit
    hari_per_1000km = janji / max(jarak / 1000.0, 0.1)
    surge = 1 if vol_clipped >= 12000 else 0
    log_vol = math.log1p(vol_clipped)
    hujan_lebat = 1 if cuaca.lower() == "hujan lebat" else 0

    fitur = {
        "tanggal_pickup": tgl.isoformat(),
        "jam_pickup": jam,
        "kota_tujuan": kota,
        "layanan": layanan,
        "berat_kg": berat,
        "jarak_km": jarak,
        "jumlah_transit": transit,
        "janji_hari": janji,
        "volume_hub_hari": vol_clipped,
        "cuaca": cuaca,
        "jenis_paket": str(payload.get("jenis_paket") or "Umum"),
        "after_cutoff": after_cutoff,
        "senin": senin,
        "slack": slack,
        "hari_per_1000km": hari_per_1000km,
        "surge": surge,
        "log_vol": log_vol,
        "hujan_lebat": hujan_lebat,
    }
    return fitur, peringatan


def _estimasi_skor_mandiri(fitur: dict[str, Any]) -> float:
    """Skor risiko logistik berbasis aturan operasional saat file model eksternal belum tersedia."""
    logit = -2.65
    logit += 0.85 * fitur["after_cutoff"]
    logit += 0.35 * fitur["senin"]
    logit += 0.90 * fitur["surge"]
    logit += 0.75 * fitur["hujan_lebat"]
    logit += 0.38 * max(fitur["jumlah_transit"] - 1, 0)
    logit += 0.55 if fitur["slack"] <= 0 else -0.25 * min(fitur["slack"], 3)
    logit += 0.00055 * max(fitur["jarak_km"] - 300, 0)
    if fitur["layanan"] in {"Hemat", "Kargo"}:
        logit += 0.35
    elif fitur["layanan"] in {"Express", "Same Day"}:
        logit -= 0.45
    prob = 1.0 / (1.0 + math.exp(-logit))
    return min(max(prob, 0.01), 0.99)


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
    """Menjalankan prediksi via core.py, model .joblib langsung, atau mode mandiri."""
    fitur, peringatan = periksa_dan_siapkan_fitur(payload)
    raw_res: dict[str, Any] = {}

    if BACKEND_MODE == "core" and CORE_PREDIKSI is not None:
        raw_res = CORE_PREDIKSI(payload)
        if not isinstance(raw_res, dict):
            raw_res = {"probabilitas": float(raw_res)}
        prob = float(
            raw_res.get("probabilitas")
            or raw_res.get("prob")
            or raw_res.get("probability")
            or raw_res.get("skor_risiko")
            or 0.0
        )
        peringatan_core = raw_res.get("peringatan") or raw_res.get("warnings") or []
        if isinstance(peringatan_core, list):
            peringatan = list(dict.fromkeys([*peringatan_core, *peringatan]))
    elif BACKEND_MODE == "joblib" and JOBLIB_MODEL is not None:
        df_row = pd.DataFrame([fitur])
        if hasattr(JOBLIB_MODEL, "predict_proba"):
            prob = float(JOBLIB_MODEL.predict_proba(df_row)[0][1])
        else:
            prob = float(JOBLIB_MODEL.predict(df_row)[0])
    else:
        prob = _estimasi_skor_mandiri(fitur)

    label_default, status_type, aksi_default = klasifikasi_risiko(prob, THRESHOLD, BATAS_PRIORITAS)

    return {
        "probabilitas": prob,
        "label": raw_res.get("kategori") or raw_res.get("label") or raw_res.get("status") or label_default,
        "status_type": status_type,
        "rekomendasi": raw_res.get("rekomendasi") or raw_res.get("aksi") or aksi_default,
        "peringatan": peringatan,
        "fitur": fitur,
    }


# ---------------------------------------------------------------------------
# Sidebar Informasi Model
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📦 Hub Bandung")
    st.caption("Sistem Peringatan Dini Keterlambatan Paket")
    st.divider()

    st.subheader("Status Mesin Prediksi")
    if BACKEND_MODE == "mandiri":
        st.warning(f"**{BACKEND_DESC}** — unggah `core.py` atau `model_terlambat.joblib` untuk bobot terlatih.")
    else:
        st.success(f"**Aktif:** {BACKEND_DESC}")

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

tab_single, tab_batch = st.tabs(["🔍 Prediksi Paket Tunggal", "📂 Prediksi Batch (Upload CSV)"])


# ---------------------------------------------------------------------------
# Tab 1: Prediksi Paket Tunggal
# ---------------------------------------------------------------------------
with tab_single:
    preset_col1, preset_col2, _ = st.columns([1, 1, 2])
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
                    for item in peringatan:
                        st.warning(str(item))

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
