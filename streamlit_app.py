"""Aplikasi Streamlit: Kasus 08 · Prediksi Keterlambatan Paket (Hub Bandung)
Disesuaikan sepenuhnya dengan 08_keterlambatan_paket_revisi_3.ipynb.
Jalankan lokal: streamlit run streamlit_app.py
"""
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from core import JARAK_KOTA, META, prediksi  # type: ignore

st.set_page_config(
    page_title="Kasus 08 · Prediksi Keterlambatan Paket (Hub Bandung)",
    page_icon="📦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .badge {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 6px;
        color: #ffffff;
        font-weight: 600;
        font-size: 15px;
        margin-top: 4px;
        margin-bottom: 10px;
    }
    .AMAN { background-color: #2e8b57; }
    .KABARI { background-color: #d98b00; }
    .PRIORITAS { background-color: #c0392b; }
    .DITOLAK { background-color: #6c757d; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📦 Kasus 08 · Prediksi Keterlambatan Paket (Hub Bandung)")
st.caption(
    f"Model Final: **{META['nama_model']}** (v{META['versi_model']}) · "
    f"PR-AUC Validasi: **{META['metrik_validasi']['pr_auc']}** · "
    f"PR-AUC Test: **{META['metrik_test']['pr_auc']}** (Recall: {META['metrik_test']['recall']}, Precision: {META['metrik_test']['precision']}) · "
    f"Threshold Biaya (FN Rp50.000, FP Rp5.000): **{META['threshold']}**"
)

DAFTAR_KOTA = sorted(JARAK_KOTA.keys())

PRESETS = {
    "Manual / Kustom": None,
    "Contoh 1: Paket Risiko Tinggi (Medan, Express, Jam 19, Hujan Lebat, COD)": {
        "tanggal_pickup": datetime.date(2026, 7, 6),
        "jam_pickup": 19,
        "kota_pilihan": "Medan",
        "kota_manual": "",
        "jarak_km": "1900",
        "jumlah_transit": 4,
        "layanan": "EXPRESS",
        "janji_hari": 3,
        "berat_kg": "2.0",
        "metode_bayar": "COD",
        "cuaca_saat_pickup": "hujan_lebat",
        "volume_hub_hari": "5400",
    },
    "Contoh 2: Paket Risiko Rendah / Aman (Bogor, REG, Jam 9, Cerah, non_COD)": {
        "tanggal_pickup": datetime.date(2026, 7, 6),
        "jam_pickup": 9,
        "kota_pilihan": "Bogor",
        "kota_manual": "",
        "jarak_km": "125",
        "jumlah_transit": 1,
        "layanan": "REG",
        "janji_hari": 3,
        "berat_kg": "2.0",
        "metode_bayar": "non_COD",
        "cuaca_saat_pickup": "cerah",
        "volume_hub_hari": "5400",
    },
    "Contoh 3: Jarak & Berat Kosong (Otomatis Diisi Median Kota & Median Berat)": {
        "tanggal_pickup": datetime.date(2026, 7, 6),
        "jam_pickup": 19,
        "kota_pilihan": "Medan",
        "kota_manual": "",
        "jarak_km": "",
        "jumlah_transit": 4,
        "layanan": "EXPRESS",
        "janji_hari": 3,
        "berat_kg": "",
        "metode_bayar": "COD",
        "cuaca_saat_pickup": "hujan_lebat",
        "volume_hub_hari": "5400",
    },
    "Contoh 4: Berat dalam Gram (1500g), Alias Kota 'Jkt', Cuaca Kosong": {
        "tanggal_pickup": datetime.date(2026, 7, 6),
        "jam_pickup": 19,
        "kota_pilihan": "Lainnya / Ketik Manual (Termasuk Alias/Kota Baru)",
        "kota_manual": "Jkt",
        "jarak_km": "",
        "jumlah_transit": 4,
        "layanan": "EXPRESS",
        "janji_hari": 3,
        "berat_kg": "1500",
        "metode_bayar": "COD",
        "cuaca_saat_pickup": "tidak_diketahui",
        "volume_hub_hari": "5400",
    },
    "Contoh 5: Kota di Luar Data Latih ('Atlantis')": {
        "tanggal_pickup": datetime.date(2026, 7, 6),
        "jam_pickup": 19,
        "kota_pilihan": "Lainnya / Ketik Manual (Termasuk Alias/Kota Baru)",
        "kota_manual": "Atlantis",
        "jarak_km": "1900",
        "jumlah_transit": 4,
        "layanan": "EXPRESS",
        "janji_hari": 3,
        "berat_kg": "2.0",
        "metode_bayar": "COD",
        "cuaca_saat_pickup": "hujan_lebat",
        "volume_hub_hari": "5400",
    },
    "Contoh 6: Volume Ekstrem di Luar Rentang Latih (40.000 -> Dipotong ke 18.000)": {
        "tanggal_pickup": datetime.date(2026, 7, 6),
        "jam_pickup": 19,
        "kota_pilihan": "Medan",
        "kota_manual": "",
        "jarak_km": "1900",
        "jumlah_transit": 4,
        "layanan": "EXPRESS",
        "janji_hari": 3,
        "berat_kg": "2.0",
        "metode_bayar": "COD",
        "cuaca_saat_pickup": "hujan_lebat",
        "volume_hub_hari": "40000",
    },
}

col_form, col_out = st.columns([1.25, 1], gap="large")

with col_form:
    preset_nama = st.selectbox(
        "Pilih Skenario Uji (Sesuai Notebook Kasus 08)",
        options=list(PRESETS.keys()),
        index=1,
    )
    p_val = PRESETS[preset_nama] or PRESETS["Contoh 1: Paket Risiko Tinggi (Medan, Express, Jam 19, Hujan Lebat, COD)"]

    with st.form("form_prediksi"):
        c1, c2 = st.columns(2)
        with c1:
            tanggal_pickup = st.date_input(
                "Tanggal pickup",
                value=p_val["tanggal_pickup"],
            )
            opsi_kota = DAFTAR_KOTA + ["Lainnya / Ketik Manual (Termasuk Alias/Kota Baru)"]
            idx_kota = opsi_kota.index(p_val["kota_pilihan"]) if p_val["kota_pilihan"] in opsi_kota else 0
            kota_pilihan = st.selectbox(
                "Kota tujuan",
                options=opsi_kota,
                index=idx_kota,
            )
            kota_manual = st.text_input(
                "Nama kota manual / alias (jika memilih Lainnya, mis. Jkt, Sby, Atlantis)",
                value=p_val["kota_manual"],
            )
            opsi_layanan = ["EXPRESS", "REG", "HEMAT"]
            layanan = st.selectbox(
                "Layanan",
                options=opsi_layanan,
                index=opsi_layanan.index(p_val["layanan"]),
            )
            jumlah_transit = st.number_input(
                "Jumlah transit (1-6)",
                min_value=1,
                max_value=6,
                value=int(p_val["jumlah_transit"]),
                step=1,
            )
            opsi_mb = ["COD", "non_COD"]
            metode_bayar = st.selectbox(
                "Metode bayar",
                options=opsi_mb,
                index=opsi_mb.index(p_val["metode_bayar"]),
            )

        with c2:
            jam_pickup = st.number_input(
                "Jam pickup (7-22, cutoff >= 17)",
                min_value=0,
                max_value=23,
                value=int(p_val["jam_pickup"]),
                step=1,
            )
            jarak_km = st.text_input(
                "Jarak (km) — boleh kosong (diisi median kota)",
                value=p_val["jarak_km"],
            )
            janji_hari = st.number_input(
                "Janji hari (1-6)",
                min_value=1,
                max_value=6,
                value=int(p_val["janji_hari"]),
                step=1,
            )
            berat_kg = st.text_input(
                "Berat (kg) — boleh kosong / >=100 dianggap gram",
                value=p_val["berat_kg"],
            )
            opsi_cuaca = ["hujan_lebat", "cerah", "berawan", "hujan", "tidak_diketahui"]
            cuaca_saat_pickup = st.selectbox(
                "Cuaca saat pickup",
                options=opsi_cuaca,
                index=opsi_cuaca.index(p_val["cuaca_saat_pickup"]),
            )
            volume_hub_hari = st.text_input(
                "Volume hub hari ini (latih: 2.861–17.979, lonjakan > 9.000)",
                value=p_val["volume_hub_hari"],
            )

        submitted = st.form_submit_button("Prediksi Risiko Keterlambatan", use_container_width=True, type="primary")

with col_out:
    st.subheader("Hasil Prediksi & Rekomendasi")
    kota_final = (
        kota_manual.strip()
        if kota_pilihan.startswith("Lainnya")
        else kota_pilihan
    )
    payload = {
        "tanggal_pickup": tanggal_pickup.strftime("%Y-%m-%d"),
        "jam_pickup": jam_pickup,
        "kota_tujuan": kota_final,
        "jarak_km": None if jarak_km.strip() == "" else jarak_km.strip(),
        "jumlah_transit": jumlah_transit,
        "layanan": layanan,
        "janji_hari": janji_hari,
        "berat_kg": None if berat_kg.strip() == "" else berat_kg.strip(),
        "metode_bayar": metode_bayar,
        "cuaca_saat_pickup": None if cuaca_saat_pickup == "tidak_diketahui" else cuaca_saat_pickup,
        "volume_hub_hari": None if volume_hub_hari.strip() == "" else volume_hub_hari.strip(),
    }

    r = prediksi(payload)

    for err_msg in r.get("masalah", []):
        st.error(f"**Masalah:** {err_msg}")

    if r.get("status") != "DITOLAK" and r.get("ok"):
        prob = r["probabilitas_terlambat"]
        rek = r["rekomendasi"]
        kelas = (
            "AMAN" if rek.startswith("AMAN")
            else "KABARI" if rek.startswith("KABARI")
            else "PRIORITAS"
        )
        m1, m2 = st.columns(2)
        m1.metric("Probabilitas Terlambat", f"{prob * 100:.2f}%", f"Skor: {prob:.4f}")
        m2.metric("Threshold Keputusan", f"{r['threshold']:.2f}", f"Prioritas >= {r['batas_prioritas']:.2f}")

        st.markdown(f'<div class="badge {kelas}">{rek}</div>', unsafe_allow_html=True)

        if r.get("lonjakan"):
            st.warning("**Hari Lonjakan Volume (> 9.000):** Risiko keterlambatan meningkat tajam (67,3% pada data historis).")
        if not r.get("andal"):
            st.warning("**Prediksi Kurang Andal:** Terdapat masukan di luar rentang data latih.")

        for warn_msg in r.get("peringatan", []):
            st.warning(f"**Peringatan:** {warn_msg}")

        with st.expander("Lihat Output JSON (Format Notebook Kasus 08)", expanded=True):
            st.json({
                "probabilitas_terlambat": r["probabilitas_terlambat"],
                "rekomendasi": r["rekomendasi"],
                "peringatan": r["peringatan"],
                "threshold": r["threshold"],
                "versi_model": r["versi_model"],
                "latency_ms": r["latency_ms"],
            })
    else:
        st.markdown('<div class="badge DITOLAK">STATUS: DITOLAK</div>', unsafe_allow_html=True)
        st.error("**Prediksi tidak dijalankan.** Silakan perbaiki masukan yang tidak valid.")
        with st.expander("Lihat Output JSON (Format Notebook Kasus 08)", expanded=True):
            st.json({
                "status": "DITOLAK",
                "masalah": r.get("masalah", []),
            })
