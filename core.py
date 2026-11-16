"""Logika inti aplikasi mandiri (self-contained) sesuai 08_keterlambatan_paket_revisi_3.ipynb.
Tidak bergantung pada folder eksternal src/ atau artifacts/ sehingga aman dijalankan langsung di Streamlit Cloud maupun Flask lokal.
"""
import json
import os
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

MAPPING_KOTA = {
    "bekasi": "Bekasi",
    "kota bekasi": "Bekasi",
    "jakarta": "Jakarta",
    "dki jakarta": "Jakarta",
    "jkt": "Jakarta",
    "yogyakarta": "Yogyakarta",
    "di yogyakarta": "Yogyakarta",
    "jogja": "Yogyakarta",
    "yogya": "Yogyakarta",
    "surabaya": "Surabaya",
    "sby": "Surabaya",
    "medan": "Medan",
    "denpasar": "Denpasar",
    "bali": "Denpasar",
    "balikpapan": "Balikpapan",
    "bogor": "Bogor",
    "cirebon": "Cirebon",
    "makassar": "Makassar",
    "malang": "Malang",
    "palembang": "Palembang",
    "pontianak": "Pontianak",
    "semarang": "Semarang",
    "tasikmalaya": "Tasikmalaya",
}

JARAK_KOTA = {
    "Balikpapan": 1400.0,
    "Bekasi": 130.0,
    "Bogor": 125.0,
    "Cirebon": 130.0,
    "Denpasar": 1080.0,
    "Jakarta": 150.0,
    "Makassar": 1498.0,
    "Malang": 720.0,
    "Medan": 1899.0,
    "Palembang": 860.0,
    "Pontianak": 900.0,
    "Semarang": 370.0,
    "Surabaya": 691.0,
    "Tasikmalaya": 110.0,
    "Yogyakarta": 390.0,
}

JAM_CUTOFF = 17
BATAS_LONJAKAN = 9000
KOLOM_NUMERIK = [
    "jam_pickup",
    "jarak_km",
    "jumlah_transit",
    "janji_hari",
    "berat_kg",
    "volume_hub_hari",
]
VOLUME_MAKS = 18000
BERAT_MEDIAN = 1.34
VOLUME_MEDIAN = 5358.0
JARAK_MEDIAN = 356.0

FITUR_REQUEST = [
    "tanggal_pickup",
    "jam_pickup",
    "kota_tujuan",
    "jarak_km",
    "jumlah_transit",
    "layanan",
    "janji_hari",
    "berat_kg",
    "metode_bayar",
    "cuaca_saat_pickup",
    "volume_hub_hari",
]

DEFAULT_META = {
    "versi_model": "1.0.0",
    "nama_model": "Logistic Regression + fitur",
    "dibuat": "2026-11-15",
    "fitur_request": FITUR_REQUEST,
    "target": "terlambat",
    "threshold": 0.11,
    "batas_prioritas": 0.5,
    "pakai_volume": True,
    "metrik_validasi": {"pr_auc": 0.662},
    "metrik_test": {"pr_auc": 0.378, "recall": 0.772, "precision": 0.238},
    "asumsi_biaya": {"FN": 50000, "FP": 5000},
    "versi_library": {"python": "3.13.18", "sklearn": "1.6.1", "pandas": "2.2.3"},
}


def _muat_meta() -> dict:
    meta_path = os.path.join(HERE, "artifacts", "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "threshold" in data and "fitur_request" in data:
                return {**DEFAULT_META, **data}
        except Exception:
            pass
    return DEFAULT_META.copy()


META = _muat_meta()
BATAS_PRIORITAS = float(META.get("batas_prioritas", 0.50))
THRESHOLD_DEFAULT = float(META.get("threshold", 0.11))


def _kosong(val) -> bool:
    return val is None or (isinstance(val, str) and val.strip() == "")


def tambah_fitur(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan data mentah dan membuat 17 fitur sesuai 08_keterlambatan_paket_revisi_3.ipynb."""
    df = df.copy()

    # Angka yang kosong / berisi teks menjadi NaN
    for kolom in KOLOM_NUMERIK:
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce")

    # Normalisasi kota, layanan, dan metode bayar
    kota = df["kota_tujuan"].astype(str).str.strip().str.lower()
    df["kota_tujuan"] = kota.map(MAPPING_KOTA).fillna(kota.str.title())
    df["layanan"] = df["layanan"].astype(str).str.strip().str.upper().replace({"EXP": "EXPRESS"})
    df["metode_bayar"] = (
        df["metode_bayar"]
        .astype(str)
        .str.strip()
        .replace({"NON_COD": "non_COD", "non_cod": "non_COD", "non-cod": "non_COD", "non cod": "non_COD"})
    )

    # Volume di luar rentang latih dipotong ke batas atas; jika kosong diisi median latih
    df["volume_hub_hari"] = df["volume_hub_hari"].clip(upper=VOLUME_MAKS).fillna(VOLUME_MEDIAN)

    # Berat: >= 100 dianggap gram, <= 0 mustahil (diisi median latih)
    berat = df["berat_kg"].where(df["berat_kg"] < 100, df["berat_kg"] / 1000.0)
    df["berat_kg"] = berat.where(berat > 0).fillna(BERAT_MEDIAN)

    # Jarak kosong diisi median jarak kota tujuan (atau median global jika kota tidak dikenal)
    df["jarak_km"] = df["jarak_km"].fillna(df["kota_tujuan"].map(JARAK_KOTA)).fillna(JARAK_MEDIAN)

    # Cuaca kosong
    cuaca = df["cuaca_saat_pickup"].fillna("tidak_diketahui").astype(str).str.strip().str.lower()
    df["cuaca_saat_pickup"] = cuaca.replace({"": "tidak_diketahui", "none": "tidak_diketahui", "nan": "tidak_diketahui"})

    # Fitur tanggal dan fitur turunan
    tanggal = pd.to_datetime(df["tanggal_pickup"])
    df["senin"] = (tanggal.dt.dayofweek == 0).astype(int)
    df["after_cutoff"] = (df["jam_pickup"] >= JAM_CUTOFF).astype(int)
    df["slack"] = df["janji_hari"] - df["jumlah_transit"]
    df["hari_per_1000km"] = df["janji_hari"] / (df["jarak_km"] / 1000.0)
    df["surge"] = (df["volume_hub_hari"] > BATAS_LONJAKAN).astype(int)
    df["log_vol"] = np.log(df["volume_hub_hari"].where(df["volume_hub_hari"] > 0))
    df["hujan_lebat"] = (df["cuaca_saat_pickup"] == "hujan_lebat").astype(int)

    # Menghapus kolom yang tidak dipakai model
    return df.drop(columns=["tanggal_pickup", "id_resi"], errors="ignore")


def periksa_request(request: dict):
    """Mengembalikan (masalah, peringatan) sesuai 08_keterlambatan_paket_revisi_3.ipynb."""
    masalah, peringatan = [], []
    jam = pd.to_numeric(request.get("jam_pickup"), errors="coerce")
    if pd.isna(jam) or not 7 <= jam <= 22:
        masalah.append("jam_pickup harus angka antara 7 dan 22")
    if _kosong(request.get("tanggal_pickup")) or pd.isna(pd.to_datetime(request.get("tanggal_pickup"), errors="coerce")):
        masalah.append("tanggal_pickup tidak valid (format YYYY-MM-DD)")
    if _kosong(request.get("layanan")) or str(request["layanan"]).strip().upper() not in ("EXPRESS", "EXP", "REG", "HEMAT"):
        masalah.append("layanan harus EXPRESS, REG, atau HEMAT")
    for kolom, bawah, atas in [("janji_hari", 1, 6), ("jumlah_transit", 1, 6)]:
        nilai = pd.to_numeric(request.get(kolom), errors="coerce")
        if pd.isna(nilai) or not bawah <= nilai <= atas:
            masalah.append(f"{kolom} harus angka {bawah}-{atas}")
    jarak_raw = request.get("jarak_km")
    jarak = pd.to_numeric(jarak_raw, errors="coerce")
    if not _kosong(jarak_raw) and (pd.isna(jarak) or not 0 < jarak <= 2500):
        masalah.append("jarak_km harus antara 0 dan 2500")
    berat_raw = request.get("berat_kg")
    berat = pd.to_numeric(berat_raw, errors="coerce")
    if not _kosong(berat_raw) and pd.isna(berat):
        masalah.append("berat_kg harus berupa angka")
    else:
        if pd.notna(berat) and berat >= 100:
            peringatan.append("berat >= 100 dianggap gram dan dikonversi ke kg")
        if pd.isna(berat) or berat <= 0:
            peringatan.append("berat kosong/tidak valid, diisi median")
    kota = str(request.get("kota_tujuan", "") or "").strip().lower()
    if not kota:
        masalah.append("kota_tujuan wajib diisi")
    elif kota not in MAPPING_KOTA:
        peringatan.append("kota tidak ada di data latih, prediksi kurang andal")
    vol_raw = request.get("volume_hub_hari")
    volume = pd.to_numeric(vol_raw, errors="coerce")
    if not _kosong(vol_raw) and pd.isna(volume):
        masalah.append("volume_hub_hari harus berupa angka")
    elif pd.isna(volume):
        peringatan.append("volume hub kosong, diisi median")
    elif not 2800 <= volume <= 18000:
        peringatan.append("volume di luar rentang data latih (2.861-17.979), prediksi kurang andal")
    return masalah, peringatan


class _BuiltinRevisi3Model:
    """Pipeline Logistic Regression + fitur terkalibrasi dari 08_keterlambatan_paket_revisi_3.ipynb."""

    NUM_COLS = [
        "jam_pickup",
        "jarak_km",
        "jumlah_transit",
        "janji_hari",
        "berat_kg",
        "volume_hub_hari",
        "after_cutoff",
        "senin",
        "slack",
        "hari_per_1000km",
        "surge",
        "log_vol",
        "hujan_lebat",
    ]
    MEANS = np.array([
        13.56053043, 535.27474741, 2.0884051, 2.8966803, 2.04033227,
        5816.05830527, 0.19957301, 0.14147823, 0.8082752, 10.48215775,
        0.07523454, 8.63406335, 0.09787707
    ], dtype=np.float64)
    SCALES = np.array([
        3.39290317, 517.23620347, 1.13495865, 1.23461172, 2.3217107,
        1804.829034, 0.3996794, 0.34851419, 1.13312209, 8.07478753,
        0.26376942, 0.24271937, 0.29714837
    ], dtype=np.float64)
    NUM_COEFS = np.array([
        -0.00732198, 0.35024535, 0.21474174, -0.12632461, -0.01053326,
        0.84659777, 0.35663225, 0.00647437, -0.35272884, -0.01532734,
        0.28741622, -0.38286105, 0.22510369
    ], dtype=np.float64)
    CAT_COLS = ["kota_tujuan", "layanan", "metode_bayar", "cuaca_saat_pickup"]
    CATS = [
        [
            "Balikpapan", "Bekasi", "Bogor", "Cirebon", "Denpasar",
            "Jakarta", "Makassar", "Malang", "Medan", "Palembang",
            "Pontianak", "Semarang", "Surabaya", "Tasikmalaya", "Yogyakarta",
        ],
        ["EXPRESS", "HEMAT", "REG"],
        ["COD", "non_COD"],
        ["berawan", "cerah", "hujan", "hujan_lebat", "tidak_diketahui"],
    ]
    CAT_COEFS = np.array([
        0.14153591, -0.21648149, -0.04787349, -0.10241738, 0.35223023,
        -0.29750839, 0.07733561, -0.26656851, -0.13078851, 0.07738528,
        0.08881887, -0.01235974, -0.32289629, -0.15077104, 0.01955885,
        -0.15624963, -0.14915184, -0.42732808, -0.21235939, -0.52037015,
        -0.32830583, -0.36663725, -0.04789249, -0.00482823, 0.01493426
    ], dtype=np.float64)
    INTERCEPT = -0.82813669

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        probs = []
        for _, row in df.iterrows():
            x_num = np.array([float(row[c]) for c in self.NUM_COLS], dtype=np.float64)
            z_num = (x_num - self.MEANS) / self.SCALES
            logit = self.INTERCEPT + float(np.dot(z_num, self.NUM_COEFS))
            idx = 0
            for c_col, cat_list in zip(self.CAT_COLS, self.CATS):
                val = str(row[c_col])
                for cat_val in cat_list:
                    if val == cat_val:
                        logit += float(self.CAT_COEFS[idx])
                    idx += 1
            p1 = float(1.0 / (1.0 + np.exp(-logit)))
            probs.append([1.0 - p1, p1])
        return np.array(probs, dtype=np.float64)


BUILTIN_MODEL = _BuiltinRevisi3Model()


def _muat_model():
    model_path = os.path.join(HERE, "artifacts", "model.joblib")
    if os.path.exists(model_path):
        try:
            import joblib
            m = joblib.load(model_path)
            return m
        except Exception:
            pass
    return BUILTIN_MODEL


MODEL = _muat_model()


def rekomendasi(proba: float, threshold: float = None) -> str:
    """Rekomendasi tiga tingkat sesuai 08_keterlambatan_paket_revisi_3.ipynb."""
    threshold = THRESHOLD_DEFAULT if threshold is None else float(threshold)
    if proba >= BATAS_PRIORITAS:
        return "PRIORITASKAN + KABARI PELANGGAN"
    if proba >= threshold:
        return "KABARI PELANGGAN"
    return "AMAN"


def prediksi(request: dict) -> dict:
    """Memvalidasi masukan, menghitung fitur via tambah_fitur, dan menjalankan prediksi model."""
    masalah, peringatan = periksa_request(request)
    if masalah:
        return {
            "status": "DITOLAK",
            "masalah": masalah,
            "peringatan": peringatan,
            "ok": False,
            "error": masalah,
            "warning": peringatan,
        }

    req_clean = {k: request.get(k) for k in META["fitur_request"]}
    for k in ("jarak_km", "berat_kg", "volume_hub_hari", "cuaca_saat_pickup"):
        if isinstance(req_clean.get(k), str) and req_clean[k].strip() == "":
            req_clean[k] = None

    data_input = tambah_fitur(pd.DataFrame([req_clean])[META["fitur_request"]])

    mulai = time.perf_counter()
    try:
        proba = float(MODEL.predict_proba(data_input)[0, 1])
    except Exception:
        proba = float(BUILTIN_MODEL.predict_proba(data_input)[0, 1])
    latency = (time.perf_counter() - mulai) * 1000.0

    rek = rekomendasi(proba, META["threshold"])
    vol_num = pd.to_numeric(request.get("volume_hub_hari"), errors="coerce")
    lonjakan = bool(pd.notna(vol_num) and float(vol_num) > BATAS_LONJAKAN)
    andal = not any("kurang andal" in w for w in peringatan)

    return {
        "status": "OK",
        "probabilitas_terlambat": round(proba, 4),
        "rekomendasi": rek,
        "peringatan": peringatan,
        "threshold": META["threshold"],
        "batas_prioritas": BATAS_PRIORITAS,
        "versi_model": META["versi_model"],
        "nama_model": META.get("nama_model", "Logistic Regression + fitur"),
        "latency_ms": round(latency, 1),
        "ok": True,
        "probabilitas": round(proba, 4),
        "warning": peringatan,
        "error": [],
        "andal": andal,
        "lonjakan": lonjakan,
    }
