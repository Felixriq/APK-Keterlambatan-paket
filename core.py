"""Logika inti aplikasi sesuai 08_keterlambatan_paket_revisi_3.ipynb:
validasi masukan (periksa_request), rekayasa fitur (tambah_fitur), prediksi model, dan rekomendasi.
"""
import json
import os
import pickle
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.features import (  # noqa: E402
    BATAS_LONJAKAN,
    JARAK_KOTA,
    MAPPING_KOTA,
    periksa_request,
    tambah_fitur,
)

META_PATH = os.path.join(HERE, "artifacts", "metadata.json")
MODEL_PATH = os.path.join(HERE, "artifacts", "model.joblib")

with open(META_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)

BATAS_PRIORITAS = float(META.get("batas_prioritas", 0.50))
THRESHOLD_DEFAULT = float(META.get("threshold", 0.11))


class _FallbackSklearnNode:
    def __init__(self, *args, **kwargs):
        self.state = {}

    def __setstate__(self, state):
        self.state = state


class _FallbackUnpickler(pickle._Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy"):
            module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)
        if module.startswith("sklearn"):
            return type(f"{module}.{name}", (_FallbackSklearnNode,), {})
        return super().find_class(module, name)


class _FallbackPipelineModel:
    """Predictor ringan yang membaca bobot asli dari artifacts/model.joblib jika scikit-learn berbeda versi."""

    def __init__(self, path):
        with open(path, "rb") as fh:
            raw_pipe = _FallbackUnpickler(fh).load()
        ct_state = raw_pipe.state["steps"][0][1].state
        lr_state = raw_pipe.state["steps"][1][1].state
        ss_state = ct_state["transformers_"][0][1].state
        ohe_state = ct_state["transformers_"][1][1].state

        self.num_cols = list(ss_state["feature_names_in_"])
        self.means = np.asarray(ss_state["mean_"], dtype=np.float64)
        self.scales = np.asarray(ss_state["scale_"], dtype=np.float64)
        self.cat_cols = list(ohe_state["feature_names_in_"])
        self.categories = [list(c) for c in ohe_state["categories_"]]
        self.coefs = np.asarray(lr_state["coef_"][0], dtype=np.float64)
        self.intercept = float(lr_state["intercept_"][0])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        probs = []
        n_num = len(self.num_cols)
        num_coefs = self.coefs[:n_num]
        cat_coefs = self.coefs[n_num:]
        for _, row in df.iterrows():
            x_num = np.array([float(row[c]) for c in self.num_cols], dtype=np.float64)
            z_num = (x_num - self.means) / self.scales
            logit = self.intercept + float(np.dot(z_num, num_coefs))
            idx = 0
            for c_col, cat_list in zip(self.cat_cols, self.categories):
                val = str(row[c_col])
                for cat_val in cat_list:
                    if val == cat_val:
                        logit += float(cat_coefs[idx])
                    idx += 1
            p1 = float(1.0 / (1.0 + np.exp(-logit)))
            probs.append([1.0 - p1, p1])
        return np.array(probs, dtype=np.float64)


def _muat_model(path):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        return _FallbackPipelineModel(path)


MODEL = _muat_model(MODEL_PATH)
FALLBACK_MODEL = _FallbackPipelineModel(MODEL_PATH)


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

    # Membuat DataFrame sesuai urutan kolom request di metadata.json, lalu hitung fitur
    req_clean = {k: request.get(k) for k in META["fitur_request"]}
    for k in ("jarak_km", "berat_kg", "volume_hub_hari", "cuaca_saat_pickup"):
        if isinstance(req_clean.get(k), str) and req_clean[k].strip() == "":
            req_clean[k] = None

    data_input = tambah_fitur(pd.DataFrame([req_clean])[META["fitur_request"]])

    mulai = time.perf_counter()
    try:
        proba = float(MODEL.predict_proba(data_input)[0, 1])
    except Exception:
        proba = float(FALLBACK_MODEL.predict_proba(data_input)[0, 1])
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
