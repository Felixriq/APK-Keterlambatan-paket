import numpy as np
import pandas as pd

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
    "tasikmalaya": "Tasikmalaya"
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
    "Yogyakarta": 390.0
}
JAM_CUTOFF = 17
BATAS_LONJAKAN = 9000
KOLOM_NUMERIK = ["jam_pickup", "jarak_km", "jumlah_transit", "janji_hari", "berat_kg", "volume_hub_hari"]
VOLUME_MAKS = 18000   # batas atas data latih; nilai lebih besar dipotong agar model tidak mengekstrapolasi
BERAT_MEDIAN = 1.34
VOLUME_MEDIAN = 5358.0
JARAK_MEDIAN = 356.0


def _kosong(val):
    return val is None or (isinstance(val, str) and val.strip() == "")


def tambah_fitur(df):
    df = df.copy()

    # Angka yang kosong / berisi teks menjadi NaN
    for kolom in KOLOM_NUMERIK:
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce")

    # Normalisasi kota dan layanan
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
    berat = df["berat_kg"].where(df["berat_kg"] < 100, df["berat_kg"] / 1000)
    df["berat_kg"] = berat.where(berat > 0).fillna(BERAT_MEDIAN)

    # Jarak kosong diisi median jarak kota tujuan (atau median global jika kota asing)
    df["jarak_km"] = df["jarak_km"].fillna(df["kota_tujuan"].map(JARAK_KOTA)).fillna(JARAK_MEDIAN)

    # Cuaca kosong
    cuaca = df["cuaca_saat_pickup"].fillna("tidak_diketahui").astype(str).str.strip().str.lower()
    df["cuaca_saat_pickup"] = cuaca.replace({"": "tidak_diketahui", "none": "tidak_diketahui", "nan": "tidak_diketahui"})

    # Fitur tanggal dan fitur turunan
    tanggal = pd.to_datetime(df["tanggal_pickup"])
    df["senin"] = (tanggal.dt.dayofweek == 0).astype(int)
    df["after_cutoff"] = (df["jam_pickup"] >= JAM_CUTOFF).astype(int)
    df["slack"] = df["janji_hari"] - df["jumlah_transit"]
    df["hari_per_1000km"] = df["janji_hari"] / (df["jarak_km"] / 1000)
    df["surge"] = (df["volume_hub_hari"] > BATAS_LONJAKAN).astype(int)
    df["log_vol"] = np.log(df["volume_hub_hari"].where(df["volume_hub_hari"] > 0))
    df["hujan_lebat"] = (df["cuaca_saat_pickup"] == "hujan_lebat").astype(int)

    # Menghapus kolom yang tidak dipakai model
    return df.drop(columns=["tanggal_pickup", "id_resi"], errors="ignore")


def periksa_request(request):
    """Mengembalikan (masalah, peringatan). Jika ada masalah, jangan diprediksi."""
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
