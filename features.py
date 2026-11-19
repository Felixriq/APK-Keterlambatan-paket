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
    "bali": "Bali",
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
JAM_CUTOFF = 17

# Kolom numerik yang digunakan
KOLOM_NUMERIK = [
    "jam_pickup",
    "jarak_km",
    "janji_hari",
    "berat_kg"]


def tambah_fitur(df):
    df = df.copy()

    # Mengubah kolom numerik menjadi angka
    for kolom in KOLOM_NUMERIK:
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce")

    # Normalisasi kota dan layanan
    kota = df["kota_tujuan"].astype("string").str.strip().str.lower()
    df["kota_tujuan"] = kota.map(MAPPING_KOTA).fillna(kota.str.title())

    df["layanan"] = (
        df["layanan"].astype("string").str.strip().str.upper()
        .replace({"EXP": "EXPRESS"}))

    df["metode_bayar"] = (
        df["metode_bayar"].astype("string").str.strip()
        .replace({"NON_COD": "non_COD", "non_cod": "non_COD"}))

    # Jarak nol atau negatif dianggap tidak valid
    df["jarak_km"] = df["jarak_km"].where(df["jarak_km"] > 0)

    # Berat: >= 100 dianggap gram, <= 0 dianggap tidak valid
    berat = df["berat_kg"].where(
        df["berat_kg"] < 100,
        df["berat_kg"] / 1000)
    df["berat_kg"] = berat.where(berat > 0)

    # Cuaca kosong
    df["cuaca_saat_pickup"] = (
        df["cuaca_saat_pickup"]
        .fillna("tidak_diketahui")
        .astype(str)
        .str.strip()
        .str.lower())

    # Fitur tanggal
    tanggal = pd.to_datetime(
        df["tanggal_pickup"], errors="coerce")

    df["senin"] = (
        tanggal.dt.dayofweek == 0
    ).astype(int)

    # Fitur waktu pickup
    df["after_cutoff"] = (
        df["jam_pickup"] >= JAM_CUTOFF
    ).astype(int)

    # Fitur efisiensi waktu pengiriman terhadap jarak
    df["hari_per_1000km"] = (
        df["janji_hari"] / (df["jarak_km"] / 1000))

    # Menghapus nilai tak hingga jika ada
    df["hari_per_1000km"] = df[
        "hari_per_1000km"
    ].replace([float("inf"), float("-inf")], float("nan"))

    # Menghapus kolom yang tidak dipakai model
    return df.drop(
        columns=["tanggal_pickup", "id_resi", "volume_hub_hari", "jumlah_transit"],
        errors="ignore")


def periksa_request(request):
    """Mengembalikan (masalah, peringatan)."""

    masalah, peringatan = [], []

    # Validasi jam pickup
    jam = pd.to_numeric(
        request.get("jam_pickup"), errors="coerce")

    if pd.isna(jam) or not 7 <= jam <= 22:
        masalah.append(
            "jam_pickup harus angka antara 7 dan 22")

    # Validasi tanggal
    if pd.isna(pd.to_datetime(
        request.get("tanggal_pickup"), errors="coerce"
    )):
        masalah.append(
            "tanggal_pickup tidak valid (format YYYY-MM-DD)")

    # Validasi layanan
    if request.get("layanan") is None or str(
        request["layanan"]
    ).strip().upper() not in ("EXPRESS", "EXP", "REG", "HEMAT"):
        masalah.append(
            "layanan harus EXPRESS, REG, atau HEMAT")

    # Validasi janji pengiriman
    janji = pd.to_numeric(
        request.get("janji_hari"), errors="coerce")

    if pd.isna(janji) or not 1 <= janji <= 6:
        masalah.append(
            "janji_hari harus angka 1-6")

    # Validasi jarak
    jarak = pd.to_numeric(
        request.get("jarak_km"), errors="coerce")

    if pd.isna(jarak):
        peringatan.append(
            "jarak kosong/tidak valid, akan diisi median data latih")
    elif not 0 < jarak <= 2500:
        masalah.append(
            "jarak_km harus antara 0 dan 2500")

    # Validasi berat
    berat = pd.to_numeric(
        request.get("berat_kg"), errors="coerce")

    if pd.notna(berat) and berat >= 100:
        peringatan.append(
            "berat >= 100 dianggap gram dan dikonversi ke kg")

    if pd.isna(berat) or berat <= 0:
        peringatan.append(
            "berat kosong/tidak valid, akan diisi median data latih")

    # Peringatan kota
    kota = str(
        request.get("kota_tujuan", "")
    ).strip().lower()

    if not kota or kota not in MAPPING_KOTA:
        peringatan.append(
            "kota tidak ada dalam mapping, pastikan kota dikenali")

    return masalah, peringatan
