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
BATAS_LONJAKAN = 9000
VOLUME_MAKS = 18000

# Kolom numerik yang digunakan
KOLOM_NUMERIK = [
    "jam_pickup",
    "jarak_km",
    "jumlah_transit",
    "janji_hari",
    "berat_kg",
    "volume_hub_hari"
]


def tambah_fitur(df):
    df = df.copy()

    # Mengubah kolom numerik menjadi angka
    for kolom in KOLOM_NUMERIK:
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce")

    # Normalisasi kota
    kota = (
        df["kota_tujuan"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    df["kota_tujuan"] = kota.map(MAPPING_KOTA).fillna(
        kota.str.title()
    )

    # Normalisasi layanan
    df["layanan"] = (
        df["layanan"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace({"EXP": "EXPRESS"})
    )

    # Normalisasi metode pembayaran
    df["metode_bayar"] = (
        df["metode_bayar"]
        .astype("string")
        .str.strip()
        .replace({
            "NON_COD": "non_COD",
            "non_cod": "non_COD"
        })
    )

    # Membatasi volume sesuai rentang data latih
    df["volume_hub_hari"] = df[
        "volume_hub_hari"
    ].clip(upper=VOLUME_MAKS)

    # Jarak nol atau negatif dianggap tidak valid
    df["jarak_km"] = df["jarak_km"].where(
        df["jarak_km"] > 0
    )

    # Berat >= 100 dianggap gram, kemudian dikonversi ke kg
    berat = df["berat_kg"].where(
        df["berat_kg"] < 100,
        df["berat_kg"] / 1000
    )
    df["berat_kg"] = berat.where(berat > 0)

    # Menangani cuaca kosong
    df["cuaca_saat_pickup"] = (
        df["cuaca_saat_pickup"]
        .fillna("tidak_diketahui")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Ekstraksi fitur tanggal
    tanggal = pd.to_datetime(
        df["tanggal_pickup"],
        errors="coerce"
    )

    df["senin"] = (
        tanggal.dt.dayofweek == 0
    ).astype(int)

    # Ekstraksi fitur waktu pickup
    df["after_cutoff"] = (
        df["jam_pickup"] >= JAM_CUTOFF
    ).astype(int)

    # Selisih waktu pengiriman dan jumlah transit
    df["slack"] = (
        df["janji_hari"] - df["jumlah_transit"]
    )

    # Efisiensi waktu pengiriman terhadap jarak
    df["hari_per_1000km"] = (
        df["janji_hari"] / (df["jarak_km"] / 1000)
    )

    # Menghapus nilai tak hingga
    df["hari_per_1000km"] = df[
        "hari_per_1000km"
    ].replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    # Indikator lonjakan volume
    df["surge"] = (
        df["volume_hub_hari"] > BATAS_LONJAKAN
    ).astype(int)

    # Transformasi logaritmik volume
    df["log_vol"] = np.log(
        df["volume_hub_hari"].where(
            df["volume_hub_hari"] > 0
        )
    )

    # Indikator hujan lebat
    df["hujan_lebat"] = (
        df["cuaca_saat_pickup"] == "hujan_lebat"
    ).astype(int)

    # Menghapus kolom yang tidak digunakan model
    return df.drop(
        columns=["tanggal_pickup", "id_resi"],
        errors="ignore"
    )


def periksa_request(request):
    """Mengembalikan (masalah, peringatan)."""

    masalah = []
    peringatan = []

    # Validasi jam pickup
    jam = pd.to_numeric(
        request.get("jam_pickup"),
        errors="coerce"
    )

    if pd.isna(jam) or not 7 <= jam <= 22:
        masalah.append(
            "jam_pickup harus angka antara 7 dan 22"
        )

    # Validasi tanggal
    tanggal = pd.to_datetime(
        request.get("tanggal_pickup"),
        errors="coerce"
    )

    if pd.isna(tanggal):
        masalah.append(
            "tanggal_pickup tidak valid (format YYYY-MM-DD)"
        )

    # Validasi layanan
    layanan = request.get("layanan")

    if layanan is None or str(layanan).strip().upper() not in (
        "EXPRESS", "EXP", "REG", "HEMAT"
    ):
        masalah.append(
            "layanan harus EXPRESS, REG, atau HEMAT"
        )

    # Validasi janji pengiriman
    janji = pd.to_numeric(
        request.get("janji_hari"),
        errors="coerce"
    )

    if pd.isna(janji) or not 1 <= janji <= 6:
        masalah.append(
            "janji_hari harus angka 1-6"
        )

    # Validasi jarak
    jarak = pd.to_numeric(
        request.get("jarak_km"),
        errors="coerce"
    )

    if pd.isna(jarak):
        peringatan.append(
            "jarak kosong/tidak valid, akan diisi median data latih"
        )
    elif not 0 < jarak <= 2500:
        masalah.append(
            "jarak_km harus antara 0 dan 2500"
        )

    # Validasi berat
    berat = pd.to_numeric(
        request.get("berat_kg"),
        errors="coerce"
    )

    if pd.notna(berat) and berat >= 100:
        peringatan.append(
            "berat >= 100 dianggap gram dan dikonversi ke kg"
        )

    if pd.isna(berat) or berat <= 0:
        peringatan.append(
            "berat kosong/tidak valid, akan diisi median data latih"
        )

    # Validasi kota
    kota = str(
        request.get("kota_tujuan", "")
    ).strip().lower()

    if not kota or kota not in MAPPING_KOTA:
        peringatan.append(
            "kota tidak ada dalam mapping, pastikan kota dikenali"
        )

    # Validasi jumlah transit
    transit = pd.to_numeric(
        request.get("jumlah_transit"),
        errors="coerce"
    )

    if pd.isna(transit):
        peringatan.append(
            "jumlah_transit kosong, fitur turunan slack akan diimputasi"
        )
    elif transit < 0:
        masalah.append(
            "jumlah_transit tidak boleh negatif"
        )

    # Validasi volume hub
    volume = pd.to_numeric(
        request.get("volume_hub_hari"),
        errors="coerce"
    )

    if pd.isna(volume):
        peringatan.append(
            "volume hub kosong, fitur numerik akan diimputasi"
        )
    elif not 2800 <= volume <= 18000:
        peringatan.append(
            "volume di luar rentang data latih (2.861-17.979), "
            "prediksi kurang andal"
        )

    return masalah, peringatan
