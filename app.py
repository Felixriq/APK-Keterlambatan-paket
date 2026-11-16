"""Aplikasi Staf Hub Bandung (Flask) sesuai 08_keterlambatan_paket_revisi_3.ipynb.
Jalankan: python app.py -> http://127.0.0.1:5000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, render_template_string, request
from core import META, JARAK_KOTA, prediksi  # type: ignore

app = Flask(__name__)

PAGE = """<!doctype html><html lang="id"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kasus 08 &middot; Prediksi Keterlambatan Paket (Hub Bandung)</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;background:#f4f6f8;margin:0;color:#1d2733}
header{background:#12344d;color:#fff;padding:14px 28px;display:flex;justify-content:space-between;align-items:center}
header .sub{font-size:12px;color:#b8cde0;font-weight:400}
main{display:grid;grid-template-columns:1.25fr 1fr;gap:24px;padding:24px 28px;max-width:1140px;margin:0 auto}
@media(max-width:800px){main{grid-template-columns:1fr}}
.card{background:#fff;border-radius:10px;padding:18px 20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
form{display:grid;grid-template-columns:1fr 1fr;gap:12px 14px;align-content:start}
label{font-size:12px;color:#51606e;display:block;margin-bottom:3px;font-weight:500}
input,select{width:100%;padding:8px 10px;border:1px solid #c5ced6;border-radius:6px;box-sizing:border-box;font-size:14px}
button{grid-column:1/3;padding:10px;background:#1b6ca8;color:#fff;border:0;border-radius:6px;font-size:15px;font-weight:600;cursor:pointer;margin-top:4px}
button:hover{background:#155787}
.presets{grid-column:1/3;display:flex;gap:8px;flex-wrap:wrap;margin-bottom:4px}
.preset-btn{background:#e8f0fe;color:#1967d2;border:1px solid #c2d7fa;padding:5px 10px;border-radius:5px;font-size:12px;cursor:pointer;font-weight:500}
.preset-btn:hover{background:#d2e3fc}
.p{font-size:44px;font-weight:700;margin:6px 0}
.badge{display:inline-block;padding:8px 14px;border-radius:6px;color:#fff;font-weight:600;font-size:14px;margin-bottom:10px}
.AMAN{background:#2e8b57}.KABARI{background:#d98b00}.PRIORITAS{background:#c0392b}.DITOLAK{background:#6c757d}
.warn{background:#fff6e0;border-left:4px solid #d98b00;padding:8px 12px;margin:8px 0;font-size:13px;border-radius:0 4px 4px 0}
.err{background:#fdeaea;border-left:4px solid #c0392b;padding:8px 12px;margin:8px 0;font-size:13px;border-radius:0 4px 4px 0}
.meta-box{margin-top:14px;padding-top:12px;border-top:1px solid #e6ebf0;font-size:12px;color:#51606e;line-height:1.6}
h3{margin:0 0 10px;font-size:16px}
</style>
<header>
  <div>Kasus 08 &middot; Prediksi Keterlambatan Paket (Hub Bandung)</div>
  <div class="sub">Model: {{ meta.nama_model }} (v{{ meta.versi_model }}) &middot; Threshold: {{ meta.threshold }}</div>
</header>
<main>
<form id="f" class="card">
  <h3>Input Data Paket Saat Pickup</h3>
  <div class="presets">
    <span style="font-size:12px;color:#51606e;align-self:center">Contoh Uji Notebook:</span>
    <button type="button" class="preset-btn" onclick="isiPreset('risiko')">Paket Risiko Tinggi (Medan)</button>
    <button type="button" class="preset-btn" onclick="isiPreset('aman')">Paket Aman (Bogor)</button>
    <button type="button" class="preset-btn" onclick="isiPreset('gram')">Uji Berat Gram &amp; Kota Jkt</button>
  </div>
  <div><label>Tanggal pickup (YYYY-MM-DD)</label><input name="tanggal_pickup" type="date" value="2026-07-06" required></div>
  <div><label>Jam pickup (7-22)</label><input name="jam_pickup" type="number" min="7" max="22" step="1" value="19" required></div>
  <div>
    <label>Kota tujuan</label>
    <input name="kota_tujuan" list="daftar_kota" value="Medan" required>
    <datalist id="daftar_kota">
      <option value="Balikpapan"><option value="Bekasi"><option value="Bogor"><option value="Cirebon">
      <option value="Denpasar"><option value="Jakarta"><option value="Makassar"><option value="Malang">
      <option value="Medan"><option value="Palembang"><option value="Pontianak"><option value="Semarang">
      <option value="Surabaya"><option value="Tasikmalaya"><option value="Yogyakarta">
    </datalist>
  </div>
  <div><label>Jarak (km) &mdash; kosong = median kota</label><input name="jarak_km" type="number" step="any" value="1900" placeholder="Kosong = median kota"></div>
  <div><label>Layanan</label><select name="layanan"><option value="EXPRESS">EXPRESS</option><option value="REG">REG</option><option value="HEMAT">HEMAT</option></select></div>
  <div><label>Janji (hari, 1-6)</label><input name="janji_hari" type="number" min="1" max="6" step="1" value="3" required></div>
  <div><label>Jumlah transit (1-6)</label><input name="jumlah_transit" type="number" min="1" max="6" step="1" value="4" required></div>
  <div><label>Berat (kg) &mdash; &ge;100 dianggap gram</label><input name="berat_kg" type="number" step="any" value="2.0" placeholder="Kosong = median (1.34 kg)"></div>
  <div><label>Metode bayar</label><select name="metode_bayar"><option value="COD">COD</option><option value="non_COD">non_COD</option></select></div>
  <div><label>Cuaca saat pickup</label><select name="cuaca_saat_pickup"><option value="hujan_lebat">hujan_lebat</option><option value="cerah">cerah</option><option value="berawan">berawan</option><option value="hujan">hujan</option><option value="">tidak_diketahui</option></select></div>
  <div style="grid-column:1/3"><label>Volume hub hari ini (rentang latih: 2.861&ndash;17.979, lonjakan &gt; 9.000)</label><input name="volume_hub_hari" type="number" step="1" value="5400" placeholder="Kosong = median (5.358)"></div>
  <button type="submit">Prediksi Risiko Keterlambatan</button>
</form>
<div id="out" class="card">
  <h3>Hasil Prediksi</h3>
  <span style="color:#51606e;font-size:14px">Isi data paket di sebelah kiri lalu klik <b>Prediksi Risiko Keterlambatan</b>.</span>
  <div class="meta-box">
    <b>Aturan Keputusan (Biaya FN Rp50.000, FP Rp5.000):</b><br>
    &bull; <b>AMAN</b>: Probabilitas &lt; {{ meta.threshold }}<br>
    &bull; <b>KABARI PELANGGAN</b>: {{ meta.threshold }} &le; Probabilitas &lt; {{ meta.batas_prioritas }}<br>
    &bull; <b>PRIORITASKAN + KABARI PELANGGAN</b>: Probabilitas &ge; {{ meta.batas_prioritas }}
  </div>
</div>
</main>
<script>
const f=document.getElementById('f'),o=document.getElementById('out');
function isiPreset(tipe){
  if(tipe==='risiko'){
    f.tanggal_pickup.value='2026-07-06';f.jam_pickup.value='19';f.kota_tujuan.value='Medan';
    f.jarak_km.value='1900';f.jumlah_transit.value='4';f.layanan.value='EXPRESS';
    f.janji_hari.value='3';f.berat_kg.value='2.0';f.metode_bayar.value='COD';
    f.cuaca_saat_pickup.value='hujan_lebat';f.volume_hub_hari.value='5400';
  }else if(tipe==='aman'){
    f.tanggal_pickup.value='2026-07-06';f.jam_pickup.value='9';f.kota_tujuan.value='Bogor';
    f.jarak_km.value='125';f.jumlah_transit.value='1';f.layanan.value='REG';
    f.janji_hari.value='3';f.berat_kg.value='2.0';f.metode_bayar.value='non_COD';
    f.cuaca_saat_pickup.value='cerah';f.volume_hub_hari.value='5400';
  }else if(tipe==='gram'){
    f.tanggal_pickup.value='2026-07-06';f.jam_pickup.value='19';f.kota_tujuan.value='Jkt';
    f.jarak_km.value='';f.jumlah_transit.value='4';f.layanan.value='EXPRESS';
    f.janji_hari.value='3';f.berat_kg.value='1500';f.metode_bayar.value='COD';
    f.cuaca_saat_pickup.value='';f.volume_hub_hari.value='5400';
  }
}
f.onsubmit=async e=>{
  e.preventDefault();
  try{
    const payload=Object.fromEntries(new FormData(f));
    const resp=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const r=await resp.json();
    let h='<h3>Hasil Prediksi</h3>';
    const masalah=r.masalah||r.error||[];
    const peringatan=r.peringatan||r.warning||[];
    masalah.forEach(m=>h+='<div class="err"><b>Masalah:</b> '+m+'</div>');
    if(r.ok && r.status!=='DITOLAK'){
      const p=r.probabilitas_terlambat!==undefined?r.probabilitas_terlambat:r.probabilitas;
      const c=r.rekomendasi.startsWith('AMAN')?'AMAN':r.rekomendasi.startsWith('KABARI')?'KABARI':'PRIORITAS';
      h+='<div style="font-size:13px;color:#51606e">Probabilitas terlambat</div>';
      h+='<div class="p">'+(p*100).toFixed(2)+'% <span style="font-size:16px;color:#51606e;font-weight:400">('+p.toFixed(4)+')</span></div>';
      h+='<span class="badge '+c+'">'+r.rekomendasi+'</span>';
      if(r.lonjakan)h+='<div class="warn"><b>Hari lonjakan volume (&gt; 9.000):</b> risiko keterlambatan naik tajam (rata-rata 67,3%).</div>';
      if(!r.andal)h+='<div class="warn"><b>Prediksi kurang andal</b> karena sebagian masukan berada di luar cakupan data latih.</div>';
      peringatan.forEach(m=>h+='<div class="warn"><b>Peringatan:</b> '+m+'</div>');
      h+='<div class="meta-box">Threshold: <b>'+r.threshold+'</b> &middot; Batas Prioritas: <b>'+r.batas_prioritas+'</b> &middot; Versi Model: <b>'+r.versi_model+'</b> &middot; Latency: <b>'+r.latency_ms+' ms</b></div>';
    }else{
      h+='<span class="badge DITOLAK">STATUS: DITOLAK</span>';
      h+='<div style="margin-top:6px;color:#c0392b;font-size:14px"><b>Prediksi tidak dijalankan.</b> Perbaiki masukan yang bermasalah di atas.</div>';
    }
    o.innerHTML=h;
  }catch(err){
    o.innerHTML='<h3>Hasil Prediksi</h3><div class="err">Gagal menghubungi server prediksi.</div>';
  }
};
</script></html>"""


@app.get("/")
def index():
    return render_template_string(PAGE, meta=META, jarak_kota=JARAK_KOTA)


@app.post("/api/predict")
def api():
    return jsonify(prediksi(request.get_json(force=True, silent=True) or {}))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
