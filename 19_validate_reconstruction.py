"""
19_validate_reconstruction.py
Validasi rekonstruksi IHK terhadap indeks resmi BPS yang di-chain-link
(Reviewer 1, komentar Data: "compare the reconstructed index with the
chain-linked official index and report the discrepancy").

MASALAH
Eq. (21) mengumpulkan laju inflasi bulanan yang dipublikasikan dengan dua
angka desimal. Pembulatan 198 kali berpotensi menumpuk menjadi drift
terhadap level indeks resmi. Besarnya harus diukur, bukan diasumsikan kecil.

TAHUN DASAR
BPS mengganti tahun dasar empat kali, sehingga level indeks resmi patah di
setiap pergantian. Perbandingan karena itu dilakukan DI DALAM tiap periode
tahun dasar: indeks resmi dan indeks rekonstruksi sama-sama dinormalkan ke
100 pada bulan pertama periode tersebut, lalu dibandingkan. Dengan begitu
tidak ada asumsi tentang faktor penyambung antar periode.

INPUT YANG HARUS DISIAPKAN
  raw/ihk_official.csv  dengan kolom:
      date         YYYY-MM-01
      ihk_official level indeks resmi BPS pada bulan itu
      base_year    tahun dasar yang berlaku (2007, 2012, 2018, 2022)
  Hanya bulan yang tersedia yang dipakai; bulan yang kosong dilewati.

KELUARAN
  output/validate_reconstruction.csv   deviasi per bulan
  output/validate_reconstruction.txt   ringkasan per periode tahun dasar
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"; OUT_DIR.mkdir(exist_ok=True)
RAW = ROOT / "raw" / "ihk_official.csv"

rec = pd.read_csv(OUT_DIR / "ihk_bps_2010_2026.csv", parse_dates=["date"])
off = pd.read_csv(RAW, parse_dates=["date"])
off = off.dropna(subset=["ihk_official", "base_year"])
df = rec.merge(off[["date", "ihk_official", "base_year"]], on="date", how="inner")
df["base_year"] = df["base_year"].astype(int)
print(f"bulan beririsan: {len(df)} dari {len(rec)} bulan rekonstruksi")

rows = []
for by, g in df.groupby("base_year", sort=True):
    g = g.sort_values("date").copy()
    # normalkan keduanya ke bulan pertama periode ini
    g["off_idx"] = 100 * g.ihk_official / g.ihk_official.iloc[0]
    g["rec_idx"] = 100 * g.IHK_recon / g.IHK_recon.iloc[0]
    g["dev_pct"] = 100 * (g.rec_idx / g.off_idx - 1)
    g["dev_logpts"] = np.log(g.rec_idx) - np.log(g.off_idx)
    rows.append(g)

cmp = pd.concat(rows, ignore_index=True)
cmp[["date", "base_year", "ihk_official", "IHK_recon", "off_idx", "rec_idx",
     "dev_pct", "dev_logpts"]].to_csv(OUT_DIR / "validate_reconstruction.csv", index=False)

lines = []
lines.append("VALIDASI REKONSTRUKSI IHK TERHADAP INDEKS RESMI BPS")
lines.append("Deviasi dihitung di dalam tiap periode tahun dasar, keduanya "
             "dinormalkan ke 100 pada bulan pertama periode.\n")
hdr = f"{'base':>6} {'n':>4} {'mulai':>10} {'akhir':>10} {'max|dev|%':>10} {'dev akhir%':>11} {'RMS dev%':>9}"
lines.append(hdr); lines.append("-" * len(hdr))
for by, g in cmp.groupby("base_year", sort=True):
    lines.append(f"{by:>6} {len(g):>4} {g.date.min():%Y-%m} {g.date.max():%Y-%m} "
                 f"{g.dev_pct.abs().max():>10.4f} {g.dev_pct.iloc[-1]:>11.4f} "
                 f"{np.sqrt((g.dev_pct**2).mean()):>9.4f}")
lines.append("")
lines.append(f"Deviasi absolut terbesar di seluruh periode: {cmp.dev_pct.abs().max():.4f} persen")
lines.append(f"Deviasi absolut terbesar dalam log poin   : {cmp.dev_logpts.abs().max():.6f}")

# bandingkan dengan skala efek yang diukur naskah
rev = OUT_DIR / "revision_diagnostics.csv"
if rev.exists():
    r = pd.read_csv(rev)
    rms = r[(r.response == "log_IHK") & (r.h == 12)]
    if len(rms):
        d = float(rms.pred_discrete.iloc[0])
        lines.append("")
        lines.append("Skala pembanding: revisi NW pada h = 12 bulan berukuran "
                     f"delta_h = {d:.3f} kali kemiringan lokal, yaitu sekitar "
                     "0.013 log poin pada kemiringan tipikal log IHK "
                     "(lihat Tabel 5 dan Gambar 4).")
        lines.append("Deviasi rekonstruksi di atas harus dibandingkan terhadap angka itu.")

txt = "\n".join(lines)
(OUT_DIR / "validate_reconstruction.txt").write_text(txt, encoding="utf-8")
print(txt)
print(f"\n[tersimpan] {OUT_DIR / 'validate_reconstruction.csv'}")
print(f"[tersimpan] {OUT_DIR / 'validate_reconstruction.txt'}")
