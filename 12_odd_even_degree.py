"""
12_odd_even_degree.py
Fenomena orde ganjil-genap (Fan & Gijbels 1996) -- justifikasi formal kenapa
p=1 (Local Linear) adalah pembanding kanonik terhadap NW (p=0) untuk studi
boundary effect, bukan pilihan sembarang dari keluarga LPR derajat-p.

TEMUAN: derajat GANJIL menghapus suku bias order-berikutnya SECARA KONSTRUKSI
(lewat reproduksi polinom Z), sehingga orde batas = orde interior. Derajat
GENAP hanya diuntungkan SIMETRI jendela di interior -- keuntungan yang hilang
di batas, sehingga orde batas turun satu tingkat dari interior.

Konsekuensi: p=1 adalah derajat ganjil dengan VARIANS TERKECIL yang sudah
menyembuhkan O(h) NW di batas -- menaikkan p lebih lanjut (p=3) tidak
menaikkan orde batas lagi (tetap sama seperti p=2), hanya menambah varians.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from lpr_core import exact_bias
from sim_functions import M2

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"; OUT_DIR.mkdir(exist_ok=True)

n = 198
X = np.arange(1, n + 1) / n
KERNEL = "epanechnikov"
tgt = M2
hs = np.array([0.20, 0.15, 0.10, 0.075, 0.05])

rows = []
for p in [0, 1, 2, 3]:
    for loc, x0 in [("interior", 0.5), ("batas", 1.0)]:
        b = np.array([exact_bias(np.array([x0]), X, h, p, tgt, KERNEL)[0] for h in hs])
        orde = np.polyfit(np.log(hs), np.log(np.abs(b)), 1)[0]
        rows.append({"p": p, "ganjil/genap": "ganjil" if p % 2 else "genap",
                     "lokasi": loc, "orde_empirik": orde})

res = pd.DataFrame(rows)
piv = res.pivot(index=["p", "ganjil/genap"], columns="lokasi", values="orde_empirik")
piv["turun_di_batas"] = piv["interior"] - piv["batas"]

print("=" * 74)
print("ORDE BIAS EMPIRIK: interior vs batas, per derajat p  (M2_exp, Epanechnikov)")
print("=" * 74)
print(piv.round(2).to_string())
print()
print("Interpretasi: 'turun_di_batas' ~ 1 untuk p GENAP (kehilangan subsidi simetri),")
print("~ 0 untuk p GANJIL (koreksi via konstruksi, bukan simetri).")

res.to_csv(OUT_DIR / "odd_even_degree.csv", index=False)
print(f"\n[tersimpan] {OUT_DIR/'odd_even_degree.csv'}")
