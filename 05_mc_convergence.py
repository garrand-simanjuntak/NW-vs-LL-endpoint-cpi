"""
05_mc_convergence.py
Validasi Monte Carlo terhadap formula bias eksak.

FAKTA ANALITIK yang mendasari uji ini:
    bias_MC(x) - bias_eksak(x) = w(x)' * eps_bar,     eps_bar = (1/R) sum_r eps_r
Suku m(.) LENYAP -- deviasi tidak bergantung pada fungsi target sama sekali,
hanya pada bobot w(x) dan realisasi galat. Karenanya:
  - cukup diuji sekali (tidak perlu diulang per fungsi target)
  - deviasi HARUS meluruh dengan laju tepat O(R^{-1/2})
  - jika formula bias eksak salah, deviasi menuju KONSTANTA tak-nol

Statistik uji: RMS deviasi sepanjang x, dirata-ratakan atas S seed independen
(bukan statistik max, yang extreme-value dan sangat bising).
"""
import numpy as np
import pandas as pd
from lpr_core import exact_bias, weight_matrix
from sim_functions import M1

from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)


n = 198
X = np.arange(1, n + 1) / n
KERNEL, SIGMA, h = "epanechnikov", 0.30, 0.10
R_GRID = [125, 250, 500, 1000, 2000, 4000]
S = 30

L = {p: weight_matrix(X, X, h, p, KERNEL) for p in (0, 1)}

print("=" * 84)
print("VALIDASI MC -- apakah |bias_MC - bias_eksak| meluruh dengan laju O(R^-1/2)?")
print(f"Statistik: RMS deviasi sepanjang x, dirata-ratakan atas S={S} seed independen")
print("=" * 84)

rows = []
for R in R_GRID:
    dev = {0: [], 1: []}
    for s in range(S):
        rng = np.random.default_rng(1000 + s * 97 + R)
        eps_bar = rng.normal(0, SIGMA, (R, n)).mean(axis=0)
        for p in (0, 1):
            d = L[p] @ eps_bar
            dev[p].append(np.sqrt(np.mean(d ** 2)))
    rows.append({
        "R": R,
        "RMSdev_NW": np.mean(dev[0]),
        "RMSdev_LL": np.mean(dev[1]),
        "RMSdev_NW*sqrt(R)": np.mean(dev[0]) * np.sqrt(R),
        "RMSdev_LL*sqrt(R)": np.mean(dev[1]) * np.sqrt(R),
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

o_nw = np.polyfit(np.log(df.R), np.log(df.RMSdev_NW), 1)[0]
o_ll = np.polyfit(np.log(df.R), np.log(df.RMSdev_LL), 1)[0]
print(f"\nOrde konvergensi empirik  NW: {o_nw:+.4f}   LL: {o_ll:+.4f}   (teori: -0.5000)")

stab_nw = df["RMSdev_NW*sqrt(R)"].std() / df["RMSdev_NW*sqrt(R)"].mean()
stab_ll = df["RMSdev_LL*sqrt(R)"].std() / df["RMSdev_LL*sqrt(R)"].mean()
print(f"Kolom dev*sqrt(R) stabil? koef. variasi NW={stab_nw:.4f}, LL={stab_ll:.4f} (harus ~0)")

ok = abs(o_nw + 0.5) < 0.02 and abs(o_ll + 0.5) < 0.02
print(f"\n=> {'FORMULA BIAS EKSAK TERVALIDASI' if ok else 'PERIKSA ULANG'}")
