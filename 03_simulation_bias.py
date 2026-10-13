"""
03_simulation_bias.py
Studi simulasi: bias NW (p=0) vs Local Linear (p=1) pada desain tetap.

Tiga komponen:
  (A) BIAS EKSAK      -- deterministik, tanpa Monte Carlo
  (B) VALIDASI MC     -- membuktikan (A) benar secara empirik
  (C) VARIANS         -- i.i.d. dan AR(1)

Desain mereplikasi data IHK: x_t = t/n, n = 198, equidistant.
Tidak bergantung pada data riil -- bisa dijalankan berdiri sendiri.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from lpr_core import exact_bias, exact_variance, weight_matrix, ar1_corr
from sim_functions import TARGETS

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(20260711)

n = 198
X = np.arange(1, n + 1) / n
KERNEL = "epanechnikov"
SIGMA = 0.30
H_GRID = [0.05, 0.10, 0.15]
R = 1000

x_eval = X.copy()
h_ref = 0.10

print("=" * 74)
print("(A) BIAS EKSAK  -- deterministik, tanpa simulasi")
print("=" * 74)

rows = []
for tgt in TARGETS:
    for h in H_GRID:
        b_nw = exact_bias(x_eval, X, h, p=0, m_func=tgt, kernel=KERNEL)
        b_ll = exact_bias(x_eval, X, h, p=1, m_func=tgt, kernel=KERNEL)
        is_bnd = (x_eval < h) | (x_eval > 1 - h)
        is_int = ~is_bnd
        rows.append({
            "target": tgt.name, "h": h,
            "NW_bias_interior": np.sqrt(np.nanmean(b_nw[is_int] ** 2)),
            "LL_bias_interior": np.sqrt(np.nanmean(b_ll[is_int] ** 2)),
            "NW_bias_batas":    np.sqrt(np.nanmean(b_nw[is_bnd] ** 2)),
            "LL_bias_batas":    np.sqrt(np.nanmean(b_ll[is_bnd] ** 2)),
            "NW_bias_x=1":      b_nw[-1],
            "LL_bias_x=1":      b_ll[-1],
        })

bias_tbl = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.float_format", "{:.5f}".format)
print(bias_tbl.to_string(index=False))

print()
print("=" * 74)
print(f"(B) VALIDASI MONTE CARLO  (R = {R}, h = {h_ref}, sigma = {SIGMA})")
print("=" * 74)

for tgt in TARGETS:
    mX = tgt(X)
    L0 = weight_matrix(x_eval, X, h_ref, p=0, kernel=KERNEL)
    L1 = weight_matrix(x_eval, X, h_ref, p=1, kernel=KERNEL)
    acc0 = np.zeros(len(x_eval)); acc1 = np.zeros(len(x_eval))
    for _ in range(R):
        Y = mX + rng.normal(0, SIGMA, n)
        acc0 += L0 @ Y
        acc1 += L1 @ Y
    bias_mc_nw = acc0 / R - tgt(x_eval)
    bias_mc_ll = acc1 / R - tgt(x_eval)
    bias_ex_nw = exact_bias(x_eval, X, h_ref, 0, tgt, KERNEL)
    bias_ex_ll = exact_bias(x_eval, X, h_ref, 1, tgt, KERNEL)
    d_nw = np.nanmax(np.abs(bias_mc_nw - bias_ex_nw))
    d_ll = np.nanmax(np.abs(bias_mc_ll - bias_ex_ll))
    print(f"{tgt.name:8s} | max|bias_MC - bias_eksak|  NW={d_nw:.5f}  LL={d_ll:.5f}")

print()
print("=" * 74)
print("(C) VARIANS  -- galat i.i.d. vs AR(1) (rho = 0.7)")
print("=" * 74)

Sig = ar1_corr(n, 0.7)
rows = []
for h in H_GRID:
    for p, lbl in [(0, "NW"), (1, "LL")]:
        v_iid = exact_variance(x_eval, X, h, p, SIGMA ** 2, KERNEL)
        v_ar1 = exact_variance(x_eval, X, h, p, SIGMA ** 2, KERNEL, Sigma=Sig)
        is_bnd = (x_eval < h) | (x_eval > 1 - h)
        rows.append({
            "h": h, "est": lbl,
            "Var_iid_interior": np.nanmean(v_iid[~is_bnd]),
            "Var_iid_batas":    np.nanmean(v_iid[is_bnd]),
            "Var_AR1_interior": np.nanmean(v_ar1[~is_bnd]),
            "Var_AR1_batas":    np.nanmean(v_ar1[is_bnd]),
        })
print(pd.DataFrame(rows).to_string(index=False))

out_path = OUT_DIR / "sim_bias_table.csv"
bias_tbl.to_csv(out_path, index=False)
print(f"\n[tersimpan] {out_path}")
