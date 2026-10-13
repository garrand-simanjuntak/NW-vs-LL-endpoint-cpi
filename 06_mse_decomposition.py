"""
06_mse_decomposition.py
MSE gabungan EKSAK (tanpa Monte Carlo) pada desain tetap:

    MSE(x) = Bias(x)^2 + Var(x)
           = [w(x)'m - m(x)]^2  +  sigma^2 * w(x)' Sigma w(x)

Kedua komponen deterministik begitu (m, h, K, X, Sigma) ditetapkan.

Dua pertanyaan yang dijawab:
  Q1. Apakah LL benar-benar menang secara MSE, atau kenaikan varians di
      batas memakan habis keuntungan penghapusan biasnya?
  Q2. Bagaimana autokorelasi AR(1) menggeser jawaban Q1?

Catatan struktural: di INTERIOR, w_NW == w_LL persis (desain equidistant),
sehingga MSE keduanya identik di sana. Seluruh perbedaan berasal dari BATAS.
Karena itu MISE global MELEMAHKAN (dilute) sinyal -- MISE khusus-batas
dilaporkan terpisah.
"""
import numpy as np
import pandas as pd
from lpr_core import exact_bias, exact_variance, ar1_corr
from sim_functions import TARGETS

from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)


n = 198
X = np.arange(1, n + 1) / n
KERNEL = "epanechnikov"
SIGMA = 0.30
RHO = 0.7
H_GRID = np.array([0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25])

Sig_ar1 = ar1_corr(n, RHO)
pd.set_option("display.width", 220)


def mse_components(tgt, h, p, Sigma=None):
    b = exact_bias(X, X, h, p, tgt, KERNEL)
    v = exact_variance(X, X, h, p, SIGMA**2, KERNEL, Sigma=Sigma)
    return b**2, v, b**2 + v


def summarize(tgt, Sigma, label):
    rows = []
    for h in H_GRID:
        is_bnd = (X < h) | (X > 1 - h)
        rec = {"h": h, "n_batas": int(is_bnd.sum())}
        for p, lbl in [(0, "NW"), (1, "LL")]:
            b2, v, mse = mse_components(tgt, h, p, Sigma)
            rec[f"MISE_{lbl}"]      = np.nanmean(mse)              # global
            rec[f"MISEbatas_{lbl}"] = np.nanmean(mse[is_bnd])      # khusus batas
            rec[f"bias2b_{lbl}"]    = np.nanmean(b2[is_bnd])
            rec[f"varb_{lbl}"]      = np.nanmean(v[is_bnd])
        rec["rasio_MISE_NW/LL"]      = rec["MISE_NW"] / rec["MISE_LL"]
        rec["rasio_MISEbatas_NW/LL"] = rec["MISEbatas_NW"] / rec["MISEbatas_LL"]
        rows.append(rec)
    df = pd.DataFrame(rows)

    print(f"\n{'=' * 100}")
    print(f"{tgt.name}  |  {label}  |  sigma={SIGMA}")
    print(f"{'=' * 100}")
    cols = ["h", "MISE_NW", "MISE_LL", "rasio_MISE_NW/LL",
            "MISEbatas_NW", "MISEbatas_LL", "rasio_MISEbatas_NW/LL"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    # h optimal masing-masing estimator (meminimumkan MISE global)
    h_nw = df.loc[df.MISE_NW.idxmin(), "h"]
    h_ll = df.loc[df.MISE_LL.idxmin(), "h"]
    m_nw = df.MISE_NW.min()
    m_ll = df.MISE_LL.min()
    print(f"\n  h* (MISE global)   NW: {h_nw:.3f} -> MISE={m_nw:.6f}"
          f"   |   LL: {h_ll:.3f} -> MISE={m_ll:.6f}"
          f"   |   keunggulan LL: {m_nw/m_ll:.3f}x")

    # dekomposisi di batas pada h optimal LL
    r = df.loc[df.MISE_LL.idxmin()]
    print(f"  Dekomposisi DI BATAS pada h*_LL={h_ll:.3f}:")
    print(f"    NW: bias^2={r.bias2b_NW:.6f}  var={r.varb_NW:.6f}  -> {r.MISEbatas_NW:.6f}")
    print(f"    LL: bias^2={r.bias2b_LL:.6f}  var={r.varb_LL:.6f}  -> {r.MISEbatas_LL:.6f}")
    frac_nw = r.bias2b_NW / r.MISEbatas_NW * 100
    frac_ll = r.bias2b_LL / r.MISEbatas_LL * 100
    print(f"    Porsi bias^2 dlm MSE batas:  NW {frac_nw:.1f}%   LL {frac_ll:.1f}%")
    return df


print("#" * 100)
print("BAGIAN 1 — GALAT i.i.d.")
print("#" * 100)
for tgt in TARGETS:
    summarize(tgt, None, "galat i.i.d.")

print()
print("#" * 100)
print(f"BAGIAN 2 — GALAT AR(1), rho = {RHO}")
print("#" * 100)
for tgt in TARGETS:
    summarize(tgt, Sig_ar1, f"galat AR(1) rho={RHO}")
