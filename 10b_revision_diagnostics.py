"""
10b_revision_diagnostics.py
Diagnostik lanjutan untuk revision analysis (menjawab Reviewer 1, komentar
Test (A), Tabel 5, dan Gambar 3). Melengkapi 10_revision_analysis.py tanpa
mengubah angka lamanya.

1. BENCHMARK DISKRET. Prediksi 0.375h adalah limit asimtotik. Pada jendela
   satu sisi berisi h titik, revisi NW untuk jalur linear lokal sama dengan
   m'(t0) * sum_k w_k k (rata-rata lag diskret, dari Eq. (7)). Untuk
   Epanechnikov nilainya 0.914-0.968 x 0.375h pada h = 9-24.

2. INFERENSI. Revisi pada t0 berurutan memakai jendela yang tumpang tindih
   sampai lag 2h-2, sehingga galat regresi berautokorelasi kuat.
   Dilaporkan CI 95% Newey-West (Bartlett) dengan L = 2h-2 dan L = h, serta
   moving block bootstrap (blok 2h, B = 2000).

3. KOMPONEN KURVATUR LL. Pada c = 0, suku O(h) LL lenyap, sehingga revisi LL
   digerakkan m''(t0): Revisi_LL ~ (mu2/2 + 0.0579) h^2 m'' = 0.158 h^2 m''
   (asimtotik). Bentuk diskret eksaknya:
       0.5 * [ sum_t w_final(t)(t-t0)^2 - sum_t w_rt,LL(t)(t-t0)^2 ].
   m''(t0) ditaksir dengan fit kuadratik lokal (p = 2) pada sampel penuh.

Keluaran: output/revision_diagnostics.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"; OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(OUT_DIR / "ihk_bps_2010_2026.csv", parse_dates=["date"])
n = len(df)
T = np.arange(1, n + 1, dtype=float)
H_GRID = [9, 12, 18, 24]
T0_MIN = 48
B_BOOT = 2000
rng = np.random.default_rng(2026)


def epan(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)


def lp_rows(t0, Tv, h, p):
    """Baris (Z'WZ)^{-1} Z'W: baris j memberi koefisien beta_j (j! m^(j))."""
    w = epan((Tv - t0) / h)
    Z = np.vander(Tv - t0, N=p + 1, increasing=True)
    ZtW = Z.T * w
    return np.linalg.inv(ZtW @ Z) @ ZtW


def discrete_mean_lag(h):
    k = np.arange(h)
    K = 1 - (k / h) ** 2
    return float((K * k).sum() / K.sum())


def discrete_ll_curvature(h):
    t0 = n // 2                          # titik interior jauh dari kedua ujung
    d = T - t0
    w_fin = lp_rows(t0, T, h, 1)[0]
    w_rt = lp_rows(t0, T[:t0], h, 1)[0]
    return 0.5 * ((w_fin * d**2).sum() - (w_rt * d[:t0] ** 2).sum())


def hac_slope(x, y, L):
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    g = X * (y - X @ b)[:, None]
    S = g.T @ g
    for l in range(1, L + 1):
        G = g[l:].T @ g[:-l]
        S += (1 - l / (L + 1)) * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    return b[1], float(np.sqrt((XtXi @ S @ XtXi)[1, 1]))


def mbb_slope_ci(x, y, block):
    N = len(x)
    nb = int(np.ceil(N / block))
    est = np.empty(B_BOOT)
    for b in range(B_BOOT):
        starts = rng.integers(0, N - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:N]
        est[b] = np.polyfit(x[idx], y[idx], 1)[0]
    return np.percentile(est, [2.5, 97.5])


rows = []
for name in ["log_IHK", "infl_pct_mtm"]:
    Y = df[name].values
    for h in H_GRID:
        slope, curv, rev_nw, rev_ll = [], [], [], []
        for t0 in range(T0_MIN, n - h + 1):
            fin = lp_rows(t0, T, h, 1)[0] @ Y
            rev_nw.append(fin - lp_rows(t0, T[:t0], h, 0)[0] @ Y[:t0])
            rev_ll.append(fin - lp_rows(t0, T[:t0], h, 1)[0] @ Y[:t0])
            slope.append(lp_rows(t0, T, h, 1)[1] @ Y)
            curv.append(2 * lp_rows(t0, T, h, 2)[2] @ Y)
        slope, curv, rev_nw, rev_ll = map(np.array, (slope, curv, rev_nw, rev_ll))

        disc = discrete_mean_lag(h)
        b, se_long = hac_slope(slope, rev_nw, 2 * h - 2)
        _, se_short = hac_slope(slope, rev_nw, h)
        lo_mbb, hi_mbb = mbb_slope_ci(slope, rev_nw, 2 * h)
        b_ll = np.polyfit(curv, rev_ll, 1)[0]
        pred_ll = discrete_ll_curvature(h)

        rows.append({
            "response": name, "h": h, "n_t0": len(slope),
            "slope_NW": b, "pred_asym": 0.375 * h, "pred_discrete": disc,
            "ratio_asym": b / (0.375 * h), "ratio_discrete": b / disc,
            "HAC2h_lo": (b - 1.96 * se_long) / disc, "HAC2h_hi": (b + 1.96 * se_long) / disc,
            "HACh_lo": (b - 1.96 * se_short) / disc, "HACh_hi": (b + 1.96 * se_short) / disc,
            "MBB_lo": lo_mbb / disc, "MBB_hi": hi_mbb / disc,
            "asym_in_HAC2h": (b - 1.96 * se_long) <= 0.375 * h <= (b + 1.96 * se_long),
            "corr_NW_slope": np.corrcoef(slope, rev_nw)[0, 1],
            "corr_LL_slope": np.corrcoef(slope, rev_ll)[0, 1],
            "slope_LL_curv": b_ll, "pred_LL_curv": pred_ll,
            "ratio_LL_curv": b_ll / pred_ll,
            "corr_LL_curv": np.corrcoef(curv, rev_ll)[0, 1],
        })

res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "revision_diagnostics.csv", index=False)

pd.set_option("display.width", 220)
fmt = lambda v: f"{v:.4f}"
for name, g in res.groupby("response", sort=False):
    print("=" * 100)
    print(f"RESPONS: {name}")
    print("=" * 100)
    print("(A) Revisi NW vs m'(t0): benchmark asimtotik dan diskret, CI rasio terhadap diskret")
    print(g[["h", "n_t0", "slope_NW", "pred_asym", "pred_discrete", "ratio_asym",
             "ratio_discrete", "HAC2h_lo", "HAC2h_hi", "HACh_lo", "HACh_hi",
             "MBB_lo", "MBB_hi", "asym_in_HAC2h"]].to_string(index=False, float_format=fmt))
    print("\n(C) Revisi LL vs m''(t0): komponen kurvatur")
    print(g[["h", "corr_LL_slope", "corr_LL_curv", "slope_LL_curv", "pred_LL_curv",
             "ratio_LL_curv"]].to_string(index=False, float_format=fmt))
    print()
print(f"[tersimpan] {OUT_DIR / 'revision_diagnostics.csv'}")
