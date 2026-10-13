"""
17_mcv_dense_ell.py
Kalibrasi panjang blok MCV pada grid ell padat (Reviewer 1, komentar
Eq. (19), Desain Simulasi, Tabel 4). Melengkapi 11_bandwidth_mcv.py.

DUA ATURAN YANG DIUJI
  Eq. (19)  : ell = ceil( ln 0.1 / ln rho )      -> rho^ell <= 0.1
              (korelasi dengan tetangga TERLUAR yang dihapus <= 10%)
  Reviewer  : ell = ceil( ln 0.1 / ln rho ) - 1  -> rho^(ell+1) <= 0.1
              (korelasi dengan titik PERTAMA di luar blok <= 10%)
  Untuk rho <= 0.1 kedua aturan memberi ell = 0 (CV biasa).

PERUBAHAN TERHADAP SKRIP 11
  * ell = 0, 1, ..., 25 sehingga kedua aturan teruji langsung
  * rho = 0.7 ditambahkan (menyamakan dengan simulasi MISE, Gambar 2)
  * dispersi dilaporkan: IQR h terpilih dan proporsi replikasi dengan
    |h terpilih - h oracle| <= 1 langkah grid (0.01), dihitung per replikasi
  * GCV ikut dihitung (CV biasa = MCV dengan ell = 0)
  * tiga fungsi target: e^{2x} (utama, Tabel 4), sin(2 pi x), -2(x-1)^2
Kelayakan: blok panjang tidak terdefinisi pada h kecil; sel tersebut
ditandai "trunc" bila median h terpilih sama dengan h terkecil yang layak.
Keluaran: output/mcv_dense_ell.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sim_functions import M1, M2, M3

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"; OUT_DIR.mkdir(exist_ok=True)

n = 198
X = np.arange(1, n + 1) / n
H_GRID = np.round(np.arange(0.03, 0.31, 0.01), 2)
STEP = 0.01
P = 1
SIGMA = 0.30
RHOS = [0.0, 0.3, 0.6, 0.7, 0.9]
ELLS = list(range(26))
R = 200
rng = np.random.default_rng(2026)


def epan(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)


def wrow(x0, Xv, h, p):
    w = epan((Xv - x0) / h)
    if (w > 0).sum() < p + 2:
        return None
    Z = np.vander(Xv - x0, N=p + 1, increasing=True)
    ZtW = Z.T * w
    try:
        return (np.linalg.inv(ZtW @ Z) @ ZtW)[0]
    except np.linalg.LinAlgError:
        return None


def rule_eq19(rho):
    return 0 if rho <= 0.1 else int(np.ceil(np.log(0.1) / np.log(rho) - 1e-12))


def rule_reviewer(rho):
    return max(rule_eq19(rho) - 1, 0)


print("pra-hitung matriks leave-block-out ...")
idx = np.arange(n)
MCVW, HMIN = {}, {}
for ell in ELLS:
    for h in H_GRID:
        M = np.zeros((n, n)); ok = True
        for i in range(n):
            mask = np.abs(idx - i) > ell
            w = wrow(X[i], X[mask], h, P)
            if w is None:
                ok = False; break
            M[i, mask] = w
        MCVW[(h, ell)] = M if ok else None
    feas = [h for h in H_GRID if MCVW[(h, ell)] is not None]
    HMIN[ell] = min(feas)
HAT = {h: np.vstack([wrow(x, X, h, P) for x in X]) for h in H_GRID}
TRH = {h: np.trace(HAT[h]) for h in H_GRID}
TARGETS_CAL = [M2, M1, M3]


def ar1_errors(rho, size):
    e = np.empty(size)
    e[0] = rng.normal(0, SIGMA)
    s = SIGMA * np.sqrt(1 - rho**2)
    for t in range(1, size):
        e[t] = rho * e[t - 1] + rng.normal(0, s)
    return e


rows = []
for tf in TARGETS_CAL:
    mX = tf(X)
    for rho in RHOS:
        print(f"kalibrasi {tf.name}, rho = {rho} ...")
        Ymat = np.column_stack([mX + (ar1_errors(rho, n) if rho > 0 else rng.normal(0, SIGMA, n))
                                for _ in range(R)])                  # (n, R)
        oracle = np.array([[np.mean((HAT[h] @ Ymat[:, r] - mX) ** 2) for h in H_GRID]
                           for r in range(R)])
        h_or = H_GRID[np.argmin(oracle, axis=1)]

        def record(method, ell, sc, hmin):
            h_sel = H_GRID[np.argmin(sc, axis=0)]
            med = np.median(h_sel)
            rows.append({
                "target": tf.name, "rho": rho, "method": method, "ell": ell,
                "h_oracle_med": np.median(h_or), "h_sel_med": med,
                "ratio": med / np.median(h_or),
                "IQR_lo": np.percentile(h_sel, 25), "IQR_hi": np.percentile(h_sel, 75),
                "within_1step": np.mean(np.abs(h_sel - h_or) <= STEP + 1e-9),
                "h_min_feasible": hmin, "trunc_low": bool(np.isclose(med, hmin)),
                "hits_grid_top": bool(np.percentile(h_sel, 75) >= H_GRID[-1] - 1e-9),
                "rule_eq19": ell == rule_eq19(rho), "rule_reviewer": ell == rule_reviewer(rho),
            })

        gcv = np.array([np.mean((Ymat - HAT[h] @ Ymat) ** 2, axis=0) / (1 - TRH[h] / n) ** 2
                        for h in H_GRID])
        record("GCV", -1, gcv, H_GRID[0])
        for ell in ELLS:
            sc = np.full((len(H_GRID), R), np.inf)
            for j, h in enumerate(H_GRID):
                M = MCVW[(h, ell)]
                if M is not None:
                    sc[j] = np.mean((Ymat - M @ Ymat) ** 2, axis=0)
            record("CV" if ell == 0 else "MCV", ell, sc, HMIN[ell])

res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "mcv_dense_ell.csv", index=False)
pd.set_option("display.width", 220)
f3 = lambda v: f"{v:.3f}"
print("\nATURAN vs HASIL (ratio = median h terpilih / median h oracle; w1 = proporsi dalam +-1 langkah)")
for (tname, rho), g in res.groupby(["target", "rho"], sort=False):
    a, b = rule_eq19(rho), rule_reviewer(rho)
    m = g[g.method != "GCV"]
    best = m.loc[(m.ratio - 1).abs().idxmin()]
    ra, rb = m[m.ell == a].iloc[0], m[m.ell == b].iloc[0]
    cv, gc = m[m.ell == 0].iloc[0], g[g.method == "GCV"].iloc[0]
    print(f"{tname:9s} rho={rho:.1f} h_or={ra.h_oracle_med:.2f} | CV {cv.ratio:.3f} GCV {gc.ratio:.3f}"
          f" | Eq19 ell={a:2d}: {ra.ratio:.3f} (w1 {ra.within_1step:.2f}, IQR {ra.IQR_lo:.2f}-{ra.IQR_hi:.2f}"
          f"{', top' if ra.hits_grid_top else ''})"
          f" | Reviewer ell={b:2d}: {rb.ratio:.3f} | terbaik ell={int(best.ell):2d} ({best.ratio:.3f})")
print("\nTABEL LENGKAP")
print(res[["target", "rho", "method", "ell", "h_oracle_med", "h_sel_med", "ratio", "IQR_lo", "IQR_hi",
           "within_1step", "h_min_feasible", "trunc_low", "hits_grid_top"]]
      .to_string(index=False, float_format=f3))
print(f"\n[tersimpan] {OUT_DIR / 'mcv_dense_ell.csv'}")
