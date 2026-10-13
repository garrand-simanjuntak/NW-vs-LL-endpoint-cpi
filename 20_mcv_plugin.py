"""
20_mcv_plugin.py
Revisi putaran 2 (Reviewer 2): ME-2, ME-3, ME-4, HA-2, TB-4.

  ME-2  rho tidak pernah diestimasi. Tiga estimator dibandingkan:
          D1      rho_hat = 1 + 2 r1(Delta Y)          (selisih pertama, tanpa bandwidth pilot)
          D2      inversi r1(Delta^2 Y) = -(rho^2 - 3 rho + 4) / (2 (3 - rho))
          RES-CV  r1 residual fit LL pada h pilihan CV biasa (jalur pilot yang sirkular)
        lalu plug-in rho_hat -> l_hat -> h_hat lewat MCV, dibandingkan dengan oracle.
        Varian konservatif D1+1SE memakai rho_hat + sqrt(2/n).
  ME-3  aturan terpadu l*(rho) = min{ l >= 0 : rho^l <= 0.1 }, satu kriteria untuk semua rho
        (korelasi titik terhapus terluar <= 0.1). l = 0 tidak pernah memenuhi (rho^0 = 1), jadi
        blok terkecil l = 1 untuk rho <= 0.1. Aturan lama (Pers. 20 naskah R1) dan aturan l* - 1
        ikut dihitung untuk reproduksi Tabel 4.
  ME-4  kalibrasi diulang pada n = 400; porsi jendela yang terhapus s = (2 l* + 1)/(2 n h_oracle).
        Rasio kebocoran terhadap varians kriteria MCV, orde pertama untuk AR(1) dan Epanechnikov:
        lambda(l) = 4 K(0) rho^(l+1) / ((1 + rho) int K^2) = 5 rho^(l+1) / (1 + rho), tanpa n dan h.
        Bagian (5) membandingkannya dengan bobot LL eksak; P(lambda(l_hat) >= 1) dicatat per sel.
  HA-2  per replikasi: |log(h_hat / h_oracle)| dan rugi ASE relatif ASE(h_hat)/ASE(h_oracle) - 1.
  TB-4  keluaran disusun untuk Tabel 4a (lokasi) dan 4b (dispersi dan akurasi).

Mesin MCV tervektorisasi: untuk tiap (h, l) matriks leave-block-out M dibentuk sekali dari
jumlah kernel S0, S1, S2 pada titik yang tersisa, lalu skor MCV = mean((Y - M Y)^2) untuk semua
replikasi sekaligus. Kelayakan sama dengan 17_mcv_dense_ell.py: (h, l) layak bila setiap titik
menyisakan >= 3 titik berbobot positif. Untuk n = 198 aliran RNG identik dengan
17_mcv_dense_ell.py (seed 2026, urutan target M2, M1, M3 dan rho), sehingga Tabel 4 naskah
tereproduksi persis; n = 400 memakai seed 2027.

Keluaran: output/k1_selection.csv, output/k1_rho_estimates.csv, output/k1_feasibility.csv,
          output/k1_lambda.csv
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sim_functions import M1, M2, M3

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"; OUT.mkdir(exist_ok=True)

H_GRID = np.round(np.arange(0.03, 0.31, 0.01), 2)
STEP, SIGMA, R = 0.01, 0.30, 200
RHOS = [0.0, 0.3, 0.6, 0.7, 0.9]
TARGETS = [M2, M1, M3]
SEEDS = {198: 2026, 400: 2027}
DENSE = list(range(26))
RHO_CAP = 0.99
pd.set_option("display.width", 250)


# ---------- aturan panjang blok ----------
def rule_old(rho):
    """Pers. (20) naskah R1: l = 0 untuk rho <= 0.1, selain itu ceil(ln 0.1 / ln rho)."""
    return 0 if rho <= 0.1 else int(np.ceil(np.log(0.1) / np.log(rho) - 1e-12))


def rule_minus1(rho):
    return max(rule_old(rho) - 1, 0)


def rule_unified(rho):
    """min{l >= 0: rho^l <= 0.1}, = 1 untuk rho <= 0.1; rho_hat negatif dipotong ke 0,
    rho_hat >= 1 dipotong ke RHO_CAP."""
    r = min(max(float(rho), 0.0), RHO_CAP)
    return 1 if r <= 0.1 else int(np.ceil(np.log(0.1) / np.log(r) - 1e-12))


# ---------- estimator rho ----------
def acf1(Z):
    Z = Z - Z.mean(axis=0)
    return (Z[1:] * Z[:-1]).sum(axis=0) / (Z * Z).sum(axis=0)


def rho_d1(Y):
    return 1.0 + 2.0 * acf1(np.diff(Y, axis=0))


def rho_d2(Y):
    r = np.clip(acf1(np.diff(Y, 2, axis=0)), -0.999999, -0.500001)
    return (3 + 2 * r - np.sqrt(4 * r**2 - 12 * r - 7)) / 2


# ---------- mesin LL leave-block-out ----------
def lbo_matrix(X, h, ell):
    """Matriks bobot fit LL di x_i tanpa titik |j - i| <= ell (ell = -1: fit penuh).
    None bila ada titik yang menyisakan < 3 titik berbobot positif."""
    n = len(X)
    D = X[None, :] - X[:, None]
    U = D / h
    K = np.where(np.abs(U) <= 1, 0.75 * (1 - U**2), 0.0)
    if ell >= 0:
        idx = np.arange(n)
        K = K * (np.abs(idx[None, :] - idx[:, None]) > ell)
    if ((K > 0).sum(axis=1) < 3).any():
        return None
    S0, S1, S2 = K.sum(1), (K * D).sum(1), (K * D * D).sum(1)
    return K * (S2[:, None] - S1[:, None] * D) / (S0 * S2 - S1**2)[:, None]


def mcv_scores(X, Y, ell):
    sc = np.full((len(H_GRID), Y.shape[1]), np.inf)
    for j, h in enumerate(H_GRID):
        M = lbo_matrix(X, h, ell)
        if M is not None:
            sc[j] = np.mean((Y - M @ Y) ** 2, axis=0)
    return sc


def ar1_errors(rng, rho, size):
    """Identik dengan 17_mcv_dense_ell.py (rekursi skalar, start stasioner)."""
    e = np.empty(size)
    e[0] = rng.normal(0, SIGMA)
    s = SIGMA * np.sqrt(1 - rho**2)
    for t in range(1, size):
        e[t] = rho * e[t - 1] + rng.normal(0, s)
    return e


sel_rows, est_rows, feas_rows = [], [], []
for n in (198, 400):
    t_start = time.time()
    X = np.arange(1, n + 1) / n
    rng = np.random.default_rng(SEEDS[n])
    blocks, Ycols, Mcols = [], [], []
    for tf in TARGETS:
        mX = tf(X)
        for rho in RHOS:
            Ymat = np.column_stack([mX + (ar1_errors(rng, rho, n) if rho > 0 else rng.normal(0, SIGMA, n))
                                    for _ in range(R)])
            blocks.append((tf.name, rho, len(Ycols) * R))
            Ycols.append(Ymat); Mcols.append(np.repeat(mX[:, None], R, axis=1))
    Y, Mtrue = np.hstack(Ycols), np.hstack(Mcols)
    N = Y.shape[1]

    # fit penuh: oracle ASE, GCV, residual untuk RES-CV
    ASE = np.empty((len(H_GRID), N)); GCV = np.empty_like(ASE); RES_R1 = np.empty_like(ASE)
    for j, h in enumerate(H_GRID):
        L = lbo_matrix(X, h, -1)
        F = L @ Y
        ASE[j] = np.mean((F - Mtrue) ** 2, axis=0)
        GCV[j] = np.mean((Y - F) ** 2, axis=0) / (1 - np.trace(L) / n) ** 2
        RES_R1[j] = acf1(Y - F)
    cols = np.arange(N)
    i_or = np.argmin(ASE, axis=0)

    # MCV pada grid l padat + kelayakan
    SC = {ell: mcv_scores(X, Y, ell) for ell in DENSE}
    i_cv = np.argmin(SC[0], axis=0)

    rho_hat = {"D1": rho_d1(Y), "D2": rho_d2(Y), "RES-CV": RES_R1[i_cv, cols]}
    rho_hat["D1+1SE"] = rho_hat["D1"] + np.sqrt(2.0 / n)
    ell_hat = {k: np.array([rule_unified(v) for v in vals]) for k, vals in rho_hat.items()}

    needed = sorted(set(np.concatenate(list(ell_hat.values())).tolist())
                    | {rule_unified(r) for r in RHOS} | {rule_old(r) for r in RHOS})
    for ell in needed:
        if ell not in SC:
            SC[ell] = mcv_scores(X, Y, ell)
    feasible = {ell: np.isfinite(SC[ell][:, 0]) for ell in SC}
    ell_max_ok = max(e for e in SC if feasible[e].any())
    for ell in sorted(SC):
        ok = feasible[ell]
        feas_rows.append({"n": n, "ell": ell, "h_min_feasible": H_GRID[ok].min() if ok.any() else np.nan,
                          "n_feasible_h": int(ok.sum())})

    def pick(ell, idx):
        e = ell if feasible[ell].any() else ell_max_ok
        return np.argmin(SC[e][:, idx], axis=0)

    def summarize(rec, isel, idx):
        hs, ho = H_GRID[isel], H_GRID[i_or[idx]]
        loss = ASE[isel, idx] / ASE[i_or[idx], idx] - 1
        rec.update({"h_oracle_med": np.median(ho), "h_sel_med": np.median(hs),
                    "ratio": np.median(hs) / np.median(ho),
                    "IQR_lo": np.percentile(hs, 25), "IQR_hi": np.percentile(hs, 75),
                    "within_1step": np.mean(np.abs(hs - ho) <= STEP + 1e-9),
                    "med_abs_log": np.median(np.abs(np.log(hs / ho))),
                    "loss_med": np.median(loss), "loss_mean": np.mean(loss),
                    "loss_p90": np.percentile(loss, 90),
                    "at_grid_floor": np.mean(hs <= H_GRID[0] + 1e-9),
                    "at_grid_top": np.mean(hs >= H_GRID[-1] - 1e-9)})
        sel_rows.append(rec)

    for tname, rho, c0 in blocks:
        idx = np.arange(c0, c0 + R)
        base = {"n": n, "target": tname, "rho": rho}
        summarize({**base, "method": "GCV", "ell": np.nan}, np.argmin(GCV[:, idx], axis=0), idx)
        for ell in DENSE:
            summarize({**base, "method": "CV" if ell == 0 else "MCV", "ell": ell},
                      np.argmin(SC[ell][:, idx], axis=0), idx)
        for lab, rule in (("rule_old", rule_old), ("rule_minus1", rule_minus1), ("rule_unified", rule_unified)):
            ell = rule(rho)
            summarize({**base, "method": lab, "ell": ell, "s_window": (2 * ell + 1) / (2 * n *
                       np.median(H_GRID[i_or[idx]]))}, pick(ell, idx), idx)
        l_true = rule_unified(rho)
        for est in ("D1", "D1+1SE", "D2", "RES-CV"):
            lh = ell_hat[est][idx]
            isel = np.empty(R, dtype=int)
            for e in np.unique(lh):
                k = lh == e
                isel[k] = pick(int(e), idx[k])
            summarize({**base, "method": f"plugin_{est}", "ell": np.nan,
                       "ell_hat_med": np.median(lh), "ell_hat_p25": np.percentile(lh, 25),
                       "ell_hat_p75": np.percentile(lh, 75),
                       "share_fallback": np.mean([not feasible[int(e)].any() for e in lh])}, isel, idx)
            rh = rho_hat[est][idx]
            lam = 5 * rho ** (lh + 1) / (1 + rho)
            est_rows.append({**base, "estimator": est, "mean": rh.mean(), "sd": rh.std(ddof=1),
                             "rmse": np.sqrt(np.mean((rh - rho) ** 2)),
                             "p05": np.percentile(rh, 5), "p95": np.percentile(rh, 95),
                             "ell_true": l_true, "ell_hat_med": np.median(lh),
                             "P_equal": np.mean(lh == l_true), "P_shorter": np.mean(lh < l_true),
                             "P_longer": np.mean(lh > l_true), "P_short2": np.mean(lh <= l_true - 2),
                             "P_lam_ge_05": np.mean(lam >= 0.5), "P_lam_ge_1": np.mean(lam >= 1)})
    print(f"n = {n}: selesai dalam {time.time() - t_start:.0f} s; l terbesar yang layak = {ell_max_ok}")

sel = pd.DataFrame(sel_rows); est = pd.DataFrame(est_rows); feas = pd.DataFrame(feas_rows)
sel.to_csv(OUT / "k1_selection.csv", index=False)
est.to_csv(OUT / "k1_rho_estimates.csv", index=False)
feas.to_csv(OUT / "k1_feasibility.csv", index=False)
f3 = lambda v: f"{v:.3f}"

# ---------- (1) reproduksi Tabel 4 naskah (n = 198, M2) ----------
PAPER = {0.0: (0.13, 1.000, 1.000, None, 1.000, 0.185, 0.11, 0.16),
         0.3: (0.15, 0.267, 0.200, 0.933, 1.000, 0.175, 0.11, 0.18),
         0.6: (0.18, 0.167, 0.167, 0.889, 1.000, 0.165, 0.13, 0.22),
         0.7: (0.20, 0.150, 0.150, 0.950, 0.950, 0.140, 0.16, 0.24),
         0.9: (0.23, 0.130, 0.130, 1.304, 1.326, 0.185, 0.25, 0.31)}
print("\n(1) REPRODUKSI TABEL 4 NASKAH (n = 198, e^{2x}); 'ok' = sama dengan naskah pada 3 desimal")
g = sel[(sel.n == 198) & (sel.target == "M2_exp")]
all_ok = True
for rho, (hor, cv, gcv, m1, m0, w1, q1, q3) in PAPER.items():
    r = g[g.rho == rho].set_index("method")
    got = (r.loc["CV", "h_oracle_med"], r.loc["CV", "ratio"], r.loc["GCV", "ratio"],
           None if m1 is None else r.loc["rule_minus1", "ratio"], r.loc["rule_old", "ratio"],
           r.loc["rule_old", "within_1step"], r.loc["rule_old", "IQR_lo"], r.loc["rule_old", "IQR_hi"])
    ok = all(a is None or abs(round(a, 3) - b) < 1e-9 for a, b in zip(got, (hor, cv, gcv, m1, m0, w1, q1, q3)))
    all_ok &= ok
    print(f"  rho={rho:.1f}: " + " ".join("--" if a is None else f3(a) for a in got) + f"  -> {'ok' if ok else 'BEDA'}")
print(f"  semua sel Tabel 4 tereproduksi: {all_ok}")

# ---------- (2) estimator rho ----------
print("\n(2) ESTIMATOR rho: bias, SD, RMSE, dan distribusi l_hat (aturan terpadu)")
print(est.to_string(index=False, float_format=f3))

# ---------- (3) Tabel 4a/4b (n = 198, e^{2x}) dan ringkasan semua sel ----------
# Rincian lengkap per metode dan per l ada di output/k1_selection.csv.
print("\n(3a) TABEL 4a/4b (n = 198, e^{2x}); F = median di lantai grid, T = kuartil atas di puncak grid")
g = sel[(sel.n == 198) & (sel.target == "M2_exp")]
for rho in RHOS:
    r = g[g.rho == rho].set_index("method")
    lu = int(r.loc["rule_unified", "ell"])
    m1 = g[(g.rho == rho) & (g.ell == lu - 1) & g.method.isin(["CV", "MCV"])].iloc[0]
    flag = lambda m: ("F" if r.loc[m, "h_sel_med"] <= H_GRID[0] + 1e-9 else "") + \
                     ("T" if r.loc[m, "IQR_hi"] >= H_GRID[-1] - 1e-9 else "")
    print(f"  4a rho={rho:.1f} h_or={r.loc['CV', 'h_oracle_med']:.2f} CV {r.loc['CV', 'ratio']:.3f}{flag('CV')} "
          f"GCV {r.loc['GCV', 'ratio']:.3f}{flag('GCV')} l*={lu} MCV_l*-1 {m1.ratio:.3f} "
          f"MCV_l* {r.loc['rule_unified', 'ratio']:.3f}{flag('rule_unified')} "
          f"MCV_lhat {r.loc['plugin_D1', 'ratio']:.3f}{flag('plugin_D1')} "
          f"(fallback {r.loc['plugin_D1', 'share_fallback']:.2f})")
for rho in RHOS:
    r = g[g.rho == rho].set_index("method")
    u, p = r.loc["rule_unified"], r.loc["plugin_D1"]
    print(f"  4b rho={rho:.1f} l_hat {p.ell_hat_med:.0f} ({p.ell_hat_p25:.0f}-{p.ell_hat_p75:.0f}) "
          f"w1 {u.within_1step:.3f} IQR {u.IQR_lo:.2f}-{u.IQR_hi:.2f} |log| l* {u.med_abs_log:.3f} "
          f"plug {p.med_abs_log:.3f} CV {r.loc['CV', 'med_abs_log']:.3f} | rugi CV {r.loc['CV', 'loss_med']:.3f} "
          f"l* {u.loss_med:.3f} plug {p.loss_med:.3f} | s {u.s_window:.3f}")
SHOW = ["CV", "GCV", "rule_minus1", "rule_unified", "plugin_D1", "plugin_D1+1SE", "plugin_D2", "plugin_RES-CV"]
for val in ("ratio", "loss_med"):
    pv = sel[sel.method.isin(SHOW)].pivot_table(index=["n", "target", "rho"], columns="method", values=val)[SHOW]
    print(f"\n(3b) {val} per sel (baris n, target, rho; kolom metode)")
    print(pv.to_string(float_format=f3))

# ---------- (4) rho = 0: biaya blok pada galat tak berkorelasi ----------
print("\n(4) rho = 0 (n = 198, e^{2x}): ratio dan rugi ASE per panjang blok")
z = sel[(sel.n == 198) & (sel.target == "M2_exp") & (sel.rho == 0.0) & sel.method.isin(["CV", "MCV"])]
print(z[["ell", "ratio", "within_1step", "med_abs_log", "loss_med", "loss_mean"]].head(10)
      .to_string(index=False, float_format=f3))

# ---------- (5) rasio kebocoran lambda: bobot LL eksak vs orde pertama ----------
def lam_exact(n, h, ell, rho):
    """2 * kebocoran / varians fit leave-block-out di titik interior i = n // 2."""
    X = np.arange(1, n + 1) / n
    i = n // 2
    w = lbo_matrix(X, h, ell)[i]
    k = np.arange(n)
    return 2 * (w * rho ** np.abs(k - i)).sum() / (w @ (rho ** np.abs(k[:, None] - k[None, :])) @ w)

lam_rows = []
for n in (198, 400):
    for h in (0.10, 0.15, 0.20, 0.25):
        for rho in (0.3, 0.6, 0.7, 0.9):
            for ell in sorted({0, rule_unified(rho) - 1, rule_unified(rho)}):
                if lbo_matrix(np.arange(1, n + 1) / n, h, ell) is None:
                    continue
                lam_rows.append({"n": n, "h": h, "nh": n * h, "rho": rho, "ell": ell,
                                 "is_rule": ell == rule_unified(rho),
                                 "lam_exact": lam_exact(n, h, ell, rho),
                                 "lam_approx": 5 * rho ** (ell + 1) / (1 + rho)})
lam = pd.DataFrame(lam_rows)
lam["shrink_exact"] = np.clip(1 - lam.lam_exact, 0, None) ** 0.2
lam.to_csv(OUT / "k1_lambda.csv", index=False)
print("\n(5) LAMBDA = 2 x kebocoran / varians (titik interior): eksak vs 5 rho^(l+1)/(1+rho)")
print(lam.to_string(index=False, float_format=f3))
q = lam[lam.is_rule & (lam.rho <= 0.7)]
print(f"  aturan l*, rho <= 0.7: lambda eksak {q.lam_exact.min():.3f} - {q.lam_exact.max():.3f}; "
      f"|eksak - aproksimasi| maks {np.abs(q.lam_exact - q.lam_approx).max():.3f}; "
      f"faktor (1 - lambda)^(1/5) min {q.shrink_exact.min():.3f}")
q0 = lam[lam.ell == 0]
print(f"  CV (l = 0): lambda eksak {q0.lam_exact.min():.3f} - {q0.lam_exact.max():.3f} (>= 1 berarti kolaps); "
      f"aproksimasi = 1 pada rho = 0.25")

# ---------- (6) angka yang dikutip naskah (audit P6) ----------
print("\n(6) ANGKA NASKAH")
def cell(n, t, rho, m):
    return sel[(sel.n == n) & (sel.target == t) & np.isclose(sel.rho, rho) & (sel.method == m)].iloc[0]
TG = ["M2_exp", "M1_sin", "M3_quad"]
low = [(n, t, r) for n in (198, 400) for t in TG for r in RHOS if r <= 0.7]
gap = [cell(n, t, r, "plugin_D1").loss_med - cell(n, t, r, "rule_unified").loss_med for n, t, r in low]
pl = [cell(n, t, r, "plugin_D1").loss_med for n, t, r in low]
cvh = [cell(n, t, r, "CV").loss_med for n in (198, 400) for t in TG for r in RHOS if 0.3 <= r <= 0.7]
print(f"  rho <= 0.7, 3 target x 2 n: rugi median plug-in D1 {min(pl):.3f} - {max(pl):.3f}; "
      f"selisih vs l* benar {min(gap):+.3f} s.d. {max(gap):+.3f}; CV pada 0.3 <= rho <= 0.7: {min(cvh):.3f} - {max(cvh):.3f}")
for n in (198, 400):
    rr = [cell(n, t, r, "rule_unified").ratio for t in TG for r in RHOS if r <= 0.7]
    ss = [cell(n, t, r, "rule_unified").s_window for t in TG for r in RHOS if r <= 0.7]
    s9 = "; ".join(f"{t} s = {cell(n, t, 0.9, 'rule_unified').s_window:.3f}, rasio = "
                   f"{cell(n, t, 0.9, 'rule_unified').ratio:.3f}" for t in TG)
    print(f"  n = {n}: rasio aturan l*, rho <= 0.7: {min(rr):.3f} - {max(rr):.3f}; s maks {max(ss):.3f}; "
          f"rho = 0.9: {s9}")
d1 = est[est.estimator == "D1"]
print(f"  D1: |bias| maks {np.abs(d1['mean'] - d1.rho).max():.3f}; SD n = 198 {d1[d1.n == 198].sd.min():.3f} - "
      f"{d1[d1.n == 198].sd.max():.3f}, n = 400 {d1[d1.n == 400].sd.min():.3f} - {d1[d1.n == 400].sd.max():.3f}")
d1m = d1[(d1.rho > 0) & (d1.rho <= 0.7)]
print(f"  D1, 0.3 <= rho <= 0.7: P(l_hat <= l* - 2) maks {d1m.P_short2.max():.3f}; "
      f"P(lambda >= 0.5) maks {d1m.P_lam_ge_05.max():.3f}; replikasi dengan lambda >= 1: "
      f"{int(round((d1m.P_lam_ge_1 * R).sum()))} dari {len(d1m) * R}")
for n in (198, 400):
    for r in (0.3, 0.6, 0.7):
        v = sel[(sel.n == n) & np.isclose(sel.rho, r) & sel.method.isin(["CV", "MCV"])].groupby("ell").loss_med.mean()
        print(f"  rugi median rata-rata 3 target, n = {n}, rho = {r}: "
              + " ".join(f"{int(e)}:{x:.2f}" for e, x in v.items() if e <= 25))
print(f"\n[tersimpan] {OUT / 'k1_selection.csv'}, {OUT / 'k1_rho_estimates.csv'}, "
      f"{OUT / 'k1_feasibility.csv'}, {OUT / 'k1_lambda.csv'}")
