"""
22_testB_sensitivity.py
Revisi putaran 2 (Reviewer 2), K3: sensitivitas Test (B) dengan bandwidth masing-masing
(Tabel 6 Panel B). Melayani ME-6 (h_ref arbitrer), ME-7 (optimum di tepi grid), DA-5 (COVID di
sampel evaluasi).

Test (B) (Pers. 26):  D_e(h) = mean_t0 [ m_ref(t0) - m_rt,e(t0; h) ]^2,  e in {NW, LL},
m_ref = LL interior pada h_ref, m_rt = estimasi real-time satu sisi di t0. Semua estimasi linear
dalam Y: m_ref = a'Y, m_rt = b'Y. Untuk Y = m + eps, Cov(eps) = Sigma:
  E D_e(h) = mean_t0 { ((a - b)'m)^2 + (a - b)' Sigma (a - b) }
           = MSE_e(h) + MSE_ref - 2 Cross_e(h),
  Cross_e(h) = mean_t0 { (a'm - m_t0)(b'm - m_t0) + a' Sigma b }.
Proxy memihak bila MSE_ref - 2 Cross_e(h) berbeda antar e atau berubah dengan h.

Bagian:
 (1) Reproduksi Tabel 6 Panel B (S0 mentah dan S2), logika identik 16_testB_own_bandwidth.py.
 (2) Kerataan kurva D_e(h): rentang h dalam 1% dan 5% dari minimum, D_LL(3) vs D_LL(21).
 (3) Proxy vs kebenaran pada inflasi (ME-6), model K2 (21_testA_eiv.py): tren LL h = 36 diskalakan
     sampai median SNR = SNR inflasi, galat N(0, Sigma variogram), seed dan 300 replikasi sama.
     (a) eksak: argmin dan pemenang E D_e vs MSE_e, suku silang; (b) Monte Carlo: P(pemenang sama),
     rugi seleksi proxy ASE(h_hat)/ASE(h_oracle) - 1, h terpilih; (c) log CPI dibangun dari inflasi
     simulasi: apakah proxy mereproduksi Panel B (tanpa kebenaran, karena galatnya terintegrasi).
 (4) ME-7: lantai disamakan (NW h >= 1, LL h >= 2, keduanya = Y_t0; syarat p + 1 titik), batas atas
     h <= 72 dengan t0 >= 72, dan baseline t0 >= 72 untuk memisahkan efek rentang t0.
 (5) DA-5: rasio OOS pada 2020-21 (t0 121-144) dan 2022-26 (t0 >= 145), titik pisah 96 dan 144,
     in-sample tanpa 2020-21.

Keluaran: output/k3_panelB.csv, output/k3_curves.csv, output/k3_flat.csv, output/k3_exact.csv,
          output/k3_mc.csv, output/k3_floor_upper.csv, output/k3_oos.csv, output/fig_k3_testB_curves.png
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"; OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 250)
t_start = time.time()

df = pd.read_csv(OUT / "ihk_bps_2010_2026.csv", parse_dates=["date"])
n = len(df)
T = np.arange(1, n + 1, dtype=float)
H_REF = [12, 18, 24]
T0_SPLIT = 120
R_SIM, SEED, H_TREND = 300, 20261010, 36          # sama dengan 21_testA_eiv.py
EID_MONTH = {2010: 9, 2011: 8, 2012: 8, 2013: 8, 2014: 7, 2015: 7, 2016: 7, 2017: 6,
             2018: 6, 2019: 6, 2020: 5, 2021: 5, 2022: 5, 2023: 4, 2024: 4, 2025: 3, 2026: 3}
LAB = {0: "NW", 1: "LL"}
# konfigurasi grid: (h_min, h_max, t0_min, syarat titik positif = p + extra)
CFG = {"base":       (2, 48, 48, 2),             # = 16_testB_own_bandwidth.py (Tabel 6 Panel B)
       "floor_eq":   (1, 48, 48, 1),             # NW h = 1 dan LL h = 2 keduanya = Y_t0
       "base_t72":   (2, 48, 72, 2),             # baseline dengan rentang t0 yang sama dengan upper72
       "upper72":    (2, 72, 72, 2),
       "floor_eq72": (1, 72, 72, 1)}


def epan(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)


def lp0(t0, Tv, h, p, extra=2):
    """Baris pertama (Z'WZ)^{-1} Z'W di t0; None bila titik berbobot positif < p + extra."""
    w = epan((Tv - t0) / h)
    if (w > 0).sum() < p + extra:
        return None
    Z = np.vander(Tv - t0, N=p + 1, increasing=True)
    ZtW = Z.T * w
    return (np.linalg.inv(ZtW @ Z) @ ZtW)[0]


def adjust_s2(r):
    """Identik dengan 16_testB_own_bandwidth.py: dummy bulan + bulan Idulfitri + bulan sebelumnya."""
    month = df.date.dt.month.values
    D = np.column_stack([(month == m).astype(float) for m in range(1, 13)])
    eid = np.array([1.0 if m == EID_MONTH[y] else 0.0 for y, m in zip(df.date.dt.year, month)])
    pre = np.roll(eid, -1); pre[-1] = 0.0
    S = np.column_stack([D, eid, pre]); S = S - S.mean(axis=0)
    Xd = np.column_stack([np.ones(n), S])
    b = np.linalg.lstsq(Xd, r, rcond=None)[0]
    return r - S @ b[1:]


def variogram_cov(y, kmax=36, plateau=(25, 36), lag_max=24):
    """Sama dengan 21_testA_eiv.py dan 23_data_checks.py."""
    V = np.array([0.5 * np.mean((y[k:] - y[:-k]) ** 2) for k in range(1, kmax + 1)])
    g0 = V[plateau[0] - 1:plateau[1]].mean()
    rho = 1 - V[:lag_max] / g0
    return g0, rho, rho * (1 - np.arange(1, lag_max + 1) / (lag_max + 1))


def toeplitz(rho_tap, size):
    col = np.r_[1.0, rho_tap, np.zeros(size - 1 - len(rho_tap))]
    return col[np.abs(np.subtract.outer(np.arange(size), np.arange(size)))]


r0 = df.infl_pct_mtm.values
SERIES = {}
for vname, r in [("S0", r0), ("S2", adjust_s2(r0))]:
    SERIES[(vname, "log CPI")] = np.log(100 * np.cumprod(1 + r / 100))
    SERIES[(vname, "inflation")] = r

# ---------- matriks bobot (tidak bergantung pada Y) ----------
_RT, _REF = {}, {}


def rt_matrix(p, h, t0_min, extra):
    """Baris t0 = t0_min..n-12: bobot real-time satu sisi; None bila tak layak di salah satu t0."""
    key = (p, h, t0_min, extra)
    if key not in _RT:
        t0s = np.arange(t0_min, n - min(H_REF) + 1)
        M = np.zeros((len(t0s), n))
        ok = True
        for i, t0 in enumerate(t0s):
            w = lp0(t0, T[:t0], h, p, extra)
            if w is None:
                ok = False
                break
            M[i, :t0] = w
        _RT[key] = M if ok else None
    return _RT[key]


def ref_matrix(h_ref, t0_min):
    key = (h_ref, t0_min)
    if key not in _REF:
        t0s = np.arange(t0_min, n - h_ref + 1)
        _REF[key] = (t0s, np.vstack([lp0(t0, T, h_ref, 1) for t0 in t0s]))
    return _REF[key]


def grid(cfg):
    h_min, h_max, t0_min, extra = CFG[cfg]
    return np.arange(h_min, h_max + 1), t0_min, extra


def sq_err(Y, cfg, h_ref):
    """{p: array (H, t0)} galat kuadrat terhadap referensi; baris inf bila h tak layak."""
    hs, t0_min, extra = grid(cfg)
    t0s, A = ref_matrix(h_ref, t0_min)
    ref = A @ Y
    out = {}
    for p in (0, 1):
        E2 = np.full((len(hs), len(t0s)), np.inf)
        for j, h in enumerate(hs):
            B = rt_matrix(p, h, t0_min, extra)
            if B is not None:
                E2[j] = (ref - B[:len(t0s)] @ Y) ** 2
        out[p] = E2
    return hs, t0s, out


def select(hs, E2, train, test):
    """h terpilih pada train, MSE pada test (rata-rata atas t0)."""
    m = E2[:, train].mean(axis=1)
    j = int(np.argmin(m))
    return hs[j], E2[j, test].mean(), j


def panel(Y, cfg, h_ref, split=T0_SPLIT, test_mask=None, insample_mask=None):
    hs, t0s, E2 = sq_err(Y, cfg, h_ref)
    allm = np.ones(len(t0s), bool) if insample_mask is None else insample_mask(t0s)
    train = t0s <= split
    test = (t0s > split) if test_mask is None else test_mask(t0s)
    rec = {"n_t0": int(allm.sum()), "n_test": int(test.sum())}
    for p in (0, 1):
        lab = LAB[p]
        h_in, mse_in, j = select(hs, E2[p], allm, allm)
        h_tr, mse_oos, _ = select(hs, E2[p], train, test)
        feas = np.isfinite(E2[p][:, 0])
        rec.update({f"h_{lab}": h_in, f"RMS_{lab}": np.sqrt(mse_in), f"h_{lab}_train": h_tr,
                    f"RMS_{lab}_oos": np.sqrt(mse_oos),
                    f"h_{lab}_edge": ("min" if h_in == hs[feas].min() else
                                      "max" if h_in == hs[feas].max() else "")})
    rec["ratio"] = rec["RMS_NW"] ** 2 / rec["RMS_LL"] ** 2
    rec["ratio_oos"] = rec["RMS_NW_oos"] ** 2 / rec["RMS_LL_oos"] ** 2
    rec["pref"] = "LL" if rec["ratio"] > 1 else "NW"
    rec["pref_oos"] = "LL" if rec["ratio_oos"] > 1 else "NW"
    return rec


f3 = lambda v: f"{v:.3f}"
f5 = lambda v: f"{v:.5g}"

# ---------- (1) reproduksi Tabel 6 Panel B ----------
rows = []
for (vname, resp), Y in SERIES.items():
    for h_ref in H_REF:
        rows.append({"variant": vname, "response": resp, "h_ref": h_ref, **panel(Y, "base", h_ref)})
pb = pd.DataFrame(rows)
pb.to_csv(OUT / "k3_panelB.csv", index=False)
COLS = ["variant", "response", "h_ref", "n_t0", "h_NW", "RMS_NW", "h_LL", "RMS_LL", "ratio", "pref",
        "h_NW_train", "h_LL_train", "ratio_oos", "pref_oos", "h_NW_edge", "h_LL_edge"]
print("(1) REPRODUKSI TABEL 6 PANEL B (grid h = 2..48, t0 = 48..n - h_ref, latih t0 <= 120)")
print(pb[COLS].to_string(index=False, float_format=f5))
EXPECT = {("log CPI", 12): (2, 21, 0.946, 0.624), ("log CPI", 18): (3, 3, 0.825, 0.589),
          ("log CPI", 24): (3, 3, 0.758, 0.919), ("inflation", 12): (18, 45, 0.595, 0.539),
          ("inflation", 18): (19, 46, 0.533, 0.438), ("inflation", 24): (25, 48, 0.487, 0.365)}
ok = all((r.h_NW, r.h_LL, round(r.ratio, 3), round(r.ratio_oos, 3)) == EXPECT[(r.response, r.h_ref)]
         for r in pb[pb.variant == "S0"].itertuples())
print(f"  cocok dengan Tabel 6 Panel B naskah (S0): {ok}")

# ---------- (2) kerataan kurva D_e(h) ----------
curve_rows, flat_rows = [], []
for (vname, resp), Y in SERIES.items():
    for h_ref in H_REF:
        hs, t0s, E2 = sq_err(Y, "floor_eq", h_ref)        # grid 1..48 agar lantai Y_t0 ikut terlihat
        for p in (0, 1):
            Dm = E2[p].mean(axis=1)
            base_mask = hs >= (2 if p == 0 else 3)
            Db = np.where(base_mask, Dm, np.inf)
            jb = int(np.argmin(Db)); Dmin = Db[jb]
            for h, d in zip(hs, Dm):
                curve_rows.append({"variant": vname, "response": resp, "h_ref": h_ref, "est": LAB[p],
                                   "h": h, "D": d, "D_rel": d / Dmin})
            fin = np.isfinite(Db)
            in1, in5 = hs[fin & (Db <= 1.01 * Dmin)], hs[fin & (Db <= 1.05 * Dmin)]
            flat_rows.append({"variant": vname, "response": resp, "h_ref": h_ref, "est": LAB[p],
                              "h_star": hs[jb], "RMS_star": np.sqrt(Dmin),
                              "flat1_lo": in1.min(), "flat1_hi": in1.max(),
                              "flat5_lo": in5.min(), "flat5_hi": in5.max(),
                              "D_floor_rel": Dm[0 if p == 0 else 1] / Dmin})
curves = pd.DataFrame(curve_rows); curves.to_csv(OUT / "k3_curves.csv", index=False)
flat = pd.DataFrame(flat_rows); flat.to_csv(OUT / "k3_flat.csv", index=False)
print("\n(2) KERATAAN D_e(h) (grid baseline): h*, rentang h dengan D <= 1.01 dan 1.05 x minimum, "
      "D di lantai Y_t0 (NW h = 1, LL h = 2) relatif terhadap minimum")
print(flat.to_string(index=False, float_format=f3))
c = curves[(curves.variant == "S0") & (curves.response == "log CPI") & (curves.h_ref == 12)]
dll = c[c.est == "LL"].set_index("h").D; dnw = c[c.est == "NW"].set_index("h").D
print(f"  log CPI S0, h_ref = 12: D_LL(3)/D_LL(21) = {dll[3] / dll[21]:.4f}; D_LL(21)/D_NW(2) = "
      f"{dll[21] / dnw[2]:.4f}; D_LL(3)/D_NW(2) = {dll[3] / dnw[2]:.4f}; D_LL(2) = D_NW(1) = Y_t0: "
      f"{dll[2] / dnw[2]:.4f} x D_NW(2)")
for vname in ("S0", "S2"):                                  # minimum lokal LL pada log CPI (h = 21 vs 3)
    for h_ref in H_REF:
        g = curves[(curves.variant == vname) & (curves.response == "log CPI") & (curves.h_ref == h_ref) &
                   (curves.est == "LL") & (curves.h >= 3)].set_index("h").D
        d, hh = g.to_numpy(), g.index.to_numpy()
        loc = [hh[i] for i in range(len(d)) if (i == 0 or d[i] < d[i - 1]) and (i == len(d) - 1 or d[i] < d[i + 1])]
        print(f"  log CPI {vname}, h_ref = {h_ref}: minimum lokal LL (h, D/min) = "
              + ", ".join(f"({h}, {g[h] / g.min():.3f})" for h in loc))

# gambar untuk Suplemen S6
fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
for i, resp in enumerate(["log CPI", "inflation"]):
    for k, h_ref in enumerate(H_REF):
        ax = axes[i, k]
        g = curves[(curves.variant == "S0") & (curves.response == resp) & (curves.h_ref == h_ref)]
        dmin = g[((g.est == "NW") & (g.h >= 2)) | ((g.est == "LL") & (g.h >= 3))].D.min()
        for est, col, ls in [("NW", "#2a78d6", "-"), ("LL", "#eb6834", (0, (4, 2)))]:
            gg = g[(g.est == est) & np.isfinite(g.D)]
            ax.plot(gg.h, gg.D / dmin, color=col, ls=ls, lw=1.4, label=est)
            fr = flat[(flat.variant == "S0") & (flat.response == resp) & (flat.h_ref == h_ref) &
                      (flat.est == est)].iloc[0]
            ax.plot(fr.h_star, gg.set_index("h").D[fr.h_star] / dmin, "o", color=col, ms=4)
        ax.axvspan(0.5, 2.5, color="0.9", zorder=0)
        ax.set_ylim(0.95, 2.6)
        ax.set_title(f"{resp}, $h_{{ref}}$ = {h_ref}", fontsize=9)
        if k == 0:
            ax.set_ylabel("$D_e(h)$ / min", fontsize=9)
        if i == 1:
            ax.set_xlabel("real-time bandwidth h (months)", fontsize=9)
axes[0, 0].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_k3_testB_curves.png", dpi=200)
plt.close(fig)

# ---------- (3) proxy vs kebenaran: model K2 ----------
r = SERIES[("S0", "inflation")]
g0_r, rho_r, rtap_r = variogram_cov(r)
R_r = toeplitz(rtap_r, n)
SIG_r = g0_r * R_r


def lp_full(t0, h, p):
    u = (T - t0) / h
    w = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
    Z = np.vander(T - t0, N=p + 1, increasing=True)
    ZtW = Z.T * w
    return (np.linalg.inv(ZtW @ Z) @ ZtW)[0]


rng = np.random.default_rng(SEED)
L12 = np.vstack([lp_full(t, 12, 1) for t in T])
LTR = np.vstack([lp_full(t, H_TREND, 1) for t in T])


def snr(Y):
    F = L12 @ Y
    return F.std(axis=0) / (Y - F).std(axis=0)


m_fit = LTR @ r
E = np.linalg.cholesky(SIG_r) @ rng.standard_normal((n, R_SIM))
snr_obs = float(snr(r[:, None])[0])
lo, hi = 0.0, 4.0
for _ in range(50):                                  # identik dengan 21_testA_eiv.py
    c_amp = 0.5 * (lo + hi)
    m_r = m_fit.mean() + c_amp * (m_fit - m_fit.mean())
    if np.median(snr(m_r[:, None] + E)) < snr_obs:
        lo = c_amp
    else:
        hi = c_amp
m_r = m_fit.mean() + c_amp * (m_fit - m_fit.mean())
Y_r = m_r[:, None] + E
Y_l = np.log(100.0) + np.cumsum(np.log1p(Y_r / 100.0), axis=0)
print(f"\n(3) PROXY VS KEBENARAN, model K2: c = {c_amp:.3f}, SNR median simulasi {np.median(snr(Y_r)):.3f} "
      f"(data {snr_obs:.3f}); gamma0 = {g0_r:.4f}; R = {R_SIM}")

exact_rows, mc_rows = [], []
for cfg in ("base", "upper72"):
    hs, t0_min, extra = grid(cfg)
    for h_ref in H_REF:
        t0s, A = ref_matrix(h_ref, t0_min)
        mt = m_r[t0s - 1]
        ba = A @ m_r - mt
        va = (A @ SIG_r * A).sum(1)
        mse_ref = np.mean(ba**2 + va)
        refY = A @ Y_r
        ED, MSE, CR, Dsim, ASE = {}, {}, {}, {}, {}
        for p in (0, 1):
            ed, ms, cr = (np.full(len(hs), np.inf) for _ in range(3))
            ds, ae = np.full((len(hs), R_SIM), np.inf), np.full((len(hs), R_SIM), np.inf)
            for j, h in enumerate(hs):
                B = rt_matrix(p, h, t0_min, extra)
                if B is None:
                    continue
                B = B[:len(t0s)]
                G = A - B
                ed[j] = np.mean((G @ m_r) ** 2 + (G @ SIG_r * G).sum(1))
                bb = B @ m_r - mt
                ms[j] = np.mean(bb**2 + (B @ SIG_r * B).sum(1))
                cr[j] = np.mean(ba * bb + (A @ SIG_r * B).sum(1))
                rt = B @ Y_r
                ds[j] = ((refY - rt) ** 2).mean(axis=0)
                ae[j] = ((rt - mt[:, None]) ** 2).mean(axis=0)
            ED[p], MSE[p], CR[p], Dsim[p], ASE[p] = ed, ms, cr, ds, ae
        # (a) eksak
        rec = {"cfg": cfg, "h_ref": h_ref, "MSE_ref": mse_ref}
        for p in (0, 1):
            lab = LAB[p]
            jp, jt = int(np.argmin(ED[p])), int(np.argmin(MSE[p]))
            rec.update({f"h_{lab}_proxy": hs[jp], f"h_{lab}_truth": hs[jt],
                        f"ED_{lab}": ED[p][jp], f"MSE_{lab}": MSE[p][jt],
                        f"loss_{lab}": MSE[p][jp] / MSE[p][jt] - 1,
                        f"proxybias_{lab}": mse_ref - 2 * CR[p][jp],
                        f"proxybias_{lab}_at_truth_h": mse_ref - 2 * CR[p][jt]})
        rec["ratio_proxy"] = rec["ED_NW"] / rec["ED_LL"]
        rec["ratio_truth"] = rec["MSE_NW"] / rec["MSE_LL"]
        rec["win_proxy"] = "NW" if rec["ratio_proxy"] < 1 else "LL"
        rec["win_truth"] = "NW" if rec["ratio_truth"] < 1 else "LL"
        exact_rows.append(rec)
        # (b) Monte Carlo
        jh = {p: np.argmin(Dsim[p], axis=0) for p in (0, 1)}
        dsel = {p: Dsim[p][jh[p], np.arange(R_SIM)] for p in (0, 1)}
        asel = {p: ASE[p][jh[p], np.arange(R_SIM)] for p in (0, 1)}
        aor = {p: ASE[p].min(axis=0) for p in (0, 1)}
        win_p = dsel[0] < dsel[1]                     # True = NW
        win_r = asel[0] < asel[1]
        win_o = aor[0] < aor[1]
        ratio_sim = dsel[0] / dsel[1]
        mrec = {"cfg": cfg, "h_ref": h_ref, "P_NW_proxy": win_p.mean(), "P_NW_realized": win_r.mean(),
                "P_NW_oracle": win_o.mean(), "P_same_realized": (win_p == win_r).mean(),
                "P_same_oracle": (win_p == win_o).mean(),
                "P_same_exact_truth": (win_p == (rec["win_truth"] == "NW")).mean(),
                "ratio_p05": np.percentile(ratio_sim, 5), "ratio_p50": np.median(ratio_sim),
                "ratio_p95": np.percentile(ratio_sim, 95)}
        for p in (0, 1):
            lab = LAB[p]
            hsel = hs[jh[p]]
            mrec.update({f"h_{lab}_med": np.median(hsel), f"h_{lab}_q1": np.percentile(hsel, 25),
                         f"h_{lab}_q3": np.percentile(hsel, 75),
                         f"P_{lab}_at_max": np.mean(hsel == hs[np.isfinite(ED[p])].max()),
                         f"loss_{lab}_med": np.median(asel[p] / aor[p] - 1)})
        mc_rows.append(mrec)
ex = pd.DataFrame(exact_rows); ex.to_csv(OUT / "k3_exact.csv", index=False)
mc = pd.DataFrame(mc_rows); mc.to_csv(OUT / "k3_mc.csv", index=False)
print("(3a) EKSAK, inflasi: h terpilih oleh E D_e (proxy) vs MSE_e (kebenaran), rugi MSE(h_proxy)/MSE(h_truth) - 1,"
      "\n     bias proxy = MSE_ref - 2 Cross_e di h proxy dan di h kebenaran, rasio dan pemenang")
print(ex.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print("(3b) MONTE CARLO, inflasi: P(NW menang) menurut proxy, ASE di h terpilih (realized) dan oracle ASE;"
      "\n     P(pemenang proxy sama); kuantil rasio D_NW/D_LL; h terpilih (median, kuartil); rugi ASE median")
print(mc.to_string(index=False, float_format=f3))
d0 = pb[(pb.variant == "S0") & (pb.response == "inflation")].set_index("h_ref").ratio
for h_ref in H_REF:
    q = mc[(mc.cfg == "base") & (mc.h_ref == h_ref)].iloc[0]
    print(f"  inflasi h_ref = {h_ref}: rasio data {d0[h_ref]:.3f}, pita simulasi 5-95% "
          f"[{q.ratio_p05:.3f}, {q.ratio_p95:.3f}] -> di dalam: {q.ratio_p05 <= d0[h_ref] <= q.ratio_p95}")

# (3c) log CPI simulasi: reproduksi Panel B tanpa kebenaran
lc_rows = []
for h_ref in H_REF:
    hs, t0_min, extra = grid("base")
    t0s, A = ref_matrix(h_ref, t0_min)
    refY = A @ Y_l
    dsel, hsel = {}, {}
    for p in (0, 1):
        ds = np.full((len(hs), R_SIM), np.inf)
        for j, h in enumerate(hs):
            B = rt_matrix(p, h, t0_min, extra)
            if B is not None:
                ds[j] = ((refY - B[:len(t0s)] @ Y_l) ** 2).mean(axis=0)
        jh = np.argmin(ds, axis=0)
        dsel[p], hsel[p] = ds[jh, np.arange(R_SIM)], hs[jh]
    rs = dsel[0] / dsel[1]
    dl = pb[(pb.variant == "S0") & (pb.response == "log CPI") & (pb.h_ref == h_ref)].iloc[0]
    lc_rows.append({"h_ref": h_ref, "P_NW": np.mean(rs < 1), "ratio_p05": np.percentile(rs, 5),
                    "ratio_p50": np.median(rs), "ratio_p95": np.percentile(rs, 95), "ratio_data": dl.ratio,
                    "h_NW_med": np.median(hsel[0]), "P_NW_floor": np.mean(hsel[0] == 2),
                    "h_LL_med": np.median(hsel[1]), "P_LL_floor": np.mean(hsel[1] == 3)})
lc = pd.DataFrame(lc_rows)
print("(3c) LOG CPI SIMULASI (dibangun dari inflasi simulasi): P(NW menang), kuantil rasio, h terpilih, "
      "porsi di lantai (NW 2, LL 3)")
print(lc.to_string(index=False, float_format=f3))

# ---------- (4) ME-7: lantai dan batas atas ----------
fu_rows = []
for cfg in CFG:
    for (vname, resp), Y in SERIES.items():
        for h_ref in H_REF:
            fu_rows.append({"cfg": cfg, "variant": vname, "response": resp, "h_ref": h_ref,
                            **panel(Y, cfg, h_ref)})
fu = pd.DataFrame(fu_rows); fu.to_csv(OUT / "k3_floor_upper.csv", index=False)
print("\n(4) ME-7: base = h 2..48, t0 >= 48; floor_eq = h 1..48 (NW 1, LL 2 = Y_t0); base_t72 = h 2..48, t0 >= 72;"
      " upper72 = h 2..72, t0 >= 72; floor_eq72 = h 1..72, t0 >= 72")
print(fu[["cfg"] + COLS].to_string(index=False, float_format=f5))

# ---------- (5) DA-5: COVID dan titik pisah ----------
cov = lambda t: (t >= 121) & (t <= 144)
SPLITS = {"split120_all": (120, None, None),
          "split120_2020_21": (120, cov, None),
          "split120_2022_26": (120, lambda t: t >= 145, None),
          "split96": (96, None, None),
          "split144": (144, None, None),
          "insample_no_2020_21": (120, None, lambda t: ~cov(t))}
oos_rows = []
for sname, (split, tmask, imask) in SPLITS.items():
    for (vname, resp), Y in SERIES.items():
        for h_ref in H_REF:
            rec = panel(Y, "base", h_ref, split=split, test_mask=tmask, insample_mask=imask)
            oos_rows.append({"split": sname, "variant": vname, "response": resp, "h_ref": h_ref, **rec})
oos = pd.DataFrame(oos_rows); oos.to_csv(OUT / "k3_oos.csv", index=False)
print("\n(5) DA-5: pilih h pada t0 <= titik pisah, evaluasi pada sampel uji; insample_no_2020_21 = in-sample tanpa"
      " t0 121-144")
print(oos[["split", "variant", "response", "h_ref", "n_test", "h_NW_train", "h_LL_train", "ratio_oos",
           "pref_oos", "n_t0", "ratio", "pref"]].to_string(index=False, float_format=f3))

# ---------- (6) ringkasan ----------
print("\n(6) RINGKASAN")
for resp in ("log CPI", "inflation"):
    for vname in ("S0", "S2"):
        s = oos[(oos.variant == vname) & (oos.response == resp)]
        f = fu[(fu.variant == vname) & (fu.response == resp)]
        print(f"  {resp:9s} {vname}: rasio in-sample lantai disamakan {f[f.cfg == 'floor_eq'].ratio.min():.3f}-"
              f"{f[f.cfg == 'floor_eq'].ratio.max():.3f}; upper72 {f[f.cfg == 'upper72'].ratio.min():.3f}-"
              f"{f[f.cfg == 'upper72'].ratio.max():.3f} (h_LL {f[f.cfg == 'upper72'].h_LL.min()}-"
              f"{f[f.cfg == 'upper72'].h_LL.max()}); OOS 2020-21 {s[s.split == 'split120_2020_21'].ratio_oos.min():.3f}-"
              f"{s[s.split == 'split120_2020_21'].ratio_oos.max():.3f}; OOS 2022-26 "
              f"{s[s.split == 'split120_2022_26'].ratio_oos.min():.3f}-{s[s.split == 'split120_2022_26'].ratio_oos.max():.3f}; "
              f"pisah 96 {s[s.split == 'split96'].ratio_oos.min():.3f}-{s[s.split == 'split96'].ratio_oos.max():.3f}; "
              f"pisah 144 {s[s.split == 'split144'].ratio_oos.min():.3f}-{s[s.split == 'split144'].ratio_oos.max():.3f}; "
              f"sel dengan LL unggul: {int((s.pref_oos == 'LL').sum() + (f.pref == 'LL').sum() + (f.pref_oos == 'LL').sum())} "
              f"dari {len(s) + 2 * len(f)}")
print(f"\nselesai dalam {time.time() - t_start:.0f} s; [tersimpan] output/k3_*.csv, output/fig_k3_testB_curves.png")
