"""
21_testA_eiv.py
Revisi putaran 2 (Reviewer 2), ME-5: errors-in-variables dan data bersama di Test (A).

Test (A) meregresikan revisi Rev_e(t0) = m_final(t0) - m_rt,e(t0) pada regresor final
  NW: slope LL m'_H(t0),         benchmark delta_h (Pers. 25)
  LL: kelengkungan LQ m''_H(t0), benchmark gamma_h
dengan H = h di naskah R1 (Tabel 5). Revisi dan regresor linear dalam Y: Rev = d(t0)'Y dan
regresor = v(t0)'Y. Untuk Y = m + eps, Cov(eps) = Sigma, dan jalur lokal polinomial
(d'm = bench * v'm), dengan pemusatan atas t0:
  plim slope / bench = (S + A) / (S + B),
  S = Var_t0(v'm)                     sinyal,
  B = E Var_t0(v'eps)                 atenuasi, menarik rasio ke bawah 1,
  A = E Cov_t0(d'eps, v'eps) / bench  data bersama, menarik rasio ke atas 1.
Batas SNR -> 0 adalah A/B. Regresor pilot H > h mengecilkan A dan B relatif terhadap S
(kira-kira (h/H)^3 untuk slope dan (h/H)^5 untuk kelengkungan), tetapi menaikkan A/B; bila
regresor didominasi noise, rasio justru menjauh dari 1.

Bagian:
 (1) A/B eksak untuk empat struktur galat: iid; stasioner dengan autokorelasi variogram inflasi;
     terintegrasi dengan inkremen iid; terintegrasi dengan inkremen ber-autokorelasi variogram.
 (2) Data: slope/bench dengan regresor H = h (reproduksi Tabel 5), round(1.5h) dan 2h; CI HAC
     Bartlett dengan lag 2 max(h, H) - 2.
 (3) Prediksi dan koreksi momen memakai Sigma dari variogram (tanpa smoother pilot):
     rasio_pred = (S_hat + A_hat)/(S_hat + B_hat), S_hat = Var_t0(regresor) - B_hat;
     rasio_kor = (Cov - bench A_hat) / (bench (Var - B_hat)). Benchmark yang disesuaikan noise
     (ratio_null) = rasio_pred bila S_hat > 0, selain itu A/B; dicek apakah masuk CI HAC.
 (4) Simulasi dengan m diketahui pada SNR setara data: r = m_r + e, e ~ N(0, Sigma_r), m_r tren
     LL h = 36 dengan amplitudo dikalibrasi agar median SNR simulasi = SNR inflasi teramati;
     log CPI dibangun dari r persis seperti data. Rasio dengan regresor H = h, round(1.5h), 2h,
     dan dengan regresor tanpa noise (v_h'm).

Keluaran: output/k2_ab_exact.csv, output/k2_data.csv, output/k2_sim.csv
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"; OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 250)

df = pd.read_csv(OUT / "ihk_bps_2010_2026.csv")
n = len(df)
T = np.arange(1, n + 1, dtype=float)
SERIES = {"log CPI": df["log_IHK"].to_numpy(float), "inflation": df["infl_pct_mtm"].to_numpy(float)}
H_GRID = [9, 12, 18, 24]
T0_MIN = 48
R_SIM, SEED, H_TREND = 300, 20261010, 36
t_start = time.time()


def lp_rows(t0, Tv, h, p):
    """Baris (Z'WZ)^{-1} Z'W, Epanechnikov; baris j memberi koefisien beta_j = m^(j)(t0)/j!."""
    u = (Tv - t0) / h
    w = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
    Z = np.vander(Tv - t0, N=p + 1, increasing=True)
    ZtW = Z.T * w
    return np.linalg.inv(ZtW @ Z) @ ZtW


def bench(h, est):
    if est == "NW":                                    # delta_h, Pers. (25)
        k = np.arange(h); K = 1 - (k / h) ** 2
        return float((k * K).sum() / K.sum())
    t0 = n // 2; d = T - t0                            # gamma_h di titik interior
    return float(0.5 * ((lp_rows(t0, T, h, 1)[0] * d**2).sum()
                        - (lp_rows(t0, T[:t0], h, 1)[0] * d[:t0] ** 2).sum()))


def design(h, H):
    """Baris per t0 = 48..n-h: bobot revisi NW dan LL (bandwidth h), slope LL dan kelengkungan LQ (H)."""
    t0s = np.arange(T0_MIN, n - h + 1)
    N = len(t0s)
    Dnw, Dll, V, Q = (np.zeros((N, n)) for _ in range(4))
    for i, t0 in enumerate(t0s):
        fin = lp_rows(t0, T, h, 1)[0]
        Dnw[i] = fin; Dnw[i, :t0] -= lp_rows(t0, T[:t0], h, 0)[0]
        Dll[i] = fin; Dll[i, :t0] -= lp_rows(t0, T[:t0], h, 1)[0]
        V[i] = lp_rows(t0, T, H, 1)[1]
        Q[i] = 2 * lp_rows(t0, T, H, 2)[2]
    return t0s, {"NW": (Dnw, V), "LL": (Dll, Q)}


def noise_moments(Dm, Vm, Sig):
    """E Cov_t0(D eps, V eps) dan E Var_t0(V eps), pemusatan atas t0, pembagi N."""
    N = Dm.shape[0]
    Cn = np.eye(N) - 1.0 / N
    return np.trace(Cn @ Dm @ Sig @ Vm.T) / N, np.trace(Cn @ Vm @ Sig @ Vm.T) / N


def variogram_cov(y, kmax=36, plateau=(25, 36), lag_max=24):
    """Sama dengan 23_data_checks.py: gamma0 dari dataran V(k), korelasi ditaper Bartlett."""
    V = np.array([0.5 * np.mean((y[k:] - y[:-k]) ** 2) for k in range(1, kmax + 1)])
    g0 = V[plateau[0] - 1:plateau[1]].mean()
    rho = 1 - V[:lag_max] / g0
    return g0, rho, rho * (1 - np.arange(1, lag_max + 1) / (lag_max + 1))


def toeplitz(rho_tap, size):
    col = np.r_[1.0, rho_tap, np.zeros(size - 1 - len(rho_tap))]
    return col[np.abs(np.subtract.outer(np.arange(size), np.arange(size)))]


def hac_slope(x, y, L):
    """Sama dengan 10b_revision_diagnostics.py: OLS dengan intersep, SE Newey-West Bartlett lag L."""
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    g = X * (y - X @ b)[:, None]
    S = g.T @ g
    for l in range(1, L + 1):
        G = g[l:].T @ g[:-l]
        S += (1 - l / (L + 1)) * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    return b[1], float(np.sqrt((XtXi @ S @ XtXi)[1, 1]))


def col_slope(rev, reg):
    """Slope OLS per kolom (replikasi); reg boleh vektor (N,) atau matriks (N, R)."""
    if reg.ndim == 1:
        reg = reg[:, None]
    rc, gc = rev - rev.mean(0), reg - reg.mean(0)
    return (rc * gc).sum(0) / (gc * gc).sum(0)


# ---------- struktur galat dari data (tanpa smoother pilot) ----------
r = SERIES["inflation"]
u = np.diff(SERIES["log CPI"])
g0_r, rho_r, rtap_r = variogram_cov(r)
g0_u, rho_u, rtap_u = variogram_cov(u)
R_r, R_u = toeplitz(rtap_r, n), toeplitz(rtap_u, n - 1)
CUM = np.tril(np.ones((n, n - 1)), -1)          # eps_t = sum_{s=2}^{t} u_s
STRUCT = {"iid": np.eye(n), "stasioner, variogram r": R_r,
          "terintegrasi, inkremen iid": CUM @ CUM.T,
          "terintegrasi, inkremen variogram": CUM @ R_u @ CUM.T}
SIG_DATA = {"inflation": g0_r * R_r, "log CPI": CUM @ (g0_u * R_u) @ CUM.T}
print(f"variogram r: gamma0 = {g0_r:.4f}, rho1..3 = {rho_r[0]:+.3f} {rho_r[1]:+.3f} {rho_r[2]:+.3f}; "
      f"inkremen log CPI: gamma0 = {g0_u:.3e}, rho1..3 = {rho_u[0]:+.3f} {rho_u[1]:+.3f} {rho_u[2]:+.3f}")

DES = {}
for h in H_GRID:
    for H in sorted({h, int(round(1.5 * h)), 2 * h}):
        DES[(h, H)] = design(h, H)
BENCH = {(h, e): bench(h, e) for h in H_GRID for e in ("NW", "LL")}

# ---------- (1) A/B eksak ----------
ab_rows = []
for (h, H), (t0s, dd) in DES.items():
    for est, (Dm, Vm) in dd.items():
        b = BENCH[(h, est)]
        for sname, Sig in STRUCT.items():
            c, v = noise_moments(Dm, Vm, Sig)
            ab_rows.append({"h": h, "H": H, "est": est, "struct": sname, "A_over_B": c / (b * v),
                            "B_rel": np.nan})
ab = pd.DataFrame(ab_rows)
# B relatif terhadap H = h: seberapa jauh regresor pilot mengecilkan varians noise regresor
for (h, H), (t0s, dd) in DES.items():
    for est, (Dm, Vm) in dd.items():
        v_h = noise_moments(*DES[(h, h)][1][est], STRUCT["iid"])[1]
        v_H = noise_moments(Dm, Vm, STRUCT["iid"])[1]
        ab.loc[(ab.h == h) & (ab.H == H) & (ab.est == est), "B_rel"] = v_H / v_h
ab.to_csv(OUT / "k2_ab_exact.csv", index=False)
print("\n(1) BATAS SNR -> 0: A/B eksak (rata-rata atas t0 = 48..n-h). B_rel = B(H)/B(h), galat iid")
print(ab.pivot_table(index=["est", "h", "H", "B_rel"], columns="struct", values="A_over_B", sort=False)
      .to_string(float_format=lambda x: f"{x:.3f}"))

# ---------- (2)-(3) data ----------
data_rows = []
for sname, y in SERIES.items():
    Sig = SIG_DATA[sname]
    for (h, H), (t0s, dd) in DES.items():
        for est, (Dm, Vm) in dd.items():
            b = BENCH[(h, est)]
            rev, reg = Dm @ y, Vm @ y
            cov = np.mean((rev - rev.mean()) * (reg - reg.mean())); var = reg.var()
            slope, se = hac_slope(reg, rev, 2 * max(h, H) - 2)
            c, v = noise_moments(Dm, Vm, Sig)
            S_hat = var - v
            data_rows.append({"series": sname, "est": est, "h": h, "H": H, "n_t0": len(t0s), "bench": b,
                              "slope": slope, "ratio": slope / b,
                              "HAC_lo": (slope - 1.96 * se) / b, "HAC_hi": (slope + 1.96 * se) / b,
                              "corr": np.corrcoef(reg, rev)[0, 1], "noise_share": v / var,
                              "limit_AB": c / (b * v),
                              "ratio_pred": (S_hat + c / b) / var if S_hat > 0 else np.nan,
                              "ratio_corr": (cov - c) / (b * S_hat) if S_hat > 0 else np.nan})
dat = pd.DataFrame(data_rows)
# benchmark yang disesuaikan noise: prediksi momen bila S_hat > 0, selain itu batas noise murni A/B
dat["ratio_null"] = dat.ratio_pred.fillna(dat.limit_AB)
dat["null_in_HAC"] = (dat.HAC_lo <= dat.ratio_null) & (dat.ratio_null <= dat.HAC_hi)
dat.to_csv(OUT / "k2_data.csv", index=False)
f4 = lambda x: f"{x:.4f}"
print("\n(2) REPRODUKSI TABEL 5 (log CPI, NW, H = h): slope, delta_h, rasio, CI HAC 2h - 2, korelasi")
print(dat[(dat.series == "log CPI") & (dat.est == "NW") & (dat.H == dat.h)]
      [["h", "n_t0", "slope", "bench", "ratio", "HAC_lo", "HAC_hi", "corr"]].to_string(index=False, float_format=f4))
print("\n(3) DATA: rasio teramati, CI HAC, porsi noise regresor B/Var, batas A/B, rasio prediksi momen, rasio terkoreksi")
print(dat[["series", "est", "h", "H", "ratio", "HAC_lo", "HAC_hi", "corr", "noise_share", "limit_AB",
           "ratio_pred", "ratio_corr", "ratio_null", "null_in_HAC"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ---------- (4) simulasi dengan m diketahui ----------
rng = np.random.default_rng(SEED)
L12 = np.vstack([lp_rows(t, T, 12, 1)[0] for t in T])
LTR = np.vstack([lp_rows(t, T, H_TREND, 1)[0] for t in T])


def snr(Y):
    F = L12 @ Y
    return F.std(axis=0) / (Y - F).std(axis=0)


m_fit = LTR @ r
E = np.linalg.cholesky(g0_r * R_r) @ rng.standard_normal((n, R_SIM))
snr_obs = {k: float(snr(v[:, None])[0]) for k, v in SERIES.items()}
lo, hi = 0.0, 4.0
for _ in range(50):                              # kalibrasi amplitudo tren: median SNR = SNR teramati
    c_amp = 0.5 * (lo + hi)
    m_r = m_fit.mean() + c_amp * (m_fit - m_fit.mean())
    if np.median(snr(m_r[:, None] + E)) < snr_obs["inflation"]:
        lo = c_amp
    else:
        hi = c_amp
m_r = m_fit.mean() + c_amp * (m_fit - m_fit.mean())
Y_r = m_r[:, None] + E
m_log = np.log(100.0) + np.cumsum(np.log1p(m_r / 100.0))
Y_l = np.log(100.0) + np.cumsum(np.log1p(Y_r / 100.0), axis=0)
SIM = {"log CPI": (m_log, Y_l, CUM @ (g0_r * R_r[1:, 1:] / 1e4) @ CUM.T), "inflation": (m_r, Y_r, g0_r * R_r)}
print(f"\n(4) SIMULASI: amplitudo tren c = {c_amp:.3f} (tren LL h = {H_TREND}); SNR median simulasi "
      f"inflasi {np.median(snr(Y_r)):.3f} (data {snr_obs['inflation']:.3f}), log CPI {np.median(snr(Y_l)):.1f} "
      f"(data {snr_obs['log CPI']:.1f}); R = {R_SIM}")
sim_rows = []
for sname, (m_true, Ys, Sig) in SIM.items():
    for (h, H), (t0s, dd) in DES.items():
        for est, (Dm, Vm) in dd.items():
            b = BENCH[(h, est)]
            Vh = DES[(h, h)][1][est][1]
            ratio = col_slope(Dm @ Ys, Vm @ Ys) / b
            ratio_or = col_slope(Dm @ Ys, Vh @ m_true) / b
            dm, vm = Dm @ m_true, Vm @ m_true
            c, v = noise_moments(Dm, Vm, Sig)
            sig_cov = np.mean((dm - dm.mean()) * (vm - vm.mean())); S = vm.var()
            sim_rows.append({"series": sname, "est": est, "h": h, "H": H,
                             "ratio_mean": ratio.mean(), "ratio_p05": np.percentile(ratio, 5),
                             "ratio_p95": np.percentile(ratio, 95),
                             "pred_exact": (sig_cov + c) / (b * (S + v)),
                             "signal_only": sig_cov / (b * S), "noise_share": v / (S + v),
                             "oracle_mean": ratio_or.mean(), "oracle_p05": np.percentile(ratio_or, 5),
                             "oracle_p95": np.percentile(ratio_or, 95)})
sim = pd.DataFrame(sim_rows)
sim = sim.merge(dat[["series", "est", "h", "H", "ratio"]].rename(columns={"ratio": "ratio_data"}),
                on=["series", "est", "h", "H"])
sim["data_in_band"] = (sim.ratio_p05 <= sim.ratio_data) & (sim.ratio_data <= sim.ratio_p95)
sim.to_csv(OUT / "k2_sim.csv", index=False)
print("rasio = slope/bench; pred_exact = (Cov sinyal + A)/(S + B) dari m dan Sigma simulasi; signal_only = rasio"
      " tanpa noise; oracle = regresi revisi pada regresor tanpa noise v_h'm")
print(sim.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ---------- (5) ringkasan untuk naskah ----------
print("\n(5) RINGKASAN")
for sname in SERIES:
    for est in ("NW", "LL"):
        d0 = dat[(dat.series == sname) & (dat.est == est) & (dat.H == dat.h)]
        d2 = dat[(dat.series == sname) & (dat.est == est) & (dat.H == 2 * dat.h)]
        s0 = sim[(sim.series == sname) & (sim.est == est) & (sim.H == sim.h)]
        print(f"  {sname:9s} {est}: rasio H = h {d0.ratio.min():.3f}-{d0.ratio.max():.3f} | benchmark noise "
              f"{d0.ratio_null.min():.3f}-{d0.ratio_null.max():.3f}, di CI HAC {int(d0.null_in_HAC.sum())}/{len(d0)} "
              f"(CI memuat 1: {int(((d0.HAC_lo <= 1) & (d0.HAC_hi >= 1)).sum())}/{len(d0)}) | batas A/B "
              f"{d0.limit_AB.min():.3f}-{d0.limit_AB.max():.3f} | porsi noise {d0.noise_share.min():.2f}-"
              f"{d0.noise_share.max():.2f} (simulasi {s0.noise_share.min():.2f}-{s0.noise_share.max():.2f}) | "
              f"simulasi rata-rata {s0.ratio_mean.min():.3f}-{s0.ratio_mean.max():.3f}, data di pita 5-95%: "
              f"{int(s0.data_in_band.sum())}/{len(s0)}, sinyal saja {s0.signal_only.min():.3f}-"
              f"{s0.signal_only.max():.3f} | H = 2h: data {d2.ratio.min():.3f}-{d2.ratio.max():.3f}")
print(f"  pilot H > h, galat iid: B(H)/B(h) slope {ab[(ab.est == 'NW') & (ab.H != ab.h)].B_rel.min():.3f}-"
      f"{ab[(ab.est == 'NW') & (ab.H != ab.h)].B_rel.max():.3f}; A/B iid NW pada H = 2h "
      f"{ab[(ab.est == 'NW') & (ab.H == 2 * ab.h) & (ab.struct == 'iid')].A_over_B.min():.3f}-"
      f"{ab[(ab.est == 'NW') & (ab.H == 2 * ab.h) & (ab.struct == 'iid')].A_over_B.max():.3f}; "
      f"semua sel data di pita simulasi: {int(sim.data_in_band.sum())}/{len(sim)}")
print(f"\nselesai dalam {time.time() - t_start:.0f} s; [tersimpan] output/k2_ab_exact.csv, output/k2_data.csv, "
      f"output/k2_sim.csv")
