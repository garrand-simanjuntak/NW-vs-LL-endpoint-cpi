"""
18_table1_targets.py
Tabel 1 versi revisi (Reviewer 1, komentar Tabel 1): fungsi target simulasi
dengan m'(1) DAN m''(1), beserta perannya dalam desain.

Kolom m''(1) menjelaskan orde bias LL di x = 1. Untuk sin(2 pi x), m''(1) = 0,
sehingga suku utama O(h^2) bias LL lenyap dan biasnya turun ke O(h^3); inilah
sebab orde empirik LL = 3.163 pada fungsi tersebut.

Nilai eksak diambil dari turunan analitik di sim_functions.py, lalu dicek
silang dengan beda hingga terpusat (step 1e-4). Skrip berhenti dengan error
bila label eksak di tabel tidak cocok dengan nilai numeriknya.

Keluaran:
  output/table1_targets.csv   (angka)
  output/table1_targets.txt   (baris dipisah TAB: salin ke Word, lalu
                               Insert > Table > Convert Text to Table)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sim_functions import M1, M2, M3

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"; OUT_DIR.mkdir(exist_ok=True)

# (fungsi, label fungsi, label eksak m'(1), label eksak m''(1), peran)
ROWS = [
    (M1, "m(x) = sin(2πx)", "2π", "0",
     "Large slope at both boundaries; m″(1) = 0, so the leading LL boundary "
     "term vanishes and the LL bias falls to O(h³)"),
    (M2, "m(x) = e^(2x)", "2e²", "4e²",
     "Principal function; monotone increasing, resembling the CPI trend"),
    (M3, "m(x) = −2(x − 1)²", "0", "−4",
     "Negative control (falsification test)"),
]
EXACT = {"2π": 2 * np.pi, "0": 0.0, "2e²": 2 * np.e**2, "4e²": 4 * np.e**2, "−4": -4.0}


def fd1(f, x, s=1e-4):
    return (f(x + s) - f(x - s)) / (2 * s)


def fd2(f, x, s=1e-4):
    return (f(x + s) - 2 * f(x) + f(x - s)) / s**2


recs = []
for tf, lab, d1_lab, d2_lab, role in ROWS:
    d1 = float(tf.dm(1.0)); d2 = float(np.asarray(tf.d2m(1.0)))
    for name, val, ref, fd in [("m'(1)", d1, EXACT[d1_lab], fd1(tf, 1.0)),
                               ("m''(1)", d2, EXACT[d2_lab], fd2(tf, 1.0))]:
        assert abs(val - ref) < 1e-9, f"{tf.name}: {name} analitik {val} != label {ref}"
        assert abs(val - fd) < 1e-3 * max(1.0, abs(val)), f"{tf.name}: {name} beda hingga {fd}"
    recs.append({"Target function": lab, "m'(1)": d1_lab, "m''(1)": d2_lab,
                 "m'(1) numeric": d1, "m''(1) numeric": d2, "Role": role})

tab = pd.DataFrame(recs)
tab.to_csv(OUT_DIR / "table1_targets.csv", index=False, encoding="utf-8-sig")

header = ["Target function", "m′(1)", "m″(1)", "Role"]
lines = ["\t".join(header)] + ["\t".join([r["Target function"], r["m'(1)"], r["m''(1)"], r["Role"]])
                               for r in recs]
(OUT_DIR / "table1_targets.txt").write_text("\n".join(lines), encoding="utf-8")

print("Table 1. Simulation Target Functions and Their Roles in the Research Design\n")
print(tab[["Target function", "m'(1)", "m''(1)", "m'(1) numeric", "m''(1) numeric"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nSemua label eksak cocok dengan turunan analitik dan beda hingga.")
print(f"[tersimpan] {OUT_DIR / 'table1_targets.csv'}")
print(f"[tersimpan] {OUT_DIR / 'table1_targets.txt'}  (tempel ke Word -> Convert Text to Table, pemisah Tab)")
