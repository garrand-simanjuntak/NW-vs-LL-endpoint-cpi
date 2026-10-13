"""
sim_functions.py
Fungsi regresi target m(x) untuk studi simulasi, beserta turunan
analitik m'(x) dan m''(x) -- diperlukan untuk membandingkan bias
empirik dengan formula bias asimtotik.

Domain: x in [0, 1] (menyesuaikan desain x_t = t/n dari data IHK).

Pemilihan fungsi didasarkan pada teori:
  - Bias interior NW & LL sama-sama O(h^2), digerakkan m''(x).
  - Bias BATAS: NW = O(h) digerakkan m'(x); LL tetap O(h^2).
=> Untuk mengekspos efek batas, m'(x) di x=0 dan x=1 harus BESAR.
"""
import numpy as np


class TargetFunction:
    def __init__(self, name, m, dm, d2m, desc):
        self.name, self.m, self.dm, self.d2m, self.desc = name, m, dm, d2m, desc

    def __call__(self, x):
        return self.m(np.asarray(x, dtype=float))


# --- M1: sinus -- kemiringan besar di KEDUA batas, kurvatur bervariasi ---
M1 = TargetFunction(
    "M1_sin",
    m=lambda x: np.sin(2 * np.pi * x),
    dm=lambda x: 2 * np.pi * np.cos(2 * np.pi * x),
    d2m=lambda x: -(2 * np.pi) ** 2 * np.sin(2 * np.pi * x),
    desc="sin(2*pi*x) | m'(0)=m'(1)=2pi (besar) -> efek batas maksimal",
)

# --- M2: eksponensial -- monoton, kemiringan besar di batas kanan ---
M2 = TargetFunction(
    "M2_exp",
    m=lambda x: np.exp(2 * x),
    dm=lambda x: 2 * np.exp(2 * x),
    d2m=lambda x: 4 * np.exp(2 * x),
    desc="exp(2x) | monoton naik, m'(1)=2e^2 besar -> mirip tren IHK",
)

# --- M3: kuadratik -- m'' KONSTAN, m'(1)=0 (kontrol negatif) ---
M3 = TargetFunction(
    "M3_quad",
    m=lambda x: -2 * (x - 1.0) ** 2,
    dm=lambda x: -4 * (x - 1.0),
    d2m=lambda x: np.full_like(np.asarray(x, dtype=float), -4.0),
    desc="-2(x-1)^2 | m'(1)=0 -> KONTROL: efek batas NW harus hilang di x=1",
)

TARGETS = [M1, M2, M3]
