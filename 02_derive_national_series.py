"""
02_derive_national_series.py
Menurunkan seri IHK nasional kontinu dari file raw merged (01_merge_raw.py).

Metode: rekonstruksi indeks dengan mengumulasikan inflasi m-to-m nasional,
        anchor Januari 2010 = 100. Ini kebal terhadap pergantian tahun dasar
        BPS (2007=100 -> 2012=100 -> 2018=100 -> 2022=100) karena yang
        dikumulasikan adalah LAJU PERUBAHAN, bukan level indeks.
"""
from pathlib import Path

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"

raw = pd.read_csv(OUTPUT_DIR / "bps_ihk_raw_merged_2010_2026.csv")

nat = raw[raw.region == "INDONESIA"].sort_values(["year", "month"]).reset_index(drop=True)
nat["date"] = pd.to_datetime(dict(year=nat.year, month=nat.month, day=1))

# validasi tidak ada bulan bolong
expected = pd.date_range(nat.date.min(), nat.date.max(), freq="MS")
assert len(nat) == len(expected), "Ada bulan yang hilang - cek data mentah."

r = nat.infl_pct_mtm / 100.0
nat["IHK_recon"] = 100.0 * np.cumprod(1.0 + r)
nat["log_IHK"] = np.log(nat.IHK_recon)
nat["t"] = np.arange(1, len(nat) + 1)
nat["x"] = nat.t / len(nat)

out = nat[["date", "year", "month", "infl_pct_mtm", "IHK_recon", "log_IHK", "t", "x"]]
out_path = OUTPUT_DIR / "ihk_bps_2010_2026.csv"
out.to_csv(out_path, index=False)

print(f"n = {len(out)} | {out.date.min().date()} -> {out.date.max().date()}")
print(f"File output         : {out_path}")
print(f"Link                : {out_path.as_uri()}")
