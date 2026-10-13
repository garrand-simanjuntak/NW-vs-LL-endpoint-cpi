# Code and data for "Nadaraya-Watson versus local linear trend estimation at the endpoint: a revision diagnostic for Indonesia's CPI"

Garrand Malona Simanjuntak and Dewi Retno Sari Saputro, Department of Mathematics, Universitas Sebelas Maret, Surakarta, Indonesia.

Release `v1.0` is the version submitted to the Journal of Mathematics and Scientific Computing with Applications.

## Contents

- `raw/`: the 17 annual BPS tables "Month-to-Month Inflation (M-to-M)", 2010 to 2026, as downloaded from https://www.bps.go.id.
- `output/`: files written by the scripts. The reconstructed national series is `output/ihk_bps_2010_2026.csv` (columns date, year, month, infl_pct_mtm, IHK_recon, log_IHK, t, x; n = 198, January 2010 to June 2026).
- `logs/`: console logs written by the scripts. Not every script has one.
- `lpr_core.py` (Nadaraya-Watson and local polynomial weights), `sim_functions.py` (simulation targets), `fig_style.py` (figure style).
- Numbered scripts `01_` to `25_`, run in numerical order. Scripts 20 to 25 also read files written by earlier scripts. The header of each revision script lists the files it writes.

## Manuscript elements produced by the revision scripts

- Table 1: `18_table1_targets.py`
- Figure 3 and the SNR bands of Section 3.4: `14_snr_sweep.py` (the image file is named `fig4_snr_crossover.png`)
- Tables 4a and 4b, plug-in estimation of rho: `20_mcv_plugin.py`
- Noise-adjusted benchmark of Test (A), Table 5: `21_testA_eiv.py`
- Test (B) sensitivity, simulation with known trend, out-of-sample splits: `22_testB_sensitivity.py`
- Unit-root tests, linking months, coverage regimes, variogram: `23_data_checks.py`
- Table 7, Figure 5 (Graphviz source) and the policy paragraph of Section 3.7: `24_worked_example.py`
- Local cubic weights in Section 3.8: `25_degree_tradeoff.py`
- Figures 1, 2 and 4: `fig1_equivalent_weights.py`, `fig2_mise_curves.py`, `fig4_revision_diagnostics.py`

## Environment

Python 3.13 with the packages in `requirements.txt` (revision) or `requirements_R1.txt` (original submission). Install with `pip install -r requirements.txt`. Scripts that simulate fix their random seeds, so their logs are reproduced exactly.

## Licences

Code: MIT License (see `LICENSE`). Derived data in `output/`: CC BY 4.0. The BPS tables in `raw/` are redistributed unchanged under the BPS terms of use; cite them as BPS-Statistics Indonesia, "Month-to-Month Inflation (M-to-M)", https://www.bps.go.id.
