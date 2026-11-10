# ENSO as an early warning of sugarcane crush and residue supply in Maharashtra

Code and data for:

> Chougule, N. K., Ambardekar, V., Pathare, A., Pawar, S., Waghmare, A., & Bhamare, P.
> *Pre-monsoon ENSO state as an early warning of sugarcane crush and residue supply in Maharashtra, India.*
> Submitted to *Climate Risk Management*.

The script reproduces every statistic in the paper and draws Figs. 1–4.

## Quick start

```bash
pip install -r requirements.txt
python crm_analysis.py            # reads ./data, writes ./figures
```

The script runs in a few seconds. Its printed output should match `expected_output.txt`.

## Data

| File | Contents | Source |
|---|---|---|
| `data/maharashtra_cane_crush.csv` | Cane crushed in Maharashtra, 2015–16 to 2024–25 (Mt) | Vasantdada Sugar Institute statistics for 2021–22 and 2022–23. The other seasons are season totals reported in the trade and financial press; the source of each row is given in the file. |
| `data/oni.ascii.txt` | Oceanic Niño Index (ERSSTv5), 3-month running means | NOAA Climate Prediction Center, <https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt> |

All inputs are public, state-level aggregates. No factory-level or confidential data are used.

## What the script computes

- **April–May ONI per season.** This is the mean of the MAM and AMJ values for the year crushing starts. ENSO phase is assigned with a ±0.5 °C threshold.
- **Correlation and fit.** Pearson, Spearman and OLS fit of cane crushed on April–May ONI.
- **Checks on the fit:**
  - permutation test (20,000 permutations)
  - bootstrap confidence interval for r (5,000 resamples)
  - refit without 2016–17
  - refit with a linear trend
- **Sensitivity to the ONI window:** monsoon-season ONI, previous-year ONI, and a two-year mean.
- **Cross-validation:** leave-one-out comparison of the ONI regression against the historical mean, reporting MAPE, median APE and the number of seasons won.
- **Early warning:** an El Niño spring treated as a warning of below-average crush (contingency table and Fisher's exact test).
- **Residues:** bagasse (0.30 × cane) and press mud (0.035 × cane).
- **Figures:** four figures saved as 600 dpi TIFF and 300 dpi PNG.

Random seeds are fixed, so results are identical on every run.

## Notes

- The 2024–25 crush figure (84.33 Mt) was reported on 23 March 2025, near the end of the season. It is treated as the season total.
- An AI assistant (Claude, Anthropic) helped write this code. The authors reviewed it and checked its output against the paper, as stated in the manuscript.

## License

Code: MIT (see `LICENSE`). The data files are compiled from the public sources listed above; cite those sources when reusing them.
