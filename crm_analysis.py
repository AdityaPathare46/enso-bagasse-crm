"""
Reproduces every statistic and figure in:

  Chougule, N. K., Ambardekar, V., Pathare, A., Pawar, S., Waghmare, A. & Bhamare, P.
  "Pre-monsoon ENSO state as an early warning of sugarcane crush and residue supply
  in Maharashtra, India" (submitted to Climate Risk Management).

Inputs (in ./data):
  maharashtra_cane_crush.csv  state-level cane crushed per season, 2015-16 to 2024-25
  oni.ascii.txt               NOAA CPC Oceanic Nino Index (ERSSTv5), 3-month running means,
                              from https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt

"April-May ONI" for a season = mean of the MAM and AMJ values of the calendar year in
which crushing starts. Bagasse and press mud are 0.30 and 0.035 x cane crushed.

Usage:
  python crm_analysis.py [data_dir] [output_dir]      (defaults: data, figures)

Writes Fig1-Fig4 as 600 dpi TIFF and 300 dpi PNG, and prints all statistics.
"""
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
ONI_FILE = os.path.join(DATA, "oni.ascii.txt")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
MM = 1 / 25.4
FULL_W = 174 * MM   # Springer single-column (full) width

# --------------------------------------------------------------------------
# Data
# key = calendar year in which the season ends (2016 -> 2015-16 season)
with open(os.path.join(DATA, "maharashtra_cane_crush.csv"), newline="") as fh:
    cane_mt = {int(r["season_end_year"]): float(r["cane_crushed_mt"]) for r in csv.DictReader(fh)}

oni = {}
for seas, yr, _, anom in re.findall(r"(\w{3})\s+(\d{4})\s+([\d.]+)\s+(-?[\d.]+)",
                                    open(ONI_FILE).read()):
    oni[(seas, int(yr))] = float(anom)


def apr_may(y):
    return (oni[("MAM", y)] + oni[("AMJ", y)]) / 2


def jja_jas(y):
    return (oni[("JJA", y)] + oni[("JAS", y)]) / 2


ends = sorted(cane_mt)
labels = [f"{y-1}–{str(y)[2:]}" for y in ends]
c = np.array([cane_mt[y] for y in ends])
x = np.array([apr_may(y - 1) for y in ends])


def phase(v):
    return "El Niño" if v >= 0.5 else ("La Niña" if v <= -0.5 else "Neutral")


ph = [phase(v) for v in x]

# --------------------------------------------------------------------------
# Statistics
print("Season  cane(Mt)  AprMayONI  phase  bagasse(Mt)  pressmud(Mt)")
for l, cv, xv, p in zip(labels, c, x, ph):
    print(f"{l}  {cv:7.2f}  {xv:6.3f}  {p:8s}  {0.30*cv:6.2f}  {0.035*cv:6.3f}")

lr = stats.linregress(x, c)
print(f"\nOLS: slope={lr.slope:.2f} (SE {lr.stderr:.2f}) Mt per degC, "
      f"intercept={lr.intercept:.2f}, r={lr.rvalue:.3f}, R2={lr.rvalue**2:.3f}, p={lr.pvalue:.4f}")
print(f"  bagasse slope = {0.30*lr.slope:.2f} Mt/degC; press mud slope = {0.035*lr.slope:.2f} Mt/degC")

rho = stats.spearmanr(x, c)
print(f"Spearman rho={rho.statistic:.3f}, p={rho.pvalue:.4f}")

rng = np.random.default_rng(20260916)
r0 = np.corrcoef(x, c)[0, 1]
perm = np.array([np.corrcoef(rng.permutation(x), c)[0, 1] for _ in range(20000)])
print(f"Permutation p (two-sided, 20000) = {np.mean(np.abs(perm) >= abs(r0)):.4f}")

rng = np.random.default_rng(20260916)
boot = []
for _ in range(5000):
    s = rng.choice(10, 10, replace=True)
    if len(set(x[s])) > 1:
        boot.append(np.corrcoef(x[s], c[s])[0, 1])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"Bootstrap 95% CI r = [{lo:.2f}, {hi:.2f}]")

m = np.arange(10) != 1
r_drop = stats.pearsonr(x[m], c[m])
print(f"Without 2016-17: r={r_drop.statistic:.3f}, p={r_drop.pvalue:.4f}")

t = np.arange(10)
X = np.c_[np.ones(10), x, t]
b, *_ = np.linalg.lstsq(X, c, rcond=None)
res = c - X @ b
cov = (res @ res / 7) * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov))
pv = 2 * stats.t.sf(np.abs(b / se), 7)
print(f"ONI + linear trend: ONI coef={b[1]:.2f} (p={pv[1]:.3f}), trend={b[2]:.2f} Mt/yr (p={pv[2]:.3f})")

for name, f in [("JJA/JAS ONI, same year", lambda y: jja_jas(y - 1)),
                ("Apr-May ONI, one year earlier", lambda y: apr_may(y - 2)),
                ("Mean of Apr-May ONI, current and previous year",
                 lambda y: (apr_may(y - 1) + apr_may(y - 2)) / 2)]:
    xx = np.array([f(y) for y in ends])
    rr = stats.pearsonr(xx, c)
    print(f"Sensitivity, {name}: r={rr.statistic:.3f}, p={rr.pvalue:.3f}")

# Categorical early warning: El Nino spring (ONI >= 0.5) vs crush below the decade mean
below = c < c.mean()
warn = x >= 0.5
hits = int((warn & below).sum()); fa = int((warn & ~below).sum())
miss = int((~warn & below).sum()); cn = int((~warn & ~below).sum())
fisher = stats.fisher_exact([[hits, fa], [miss, cn]])
print(f"\nEarly warning (decade mean {c.mean():.1f} Mt): hits={hits}, false alarms={fa}, "
      f"misses={miss}, correct negatives={cn}; Fisher exact p={fisher.pvalue:.3f}")
print(f"  mean crush after El Nino springs {c[warn].mean():.1f} Mt vs others {c[~warn].mean():.1f} Mt "
      f"({100*(c[warn].mean()/c[~warn].mean()-1):.0f}%)")

pred_reg, pred_base = np.zeros(10), np.zeros(10)
for i in range(10):
    k = np.arange(10) != i
    s = stats.linregress(x[k], c[k])
    pred_reg[i] = s.slope * x[i] + s.intercept
    pred_base[i] = c[k].mean()
ape_r = np.abs(pred_reg - c) / c
ape_b = np.abs(pred_base - c) / c
print(f"\nLOOCV MAPE: baseline {100*ape_b.mean():.1f}%, regression {100*ape_r.mean():.1f}%, "
      f"relative reduction {100*(ape_b.mean()-ape_r.mean())/ape_b.mean():.1f}%")
print(f"LOOCV median APE: baseline {100*np.median(ape_b):.1f}%, regression {100*np.median(ape_r):.1f}%")
print(f"Regression better in {(ape_r < ape_b).sum()} of 10 seasons")
for l, cv, pr, pb in zip(labels, c, pred_reg, pred_base):
    print(f"  {l}: observed {cv:.1f}, regression {pr:.1f}, baseline {pb:.1f}")

# --------------------------------------------------------------------------
# Figures
STYLE = {"El Niño": dict(color="#b03a2e", hatch="////", marker="^"),
         "Neutral": dict(color="#7f8c8d", hatch="", marker="o"),
         "La Niña": dict(color="#2471a3", hatch="....", marker="s")}


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    plt.close(fig)


# Fig 1: crush series by ENSO phase
fig, ax = plt.subplots(figsize=(FULL_W, 70 * MM))
for i, (l, cv, p) in enumerate(zip(labels, c, ph)):
    st = STYLE[p]
    ax.bar(i, cv, color=st["color"], hatch=st["hatch"], edgecolor="black", linewidth=0.5, width=0.7)
    ax.text(i, cv + 2, f"{cv:.1f}", ha="center", fontsize=8)
ax.set_xticks(range(10), labels, fontsize=8)
ax.set_ylabel("Cane crushed (Mt)")
ax.set_xlabel("Crushing season")
ax.set_ylim(0, 145)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[Patch(facecolor=STYLE[p]["color"], hatch=STYLE[p]["hatch"], edgecolor="black",
                         linewidth=0.5, label=p) for p in STYLE],
          loc="upper left", frameon=False, title="April–May ONI phase", title_fontsize=8, fontsize=8)
fig.tight_layout()
save(fig, "Fig1")

# Fig 2: scatter + OLS fit with 95% confidence band
fig, ax = plt.subplots(figsize=(120 * MM, 85 * MM))
xs = np.linspace(-1.0, 1.05, 100)
n = 10
sxx = ((x - x.mean()) ** 2).sum()
s_res = np.sqrt(((c - (lr.slope * x + lr.intercept)) ** 2).sum() / (n - 2))
band = stats.t.ppf(0.975, n - 2) * s_res * np.sqrt(1 / n + (xs - x.mean()) ** 2 / sxx)
yfit = lr.slope * xs + lr.intercept
ax.fill_between(xs, yfit - band, yfit + band, color="0.85", lw=0)
ax.plot(xs, yfit, "k-", lw=0.9)
for p in STYLE:
    k = [i for i in range(10) if ph[i] == p]
    ax.scatter(x[k], c[k], marker=STYLE[p]["marker"], s=40, color=STYLE[p]["color"],
               edgecolor="black", linewidth=0.5, label=p, zorder=3)
offs = {"2020–21": (-14, -12), "2023–24": (5, 5), "2017–18": (5, -11), "2019–20": (-36, -3)}
for l, xv, cv in zip(labels, x, c):
    ax.annotate(l, (xv, cv), textcoords="offset points", xytext=offs.get(l, (5, 4)), fontsize=7)
ax.set_xlabel("April–May ONI (°C)")
ax.set_ylabel("Cane crushed (Mt)")
ax.text(0.98, 0.97, f"r = −{abs(lr.rvalue):.2f}, R² = {lr.rvalue**2:.2f}\np = {lr.pvalue:.3f}, n = 10",
        transform=ax.transAxes, ha="right", va="top", fontsize=8)
ax.legend(loc="lower left", frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "Fig2")

# Fig 3: (a) LOOCV predictions per season, (b) MAPE
fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 72 * MM), gridspec_kw={"width_ratios": [2.6, 1]})
ii = np.arange(10)
a1.plot(ii, c, "k-o", ms=4, lw=1, label="Observed")
a1.plot(ii, pred_reg, "--^", color="#1e8449", ms=4, lw=1, label="ONI regression (LOOCV)")
a1.plot(ii, pred_base, ":s", color="0.45", ms=4, lw=1, label="Historical mean (LOOCV)")
a1.set_xticks(ii, labels, rotation=45, ha="right", fontsize=7)
a1.set_ylabel("Cane crushed (Mt)")
a1.legend(frameon=False, fontsize=7, loc="upper left")
a1.text(-0.12, 1.02, "a", transform=a1.transAxes, fontweight="bold", fontsize=10)
vals = [100 * ape_b.mean(), 100 * ape_r.mean()]
bars = a2.bar([0, 1], vals, color=["0.75", "#1e8449"], hatch=["", "////"], edgecolor="black", linewidth=0.5, width=0.6)
for bb, v in zip(bars, vals):
    a2.text(bb.get_x() + bb.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
a2.set_xticks([0, 1], ["Historical\nmean", "ONI\nregression"], fontsize=8)
a2.set_ylabel("LOOCV MAPE (%)")
a2.set_ylim(0, 40)
a2.text(-0.35, 1.02, "b", transform=a2.transAxes, fontweight="bold", fontsize=10)
for a in (a1, a2):
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "Fig3")

# Fig 4: proposed framework (schematic drawn in code)
fig, ax = plt.subplots(figsize=(FULL_W, 92 * MM))
ax.set_xlim(0, 174)
ax.set_ylim(15, 105)
ax.axis("off")


def box(x0, y0, w, h, text, fc, fs=7, bold_first=True, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=fc, ec="black", lw=0.6, ls=ls))
    lines = text.split("\n")
    ax.text(x0 + w / 2, y0 + h / 2, "\n".join(lines), ha="center", va="center", fontsize=fs,
            linespacing=1.25)


def arrow(x0, y0, x1, y1, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black", ls=ls, shrinkA=0, shrinkB=0))


inputs = [("Meteorology\nIMD rainfall, temperature;\ndry-spell statistics; ONI", 82),
          ("Remote sensing\nMODIS / Sentinel-2\nNDVI, EVI", 61),
          ("Factory records\ncrush, juice quality,\nclarification, imbibition", 40),
          ("Crop and soil\nvariety, plant/ratoon,\ncrop age, soil moisture", 19)]
for txt, y in inputs:
    box(2, y, 38, 15, txt, "#eaf2f8")
    arrow(41, y + 7.5, 50, 52)
box(51, 40, 26, 24, "Pre-processing\nKNN imputation,\nIsolation Forest,\nalignment, VIF/PCA", "#fdf2e9")
branches = [("Random Forest\ndaily bagasse", 74), ("XGBoost\ndaily press mud", 50), ("LSTM\n7-day ahead", 26)]
for txt, y in branches:
    arrow(78, 52, 86, y + 6)
    box(87, y, 26, 12, txt, "#e9f7ef")
    arrow(114, y + 6, 121, 52)
box(122, 41, 22, 22, "Ridge stacking\nmeta-learner;\nMC-dropout\nintervals", "#e9f7ef")
arrow(145, 52, 150, 52)
box(151, 34, 21, 36, "Forecasts\nbagasse and\npress mud\n(t/day, t/week)\nwith 90%\nintervals", "#fef9e7")
ax.annotate("", xy=(100, 25.6), xytext=(161, 33),
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black", ls="--",
                            connectionstyle="angle,angleA=-90,angleB=0,rad=0"))
ax.text(150, 21, "observed vs forecast\n→ retraining", fontsize=6.5, ha="center", va="top")
# evaluated component
ax.add_patch(FancyBboxPatch((0.5, 80.5), 41, 23, boxstyle="round,pad=0.3,rounding_size=2",
                            fc="none", ec="#b03a2e", lw=1.0, ls="--"))
ax.text(44, 99, "Tested in this study (ONI only): April–May ONI → state-level seasonal cane crush\n"
        "→ bagasse and press mud via fixed ratios. All other components are proposed.",
        ha="left", va="center", fontsize=6.5, color="#b03a2e")
fig.tight_layout(pad=0.2)
save(fig, "Fig4")
print("\nFigures written to", OUT)
