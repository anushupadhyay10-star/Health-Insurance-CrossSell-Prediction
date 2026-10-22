"""Step 1 - Exploratory data analysis.

Run:  python src/eda.py
Writes figures to outputs/figures and summary tables to outputs/tables.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {0: "#4C72B0", 1: "#DD8452"}

train = pd.read_csv(ROOT / "data" / "train.csv")
test = pd.read_csv(ROOT / "data" / "test.csv")
report = []


def log(title, obj=""):
    text = f"\n=== {title} ===\n{obj}"
    print(text)
    report.append(text)


log("Shapes", f"train={train.shape}, test={test.shape}")
log("Dtypes", train.dtypes.to_string())

# ---------------------------------------------------------------- target
vc = train["Response"].value_counts()
rate = train["Response"].mean()
log("Target distribution", f"{vc.to_dict()}\nPositive rate = {rate:.4%}  "
    f"(imbalance ratio {vc[0] / vc[1]:.1f}:1)\n"
    f"Majority-class accuracy baseline = {1 - rate:.4%}")
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x="Response", data=train, hue="Response", palette=PALETTE, legend=False, ax=ax)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
ax.set_title(f"Target distribution (positive rate {rate:.1%})")
fig.tight_layout()
fig.savefig(FIG / "01_target_distribution.png", dpi=130)
plt.close(fig)

# --------------------------------------------------------- missing values
miss = pd.DataFrame({"train_missing": train.isna().sum(), "test_missing": test.isna().sum().reindex(train.columns)})
log("Missing values", miss.to_string())
miss.to_csv(TAB / "missing_values.csv")

# ------------------------------------------------------------- duplicates
dup_all = train.duplicated().sum()
dup_noid = train.drop(columns="id").duplicated().sum()
log("Duplicates", f"Exact duplicate rows: {dup_all}\nDuplicates ignoring id: {dup_noid} "
    f"({dup_noid / len(train):.3%})\nDuplicate ids: {train['id'].duplicated().sum()}")
if dup_noid:
    d = train.drop(columns="id")
    mask = d.duplicated(keep=False)
    conflict = d[mask].groupby(list(d.columns.drop("Response")))["Response"].nunique()
    log("Duplicate groups with conflicting labels (ignoring id)", f"{(conflict > 1).sum()} of {len(conflict)} groups")

# ------------------------------------------------------ numeric summaries
num_cols = ["Age", "Annual_Premium", "Vintage"]
desc = train[num_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
desc["skew"] = train[num_cols].skew()
log("Numeric summary", desc.round(2).to_string())
desc.to_csv(TAB / "numeric_summary.csv")

# --------------------------------------------------------- premium outliers
p = train["Annual_Premium"]
q1, q3 = p.quantile([.25, .75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
out = (p > hi) | (p < lo)
log("Annual_Premium outliers (IQR rule)",
    f"Q1={q1:,.0f} Q3={q3:,.0f} IQR={iqr:,.0f} -> fences [{lo:,.0f}, {hi:,.0f}]\n"
    f"Outliers: {out.sum():,} ({out.mean():.2%}); max = {p.max():,.0f}; "
    f"min = {p.min():,.0f}\n"
    f"Premium == min value (possible floor/default): {(p == p.min()).sum():,}\n"
    f"Response rate outliers vs. non-outliers: {train.loc[out, 'Response'].mean():.4f} vs "
    f"{train.loc[~out, 'Response'].mean():.4f}\n"
    f"Skew raw={p.skew():.2f}, log1p={np.log1p(p).skew():.2f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(p, bins=80, ax=axes[0], color="#4C72B0")
axes[0].set_title("Annual_Premium (raw)")
sns.histplot(np.log1p(p), bins=80, ax=axes[1], color="#55A868")
axes[1].set_title("Annual_Premium (log1p)")
sns.boxplot(x="Response", y="Annual_Premium", data=train, hue="Response", palette=PALETTE, legend=False,
            ax=axes[2], showfliers=True, flierprops={"markersize": 1, "alpha": .2})
axes[2].set_title("Premium by response")
fig.tight_layout()
fig.savefig(FIG / "02_premium_outliers.png", dpi=130)
plt.close(fig)

# ----------------------------------------------- numeric distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, c in zip(axes, ["Age", "Vintage", "Annual_Premium"]):
    sns.kdeplot(data=train, x=c, hue="Response", common_norm=False, palette=PALETTE, ax=ax, fill=True, alpha=.3)
    ax.set_title(f"{c} by response")
fig.tight_layout()
fig.savefig(FIG / "03_numeric_by_response.png", dpi=130)
plt.close(fig)

# -------------------------------------------- categorical vs. response
cat_cols = ["Gender", "Driving_License", "Previously_Insured", "Vehicle_Age", "Vehicle_Damage"]
rows = []
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, c in zip(axes.ravel(), cat_cols):
    g = train.groupby(c)["Response"].agg(["mean", "count"]).reset_index()
    g["feature"] = c
    rows.append(g.rename(columns={c: "level"}))
    sns.barplot(x=c, y="mean", data=g, ax=ax, color="#4C72B0")
    ax.axhline(rate, ls="--", c="grey", lw=1)
    ax.set_ylabel("Response rate")
    ax.set_title(f"Response rate by {c}")
    for i, r in g.iterrows():
        ax.text(i, r["mean"], f"{r['mean']:.1%}\n(n={r['count']:,})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, g["mean"].max() * 1.3)
axes.ravel()[-1].axis("off")
fig.tight_layout()
fig.savefig(FIG / "04_categorical_response_rates.png", dpi=130)
plt.close(fig)
cat_tab = pd.concat(rows)[["feature", "level", "count", "mean"]]
log("Response rate by categorical level", cat_tab.round(4).to_string(index=False))
cat_tab.to_csv(TAB / "categorical_response_rates.csv", index=False)

# age bands
train["Age_band"] = pd.cut(train["Age"], [0, 25, 30, 35, 40, 45, 50, 60, 70, 100])
ab = train.groupby("Age_band", observed=True)["Response"].agg(["mean", "count"])
log("Response rate by age band", ab.round(4).to_string())
fig, ax = plt.subplots(figsize=(8, 4))
ab["mean"].plot(kind="bar", ax=ax, color="#4C72B0")
ax.axhline(rate, ls="--", c="grey")
ax.set_ylabel("Response rate")
ax.set_title("Response rate by age band")
fig.tight_layout()
fig.savefig(FIG / "05_age_band_response.png", dpi=130)
plt.close(fig)

# interaction: damage x previously insured
inter = train.pivot_table(index="Vehicle_Damage", columns="Previously_Insured", values="Response",
                          aggfunc=["mean", "count"])
log("Vehicle_Damage x Previously_Insured", inter.round(4).to_string())

# ------------------------------------------- high-cardinality columns
for c in ["Region_Code", "Policy_Sales_Channel"]:
    g = train.groupby(c)["Response"].agg(["mean", "count"]).sort_values("count", ascending=False)
    log(f"{c}: {g.shape[0]} unique values; top 10 by volume", g.head(10).round(4).to_string())
    big = g[g["count"] >= 500].sort_values("mean")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(big)), big["mean"], color="#4C72B0")
    ax.axhline(rate, ls="--", c="grey")
    ax.set_xticks(range(len(big)))
    ax.set_xticklabels([f"{int(i)}" for i in big.index], rotation=90, fontsize=7)
    ax.set_title(f"Response rate by {c} (levels with n>=500)")
    fig.tight_layout()
    fig.savefig(FIG / f"06_{c.lower()}_response.png", dpi=130)
    plt.close(fig)

# ------------------------------------------------------------ correlation
enc = train.drop(columns=["Age_band"]).copy()
enc["Gender"] = (enc["Gender"] == "Male").astype(int)
enc["Vehicle_Damage"] = (enc["Vehicle_Damage"] == "Yes").astype(int)
enc["Vehicle_Age"] = enc["Vehicle_Age"].map({"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2})
corr = enc.drop(columns="id").corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Correlation matrix (ordinal/binary encodings)")
fig.tight_layout()
fig.savefig(FIG / "07_correlation.png", dpi=130)
plt.close(fig)
log("Correlation with Response", corr["Response"].drop("Response").sort_values().round(3).to_string())

# ------------------------------------------------- train/test drift check
drift = []
for c in ["Age", "Annual_Premium", "Vintage", "Previously_Insured", "Driving_License"]:
    drift.append((c, train[c].mean(), test[c].mean()))
log("Train vs test means (drift check)", pd.DataFrame(drift, columns=["feature", "train", "test"]).round(3).to_string(index=False))

(TAB / "eda_report.txt").write_text("\n".join(report))
print("\nEDA complete. Figures in", FIG)
