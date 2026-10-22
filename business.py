"""Step 5 - Turn predicted probabilities into a campaign decision.

Customers are ranked by predicted response probability, a contact budget is chosen, and
campaign profit is estimated under explicit, editable assumptions (see ASSUMPTIONS).

Honest evaluation: the contact share is chosen on the VALIDATION split and the resulting
profit is then measured on the untouched TEST split, using the real labels.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import get_models  # noqa: E402
from pipeline import ROOT, TARGET, load_data, split_xy  # noqa: E402

OUT, FIG, TAB = ROOT / "outputs", ROOT / "outputs" / "figures", ROOT / "outputs" / "tables"
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------ ASSUMPTIONS
ASSUMPTIONS = {
    "contact_cost": 300.0,        # cost of one outreach (call-centre time + incentive), same currency as premium
    "margin_rate": 0.10,          # first-year contribution margin = 10 % of the customer's Annual_Premium
    "campaign_size": 127_037,     # customers in test.csv that would actually be contacted/ranked
    "budget_shares": [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 1.00],
}
C, M = ASSUMPTIONS["contact_cost"], ASSUMPTIONS["margin_rate"]


def profit_curve(p, y, premium, cost=C, margin=M):
    """Rank by p (desc). Return arrays over k=1..n of cumulative conversions, revenue, cost, profit."""
    order = np.argsort(-p, kind="stable")
    y_s, prem_s = y[order], premium[order]
    k = np.arange(1, len(p) + 1)
    conv = np.cumsum(y_s)
    revenue = np.cumsum(y_s * prem_s * margin)
    spend = k * cost
    return k, conv, revenue, spend, revenue - spend


def main():
    best = json.load(open(OUT / "best_model.json"))["best_model"]
    train_full, test_unl = load_data()
    prem = train_full.set_index("id")["Annual_Premium"]

    val = pd.read_csv(OUT / "valid_predictions.csv")
    te = pd.read_csv(OUT / "test_predictions.csv")
    for d in (val, te):
        d["premium"] = d["id"].map(prem).values

    # ---- 1. choose contact share on validation (profit-maximising), evaluate on test
    kv, _, _, _, prof_v = profit_curve(val[best].values, val[TARGET].values, val["premium"].values)
    best_share = (np.argmax(prof_v) + 1) / len(val)
    print(f"Best model: {best} | profit-maximising contact share on validation: {best_share:.1%}")

    p_t, y_t, prem_t = te[best].values, te[TARGET].values, te["premium"].values
    n_t = len(te)
    k, conv, rev, spend, prof = profit_curve(p_t, y_t, prem_t)
    scale = ASSUMPTIONS["campaign_size"] / n_t
    total_pos = y_t.sum()

    def scenario(label, kk):
        kk = int(kk)
        return {
            "scenario": label, "contact_share": kk / n_t, "customers_contacted": round(kk * scale),
            "conversions": round(conv[kk - 1] * scale), "recall(share of buyers reached)": conv[kk - 1] / total_pos,
            "precision(hit rate)": conv[kk - 1] / kk, "lift": (conv[kk - 1] / kk) / y_t.mean(),
            "revenue": rev[kk - 1] * scale, "contact_cost": spend[kk - 1] * scale, "profit": prof[kk - 1] * scale,
            "ROI": prof[kk - 1] / spend[kk - 1],
        }

    rows = []
    for s in ASSUMPTIONS["budget_shares"]:
        rows.append(scenario(f"Top {s:.0%} by model" if s < 1 else "Contact everyone", max(1, round(s * n_t))))
    rows.append(scenario(f"Profit-optimal share chosen on validation ({best_share:.0%})", max(1, round(best_share * n_t))))
    # random targeting benchmark at the same share = expected values
    share = best_share
    rnd_conv = y_t.mean() * share * n_t
    rnd_rev = (y_t * prem_t).mean() * M * share * n_t
    rnd_cost = share * n_t * C
    rows.append({"scenario": f"Random {share:.0%} (benchmark)", "contact_share": share,
                 "customers_contacted": round(share * n_t * scale), "conversions": round(rnd_conv * scale),
                 "recall(share of buyers reached)": share, "precision(hit rate)": y_t.mean(), "lift": 1.0,
                 "revenue": rnd_rev * scale, "contact_cost": rnd_cost * scale,
                 "profit": (rnd_rev - rnd_cost) * scale, "ROI": (rnd_rev - rnd_cost) / rnd_cost})
    summary = pd.DataFrame(rows)
    summary.to_csv(TAB / "campaign_scenarios.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print(f"\nAssumptions: {ASSUMPTIONS}\nBreak-even response probability = cost / (margin x premium) "
          f"~ {C / (M * prem_t.mean()):.1%} at mean premium\n")
    print(summary.to_string(index=False))

    # ---- 2. expected vs. realised profit (calibration sanity check)
    sel = np.argsort(-p_t, kind="stable")[: max(1, round(best_share * n_t))]
    exp_profit = (p_t[sel] * prem_t[sel] * M - C).sum() * scale
    real_profit = (y_t[sel] * prem_t[sel] * M - C).sum() * scale
    print(f"\nExpected profit from probabilities {exp_profit:,.0f} vs realised on test labels {real_profit:,.0f}")

    # ---- 3. decile table
    dec = pd.DataFrame({"p": p_t, "y": y_t, "prem": prem_t}).sort_values("p", ascending=False).reset_index(drop=True)
    dec["decile"] = (np.arange(n_t) * 10 // n_t) + 1
    dtab = dec.groupby("decile").agg(customers=("y", "size"), min_prob=("p", "min"), max_prob=("p", "max"),
                                    responders=("y", "sum"), response_rate=("y", "mean"))
    dtab["lift"] = dtab["response_rate"] / y_t.mean()
    dtab["cum_recall"] = dtab["responders"].cumsum() / total_pos
    dtab.to_csv(TAB / "decile_table.csv")
    print("\nDecile table\n", dtab.to_string())

    # ---- 4. sensitivity: optimal contact share & profit across cost x margin (test labels, scaled)
    costs, margins = [100, 200, 300, 500, 800], [0.05, 0.10, 0.15, 0.20]
    prof_grid = pd.DataFrame(index=costs, columns=margins, dtype=float)
    share_grid = prof_grid.copy()
    for c in costs:
        for m in margins:
            _, _, _, _, pr = profit_curve(p_t, y_t, prem_t, cost=c, margin=m)
            prof_grid.loc[c, m] = max(pr.max(), 0) * scale
            share_grid.loc[c, m] = (np.argmax(pr) + 1) / n_t if pr.max() > 0 else 0.0
    prof_grid.to_csv(TAB / "sensitivity_profit.csv")
    share_grid.to_csv(TAB / "sensitivity_contact_share.csv")
    print("\nOptimal profit by contact cost (rows) x margin (cols)\n", prof_grid.round(0).to_string())
    print("\nOptimal contact share\n", share_grid.round(3).to_string())

    # ------------------------------------------------------------ figures
    x = k / n_t
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    models_all = [c for c in te.columns if c not in ("id", TARGET, "premium")]
    for m in models_all:
        o = np.argsort(-te[m].values, kind="stable")
        ax[0].plot(x, np.cumsum(y_t[o]) / total_pos, label=m, lw=2.2 if m == best else 1.1)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax[0].set(xlabel="Share of customers contacted", ylabel="Share of buyers reached", title="Cumulative gains")
    ax[0].legend()
    ax[1].bar(dtab.index, dtab["lift"], color="#4C72B0")
    ax[1].axhline(1, color="k", ls="--", lw=1)
    ax[1].set(xlabel="Decile (1 = highest predicted probability)", ylabel="Lift vs. average",
              title=f"Lift by decile ({best})")
    fig.tight_layout()
    fig.savefig(FIG / "13_gains_and_lift.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, prof * scale / 1e6, color="#2a7f62", lw=2, label=f"Model ranking ({best})")
    ax.plot(x, (((y_t * prem_t).mean() * M - C) * k * scale) / 1e6, color="grey", ls="--",
            label="Random contact order")
    ax.axvline(best_share, color="#DD8452", ls=":", label=f"Chosen on validation: {best_share:.0%}")
    ax.scatter([best_share], [prof[max(1, round(best_share * n_t)) - 1] * scale / 1e6], color="#DD8452", zorder=5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="Share of customers contacted (ranked by probability)", ylabel="Campaign profit (millions)",
           title=f"Profit vs. contact budget (cost {C:,.0f}/contact, margin {M:.0%} of premium)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "14_profit_curve.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(prof_grid.astype(float) / 1e6, annot=True, fmt=".1f", cmap="Greens", ax=axes[0])
    axes[0].set(title="Max profit (millions)", xlabel="Margin rate", ylabel="Contact cost")
    sns.heatmap(share_grid.astype(float), annot=True, fmt=".0%", cmap="Blues", ax=axes[1])
    axes[1].set(title="Profit-optimal contact share", xlabel="Margin rate", ylabel="Contact cost")
    fig.tight_layout()
    fig.savefig(FIG / "15_sensitivity.png", dpi=130)
    plt.close(fig)

    # ---- 5. final scoring of the unlabeled test.csv: refit best config on ALL labelled data
    print(f"\nRefitting {best} on all labelled data and scoring test.csv ...", flush=True)
    Xall, yall = split_xy(train_full)
    final = get_models()[best].fit(Xall, yall)
    scored = test_unl[["id"]].copy()
    scored["predicted_probability"] = final.predict_proba(test_unl)[:, 1]
    scored["Annual_Premium"] = test_unl["Annual_Premium"].values
    scored["expected_profit_if_contacted"] = scored["predicted_probability"] * scored["Annual_Premium"] * M - C
    scored = scored.sort_values("predicted_probability", ascending=False).reset_index(drop=True)
    scored["rank"] = scored.index + 1
    scored["decile"] = (scored.index * 10 // len(scored)) + 1
    n_contact = round(best_share * len(scored))
    scored["contact"] = (scored["rank"] <= n_contact).astype(int)
    scored.to_csv(OUT / "campaign_target_list.csv", index=False)
    chosen = scored[scored["contact"] == 1]
    print(f"Contact list: {n_contact:,} of {len(scored):,} customers; expected conversions "
          f"{chosen['predicted_probability'].sum():,.0f}; expected profit "
          f"{chosen['expected_profit_if_contacted'].sum():,.0f} (min prob in list {chosen['predicted_probability'].min():.3f})")


if __name__ == "__main__":
    main()
