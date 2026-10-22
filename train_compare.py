"""Steps 3 & 4 - Train, compare and evaluate all models.

Split strategy (stratified, fixed seed):
    train 64 %  -> fit models / preprocessing
    valid 16 %  -> compare models and choose the decision threshold
    test  20 %  -> untouched hold-out used for the final numbers and the business simulation
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, log_loss,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import get_models  # noqa: E402
from pipeline import ROOT, SEED, TARGET, load_data, split_xy  # noqa: E402

OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
for d in (FIG, TAB, OUT / "models"):
    d.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def metrics_at(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat),
        "f1": f1_score(y, yhat),
        "accuracy": float((yhat == y).mean()),
        "tn_fp_fn_tp": confusion_matrix(y, yhat).ravel().tolist(),
    }


def main():
    train_full, _ = load_data()
    tr, tmp = train_test_split(train_full, test_size=0.36, stratify=train_full[TARGET], random_state=SEED)
    va, te = train_test_split(tmp, test_size=20 / 36, stratify=tmp[TARGET], random_state=SEED)
    Xtr, ytr = split_xy(tr)
    Xva, yva = split_xy(va)
    Xte, yte = split_xy(te)
    print(f"train={len(tr):,} valid={len(va):,} test={len(te):,}; "
          f"positive rate {ytr.mean():.4f}/{yva.mean():.4f}/{yte.mean():.4f}", flush=True)

    rows, preds_val, preds_te, fitted = [], {}, {}, {}
    for name, pipe in get_models().items():
        t0 = time.time()
        pipe.fit(Xtr, ytr)
        fit_s = time.time() - t0
        pv = pipe.predict_proba(Xva)[:, 1]
        pt = pipe.predict_proba(Xte)[:, 1]
        thr, f1v = best_f1_threshold(yva, pv)
        m_f1 = metrics_at(yte, pt, thr)
        m_50 = metrics_at(yte, pt, 0.5)
        rows.append({
            "model": name,
            "val_roc_auc": roc_auc_score(yva, pv), "val_pr_auc": average_precision_score(yva, pv),
            "test_roc_auc": roc_auc_score(yte, pt), "test_pr_auc": average_precision_score(yte, pt),
            "test_logloss": log_loss(yte, pt),
            "threshold_f1_on_val": thr,
            "test_precision": m_f1["precision"], "test_recall": m_f1["recall"], "test_f1": m_f1["f1"],
            "test_accuracy": m_f1["accuracy"],
            "test_precision@0.5": m_50["precision"], "test_recall@0.5": m_50["recall"],
            "test_f1@0.5": m_50["f1"], "test_accuracy@0.5": m_50["accuracy"],
            "fit_seconds": fit_s,
        })
        preds_val[name], preds_te[name], fitted[name] = pv, pt, pipe
        print(f"{name:22s} val AUC {rows[-1]['val_roc_auc']:.4f} PR {rows[-1]['val_pr_auc']:.4f} | "
              f"test AUC {rows[-1]['test_roc_auc']:.4f} PR {rows[-1]['test_pr_auc']:.4f} "
              f"F1 {m_f1['f1']:.4f} (thr {thr:.3f}) [{fit_s:.0f}s]", flush=True)

    res = pd.DataFrame(rows).set_index("model")
    res.to_csv(TAB / "model_comparison.csv")
    best = res["val_pr_auc"].idxmax()  # selection on validation only (no test peeking)
    print("\nBest model by validation PR-AUC:", best, flush=True)

    # persist predictions and the split ids for the business step
    pd.DataFrame({"id": te["id"].values, TARGET: yte, **{k: v for k, v in preds_te.items()}}) \
        .to_csv(OUT / "test_predictions.csv", index=False)
    pd.DataFrame({"id": va["id"].values, TARGET: yva, **{k: v for k, v in preds_val.items()}}) \
        .to_csv(OUT / "valid_predictions.csv", index=False)
    json.dump({"best_model": best, "threshold": float(res.loc[best, "threshold_f1_on_val"]),
               "seed": SEED}, open(OUT / "best_model.json", "w"), indent=2)
    joblib.dump(fitted[best], OUT / "models" / "best_model.joblib")

    # ------------------------------------------------------------ figures
    names = list(fitted)
    colors = dict(zip(names, sns.color_palette("tab10", len(names))))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for n in names:
        fpr, tpr, _ = roc_curve(yte, preds_te[n])
        ax[0].plot(fpr, tpr, color=colors[n], label=f"{n} (AUC {res.loc[n, 'test_roc_auc']:.3f})")
        prec, rec, _ = precision_recall_curve(yte, preds_te[n])
        ax[1].plot(rec, prec, color=colors[n], label=f"{n} (AP {res.loc[n, 'test_pr_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (hold-out test)")
    ax[1].axhline(yte.mean(), color="k", ls="--", lw=1, label=f"No-skill ({yte.mean():.3f})")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (hold-out test)")
    ax[0].legend(loc="lower right")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "08_roc_pr_curves.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.8))
    for ax, n in zip(axes, names):
        thr = res.loc[n, "threshold_f1_on_val"]
        cm = confusion_matrix(yte, (preds_te[n] >= thr).astype(int))
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        ax.set(title=f"{n}\nthr={thr:.2f}", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(FIG / "09_confusion_matrices.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot = res[["test_roc_auc", "test_pr_auc", "test_f1"]].rename(
        columns={"test_roc_auc": "ROC-AUC", "test_pr_auc": "PR-AUC", "test_f1": "F1 (val-tuned thr)"})
    plot.plot(kind="bar", ax=ax, rot=20)
    ax.set_title("Model comparison on hold-out test")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "10_model_comparison_bars.png", dpi=130)
    plt.close(fig)

    # best-model F1 vs. threshold, plus precision/recall trade-off
    pb = preds_val[best]
    prec, rec, thr_arr = precision_recall_curve(yva, pb)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thr_arr, prec[:-1], label="Precision")
    ax.plot(thr_arr, rec[:-1], label="Recall")
    ax.plot(thr_arr, f1, label="F1")
    ax.axvline(res.loc[best, "threshold_f1_on_val"], color="k", ls="--", lw=1)
    ax.set(xlabel="Threshold", title=f"{best}: precision / recall / F1 vs. threshold (validation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "11_threshold_tradeoff.png", dpi=130)
    plt.close(fig)

    # permutation-free importance for the best model when available
    try:
        mdl = fitted[best].named_steps["model"]
        if hasattr(mdl, "feature_importances_"):
            from pipeline import ALL_FEATURES
            names_out = (fitted[best][:-1].get_feature_names_out()
                         if hasattr(fitted[best][:-1], "get_feature_names_out") else ALL_FEATURES)
            imp = pd.Series(mdl.feature_importances_, index=[str(s).split("__")[-1] for s in names_out]) \
                .sort_values()
            fig, ax = plt.subplots(figsize=(7, 5))
            imp.plot(kind="barh", ax=ax, color="#4C72B0")
            ax.set_title(f"{best} feature importance")
            fig.tight_layout()
            fig.savefig(FIG / "12_feature_importance.png", dpi=130)
            plt.close(fig)
            imp.sort_values(ascending=False).to_csv(TAB / "feature_importance.csv")
    except Exception as e:  # importance plot is a nice-to-have
        print("importance plot skipped:", e)

    pd.set_option("display.width", 220)
    print("\n", res.round(4).T.to_string(), flush=True)


if __name__ == "__main__":
    main()
