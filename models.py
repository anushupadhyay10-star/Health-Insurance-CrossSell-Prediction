"""Model factory: one place that defines every candidate model and its hyper-parameters."""
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from pipeline import FeatureEngineer, HIGH_CARD, SEED, make_pipeline

N_JOBS = 2


def get_models():
    """Return {name: unfitted sklearn Pipeline}.

    No class re-weighting is used on purpose: we want well-calibrated probabilities,
    because the business step ranks customers and multiplies probabilities by revenue.
    Imbalance is handled afterwards via threshold / budget selection.
    """
    models = {}

    models["Logistic Regression"] = make_pipeline(
        LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED), kind="linear")

    models["Random Forest"] = make_pipeline(
        RandomForestClassifier(n_estimators=250, max_depth=16, min_samples_leaf=30,
                               max_features="sqrt", n_jobs=N_JOBS, random_state=SEED), kind="tree")

    models["XGBoost"] = make_pipeline(
        XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                      tree_method="hist", eval_metric="aucpr", n_jobs=N_JOBS,
                      random_state=SEED, verbosity=0), kind="tree")

    models["LightGBM"] = make_pipeline(
        LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=48, min_child_samples=40,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
                       n_jobs=N_JOBS, random_state=SEED, verbose=-1), kind="tree")

    # CatBoost handles the high-cardinality codes natively (ordered target statistics),
    # so it skips the target-encoding preprocessor and gets the engineered DataFrame directly.
    models["CatBoost"] = Pipeline([
        ("fe", FeatureEngineer()),
        ("model", CatBoostClassifier(iterations=600, learning_rate=0.08, depth=6, l2_leaf_reg=5,
                                     cat_features=HIGH_CARD, eval_metric="AUC", thread_count=N_JOBS,
                                     random_seed=SEED, verbose=0, allow_writing_files=False)),
    ])
    return models
