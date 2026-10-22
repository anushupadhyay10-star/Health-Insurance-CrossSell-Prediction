"""Step 2 - Feature engineering and leakage-safe preprocessing.

Everything that learns from data (scalers, encoders, target encoding) lives inside
sklearn Pipelines / ColumnTransformers, so it is fitted on the training fold only and
merely applied to validation / test data.  `id` is dropped and never used as a predictor.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

ROOT = Path(__file__).resolve().parents[1]
TARGET = "Response"
ID_COL = "id"
SEED = 42
PREMIUM_FLOOR = 2630.0  # 17% of customers sit exactly on the minimum premium

VEHICLE_AGE_MAP = {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}


def load_data():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    return train, test


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless row-wise transformations (no fitting => no leakage possible).

    * drops `id`
    * Gender / Vehicle_Damage -> 0/1, Vehicle_Age -> ordinal 0/1/2
    * log1p(Annual_Premium) to tame the right skew (skew 1.77 raw)
    * flag for premiums sitting exactly at the minimum value
    * Damage x Not-previously-insured interaction (the strongest segment in the EDA)
    * Region_Code / Policy_Sales_Channel cast to integer category labels
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=[c for c in [ID_COL, TARGET] if c in X.columns])
        out = pd.DataFrame(index=X.index)
        out["Gender_Male"] = (X["Gender"] == "Male").astype(int)
        out["Age"] = X["Age"].astype(int)
        out["Driving_License"] = X["Driving_License"].astype(int)
        out["Previously_Insured"] = X["Previously_Insured"].astype(int)
        out["Vehicle_Age"] = X["Vehicle_Age"].map(VEHICLE_AGE_MAP).astype(int)
        out["Vehicle_Damage"] = (X["Vehicle_Damage"] == "Yes").astype(int)
        out["Log_Premium"] = np.log1p(X["Annual_Premium"])
        out["Premium_Floor"] = (X["Annual_Premium"] <= PREMIUM_FLOOR).astype(int)
        out["Vintage"] = X["Vintage"].astype(int)
        out["Damage_NotInsured"] = out["Vehicle_Damage"] * (1 - out["Previously_Insured"])
        out["Region_Code"] = X["Region_Code"].astype(int).astype(str)
        out["Policy_Sales_Channel"] = X["Policy_Sales_Channel"].astype(int).astype(str)
        self.columns_ = list(out.columns)
        return out

    def get_feature_names_out(self, input_features=None):
        # actual column order of the engineered frame (needed to label feature importances correctly)
        return np.array(self.columns_)


BINARY = ["Gender_Male", "Driving_License", "Previously_Insured", "Vehicle_Damage",
          "Premium_Floor", "Damage_NotInsured"]
NUMERIC = ["Age", "Vehicle_Age", "Log_Premium", "Vintage"]
HIGH_CARD = ["Region_Code", "Policy_Sales_Channel"]
ALL_FEATURES = NUMERIC + BINARY + HIGH_CARD


def linear_preprocessor():
    """For Logistic Regression: scale numerics, one-hot high-cardinality codes (rare levels pooled)."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                              sparse_output=False), HIGH_CARD),
    ])


def tree_preprocessor():
    """For Random Forest / XGBoost / LightGBM: cross-fitted target encoding of the high-cardinality codes.

    sklearn's TargetEncoder uses internal 5-fold cross-fitting inside `fit_transform`,
    so the training rows never see their own label in their encoded value.
    """
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC + BINARY),
        ("te", TargetEncoder(target_type="binary", smooth="auto",
                             cv=KFold(n_splits=5, shuffle=True, random_state=SEED)), HIGH_CARD),
    ])


def make_pipeline(model, kind="tree"):
    pre = linear_preprocessor() if kind == "linear" else tree_preprocessor()
    return Pipeline([("fe", FeatureEngineer()), ("pre", pre), ("model", model)])


def split_xy(df):
    return df.drop(columns=[TARGET]), df[TARGET].values
