"""Model training and prediction helpers for the Lexile calculator."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXCLUDE_FEATURE_PATTERNS = (
    r"^lexile$",
    r"^band$",
    r"^file$",
    r"^lexile_",
    r"^estimated_",
    r"^score_",
    r"^focus_500_900$",
)

PUBLIC_FEATURES = ["lmsl_ln", "wordfreq_content_zipf_mean"]

COMPACT_FEATURES = [
    "lmsl_ln",
    "wordfreq_content_zipf_mean",
    "subtlex_content_mlwf_floor",
    "textstat_ari",
    "textstat_dale_chall",
    "textstat_spache",
    "sentence_len_p90",
    "long_sentence_20_ratio",
    "clause_per_sentence",
    "finite_clause_per_sentence",
    "noun_phrase_len_mean",
    "aoa_p90",
    "aoa_9plus_ratio",
    "cefr_b1plus_ratio",
    "unknown_noun_adj_token_ratio",
    "proper_noun_ratio",
]

ROUTE_SPECS: dict[str, tuple[str, str]] = {
    "T": ("range_500_700", "all_hgb"),
    "H": ("range_700_900", "all_hgb"),
    "E": ("range_700_1000", "all_extra_trees"),
    "O": ("range_900_1150", "all_ridge"),
    "P": ("all", "all_hgb"),
    "N": ("all", "all_hgb"),
}

GLOBAL_SPEC = ("all", "all_hgb")


@dataclass(frozen=True)
class FittedModel:
    population: str
    model_name: str
    features: list[str]
    estimator: Any
    train_n: int


def read_training_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training feature file does not exist: {path}")
    df = pd.read_csv(path)
    df = df[df["lexile"].notna()].copy()
    df["lexile"] = df["lexile"].astype(float)
    return df


def usable_numeric_features(df: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for column in df.columns:
        if any(re.search(pattern, column) for pattern in EXCLUDE_FEATURE_PATTERNS):
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            features.append(column)
    return features


def linear_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def tree_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def model_specs(all_features: list[str]) -> dict[str, dict[str, Any]]:
    compact = [feature for feature in COMPACT_FEATURES if feature in all_features]
    public = [feature for feature in PUBLIC_FEATURES if feature in all_features]
    return {
        "public_lmsl_wordfreq_ridge": {
            "features": public,
            "estimator": linear_pipeline(Ridge(alpha=1.0)),
        },
        "compact_ridge": {
            "features": compact,
            "estimator": linear_pipeline(Ridge(alpha=10.0)),
        },
        "all_ridge": {
            "features": all_features,
            "estimator": linear_pipeline(Ridge(alpha=20.0)),
        },
        "all_hgb": {
            "features": all_features,
            "estimator": tree_pipeline(
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=350,
                    min_samples_leaf=18,
                    l2_regularization=0.05,
                    random_state=17,
                )
            ),
        },
        "all_extra_trees": {
            "features": all_features,
            "estimator": tree_pipeline(
                ExtraTreesRegressor(
                    n_estimators=600,
                    min_samples_leaf=5,
                    max_features=0.7,
                    random_state=17,
                    n_jobs=-1,
                )
            ),
        },
    }


def select_population(df: pd.DataFrame, population: str) -> pd.DataFrame:
    if population == "range_500_700":
        return df[df["lexile"].between(500, 700)].copy()
    if population == "range_700_900":
        return df[df["lexile"].between(700, 900)].copy()
    if population == "range_700_1000":
        return df[df["lexile"].between(700, 1000)].copy()
    if population == "range_900_1150":
        return df[df["lexile"].between(900, 1150)].copy()
    if population == "all":
        return df.copy()
    raise ValueError(f"Unknown model population: {population}")


class ModelRegistry:
    """Train and cache the small set of sklearn models used by the calculator."""

    def __init__(self, training_df: pd.DataFrame):
        self.training_df = training_df.copy()
        self.all_features = usable_numeric_features(self.training_df)
        self.specs = model_specs(self.all_features)
        self._cache: dict[tuple[str, str], FittedModel] = {}

    @classmethod
    def from_feature_csv(cls, path: Path) -> "ModelRegistry":
        return cls(read_training_features(path))

    def get(self, population: str, model_name: str) -> FittedModel:
        key = (population, model_name)
        if key not in self._cache:
            self._cache[key] = self._fit(population, model_name)
        return self._cache[key]

    def global_model(self) -> FittedModel:
        return self.get(*GLOBAL_SPEC)

    def route_model(self, level: str) -> FittedModel:
        return self.get(*ROUTE_SPECS[level])

    def predict(self, feature_row: dict[str, Any], fitted: FittedModel) -> float:
        frame = self._frame_for_prediction(feature_row, fitted.features)
        return float(fitted.estimator.predict(frame[fitted.features])[0])

    def _fit(self, population: str, model_name: str) -> FittedModel:
        if model_name not in self.specs:
            raise ValueError(f"Unknown model: {model_name}")
        subset = select_population(self.training_df, population)
        spec = self.specs[model_name]
        features = list(spec["features"])
        if not features:
            raise ValueError(f"Model {model_name} has no available features.")
        estimator = clone(spec["estimator"])
        estimator.fit(subset[features], subset["lexile"].astype(float).to_numpy())
        return FittedModel(
            population=population,
            model_name=model_name,
            features=features,
            estimator=estimator,
            train_n=len(subset),
        )

    @staticmethod
    def _frame_for_prediction(feature_row: dict[str, Any], features: list[str]) -> pd.DataFrame:
        aligned = dict(feature_row)
        for feature in features:
            if feature not in aligned:
                aligned[feature] = np.nan
        return pd.DataFrame([aligned])

