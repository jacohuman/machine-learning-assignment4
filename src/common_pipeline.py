"""
Shared utilities for the ensemble experiment.

Using this shared pipeline allows the comparison between the ensemble learning approach and the
KNN and classification tree models of assignment 2 to be more significant.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
TARGET = "attack_cat"
INNER_CV_FOLDS = 3
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENTS_ROOT = PROJECT_ROOT.parent
ASSIGNMENT2_ROOT = ASSIGNMENTS_ROOT / "assignment2"

DEFAULT_DATA_PATH = ASSIGNMENT2_ROOT / "networkTraffic.csv"
DEFAULT_TRAIN_INDICES_PATH = ASSIGNMENT2_ROOT / "tree_results" / "train_indices.csv"
DEFAULT_TEST_INDICES_PATH = ASSIGNMENT2_ROOT / "tree_results" / "test_indices.csv"


@dataclass(frozen=True)
class PreparedData:
    """
    The clean predictors, target values and duplicate-pattern identifiers.
    """

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series


@dataclass(frozen=True)
class DataSplit:
    """
    Reproduced Assignment 2 training and test partitions.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    groups_test: pd.Series


@dataclass(frozen=True)
class SplitValidation:
    """
    Auditable checks for the reproduced held-out split.
    """

    total_rows: int
    train_rows: int
    test_rows: int
    index_overlap: int
    unassigned_rows: int
    shared_predictor_groups: int
    train_class_counts: dict[str, int]
    test_class_counts: dict[str, int]


def load_dataset(csv_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load the supplied network-traffic data
    """

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path, sep=",", na_values="?", low_memory=False)


def prepare_data(df: pd.DataFrame) -> PreparedData:
    """
    Apply the row-level preparation used in Assignment 2.

    Predictor hashes are calculated after removing ``id`` but before any
    model-specific transformation. Equal raw predictor patterns can therefore
    be kept on the same side of every model-selection split.
    """

    if TARGET not in df.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing")

    clean = df.dropna(subset=[TARGET]).copy()
    clean = clean.drop(columns=["id"], errors="ignore")
    X = clean.drop(columns=TARGET)
    y = clean[TARGET]
    groups = pd.util.hash_pandas_object(X, index=False).rename("predictor_group")
    return PreparedData(X=X, y=y, groups=groups)


def _load_index_file(path: Path) -> pd.Index:
    """
    Read and validate one Assignment 2 file."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Saved split file not found: {path}")

    frame = pd.read_csv(path)
    expected_column = "cleaned_row_index"
    if frame.columns.tolist() != [expected_column]:
        raise ValueError(
            f"{path} must contain only a {expected_column!r} column; "
            f"found {frame.columns.tolist()}"
        )
    if frame[expected_column].duplicated().any():
        raise ValueError(f"Duplicate row indices found in {path}")
    return pd.Index(frame[expected_column].to_numpy(), name=expected_column)


def reproduce_assignment2_split(
    prepared: PreparedData,
    train_indices_path: Path = DEFAULT_TRAIN_INDICES_PATH,
    test_indices_path: Path = DEFAULT_TEST_INDICES_PATH,
) -> tuple[DataSplit, SplitValidation]:
    """
    Reproduce and validate the exact split from Assignment 2.
    """

    train_index = _load_index_file(train_indices_path)
    test_index = _load_index_file(test_indices_path)
    available_index = prepared.X.index

    overlap = train_index.intersection(test_index)
    requested = train_index.union(test_index)
    missing_from_data = requested.difference(available_index)
    unassigned = available_index.difference(requested)

    if len(overlap):
        raise ValueError(f"Train/test index overlap contains {len(overlap):,} rows")
    if len(missing_from_data):
        raise ValueError(
            f"Saved split refers to {len(missing_from_data):,} unavailable rows"
        )
    if len(unassigned):
        raise ValueError(f"Saved split leaves {len(unassigned):,} rows unassigned")

    split = DataSplit(
        X_train=prepared.X.loc[train_index],
        X_test=prepared.X.loc[test_index],
        y_train=prepared.y.loc[train_index],
        y_test=prepared.y.loc[test_index],
        groups_train=prepared.groups.loc[train_index],
        groups_test=prepared.groups.loc[test_index],
    )
    shared_groups = set(split.groups_train).intersection(set(split.groups_test))

    validation = SplitValidation(
        total_rows=len(prepared.X),
        train_rows=len(split.X_train),
        test_rows=len(split.X_test),
        index_overlap=len(overlap),
        unassigned_rows=len(unassigned),
        shared_predictor_groups=len(shared_groups),
        train_class_counts=_class_counts(split.y_train),
        test_class_counts=_class_counts(split.y_test),
    )
    if validation.shared_predictor_groups:
        raise ValueError(
            "Assignment 2 split leaks identical predictor groups across "
            f"partitions ({validation.shared_predictor_groups:,} shared groups)"
        )
    return split, validation


def build_tree_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build the leakage-safe preprocessing used by tree-based models.
    """

    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()
    if not numeric_columns:
        raise ValueError("No numeric predictors were found")

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=True,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = [
        ("numeric", numeric_pipeline, numeric_columns)
    ]
    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )
    return ColumnTransformer(transformers=transformers)


def dataset_audit(df: pd.DataFrame, prepared: PreparedData) -> dict[str, Any]:

    conflicting_groups = prepared.y.groupby(prepared.groups).nunique().gt(1).sum()
    missing = df.isna().sum()
    return {
        "raw_rows": int(len(df)),
        "raw_columns": int(df.shape[1]),
        "prepared_rows": int(len(prepared.X)),
        "predictor_columns": int(prepared.X.shape[1]),
        "numeric_predictors": int(
            len(prepared.X.select_dtypes(include=np.number).columns)
        ),
        "categorical_predictors": int(
            len(prepared.X.select_dtypes(exclude=np.number).columns)
        ),
        "missing_values_by_column": {
            str(column): int(count) for column, count in missing[missing.gt(0)].items()
        },
        "duplicate_rows_with_id": int(df.duplicated().sum()),
        "duplicate_rows_ignoring_id": int(
            df.drop(columns=["id"], errors="ignore").duplicated().sum()
        ),
        "unique_predictor_groups": int(prepared.groups.nunique()),
        "conflicting_predictor_groups": int(conflicting_groups),
        "target_class_counts": _class_counts(prepared.y),
    }


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculate the overall metrics (same as Assignment 2).
    """

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def classification_report_frame(
    y_true: pd.Series, y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Create a classification report.
    """

    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    return pd.DataFrame(report).transpose().rename_axis("class").reset_index()


def save_json(value: Any, path: Path) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n")


def _class_counts(y: pd.Series) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in y.value_counts(dropna=False).sort_index().items()
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

