"""
Tune, fit, and evaluate the random-forest classifier.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import joblib

_CACHE_ROOT = Path(tempfile.gettempdir()) / "assignment4-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.common_pipeline import (
    DEFAULT_DATA_PATH,
    DEFAULT_TEST_INDICES_PATH,
    DEFAULT_TRAIN_INDICES_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    build_tree_preprocessor,
    calculate_metrics,
    classification_report_frame,
    dataset_audit,
    load_dataset,
    prepare_data,
    reproduce_assignment2_split,
    save_json,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "random_forest"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "selected_parameters.json"


def build_forest_pipeline(X: pd.DataFrame, **forest_parameters: object) -> Pipeline:
    """
    Construct one complete preprocessing and Random Forest pipeline.
    """

    parameters: dict[str, object] = {
        "n_estimators": 200,
        "criterion": "entropy",
        "max_features": "sqrt",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    parameters.update(forest_parameters)
    return Pipeline(
        [
            ("preprocess", build_tree_preprocessor(X)),
            ("forest", RandomForestClassifier(**parameters)),
        ]
    )

def run_forest_size_tuning(
    csv_path: Path,
    train_indices_path: Path,
    test_indices_path: Path,
    output_dir: Path,
) -> None:
    """
    Measure performance and cost as the number of trees increases.
    """

    output_dir = Path(output_dir) / "tuning" / "forest_size"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(csv_path)
    prepared = prepare_data(df)
    split, split_validation = reproduce_assignment2_split(
        prepared,
        train_indices_path=train_indices_path,
        test_indices_path=test_indices_path,
    )
    save_json(dataset_audit(df, prepared), output_dir / "data_audit.json")
    save_json(split_validation, output_dir / "split_validation.json")

    # Begin from the regularisation selected for the Assignment 2 tree. The
    # sqrt feature rule introduces the feature randomness required by a forest.
    model = build_forest_pipeline(
        split.X_train,
        n_estimators=100,
        criterion="entropy",
        max_features="sqrt",
        max_depth=20,
        min_samples_leaf=10,
        class_weight=None,
        n_jobs=-1,
    )
    # test for tree counts 50, 100, 200, 400
    tree_counts = [50, 100, 200, 400]
    search = GridSearchCV(
        model,
        param_grid={"forest__n_estimators": tree_counts},
        scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
        refit=False,
        cv=StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
        n_jobs=1,
        verbose=2,
        return_train_score=True,
    )
    search.fit(split.X_train, split.y_train, groups=split.groups_train)
    results = pd.DataFrame(search.cv_results_).sort_values(
        "param_forest__n_estimators"
    )
    results.to_csv(output_dir / "cv_results.csv", index=False)

    selected_count, threshold = _smallest_within_one_standard_error(results)
    best_row = results.loc[results["mean_test_macro_f1"].idxmax()]
    selected_row = results.loc[
        results["param_forest__n_estimators"].astype(int).eq(selected_count)
    ].iloc[0]
    summary = {
        "research_question": (
            "At approximately how many trees does validation macro-F1 stop "
            "improving enough to justify additional computation?"
        ),
        "held_out_test_scored": False,
        "selection_rule": (
            "Smallest forest within one standard error of the highest mean "
            "cross-validation macro-F1"
        ),
        "best_mean_tree_count": int(best_row["param_forest__n_estimators"]),
        "best_mean_cv_macro_f1": float(best_row["mean_test_macro_f1"]),
        "one_standard_error_threshold": threshold,
        "selected_tree_count": selected_count,
        "selected_cv_macro_f1_mean": float(selected_row["mean_test_macro_f1"]),
        "selected_cv_macro_f1_std": float(selected_row["std_test_macro_f1"]),
        "selected_mean_fit_seconds": float(selected_row["mean_fit_time"]),
    }
    save_json(summary, output_dir / "summary.json")
    _save_forest_size_plot(results, output_dir / "forest_size.png")
    print(json.dumps(summary, indent=2))
    print(f"Forest-size tuning artifacts saved to {output_dir}")


def run_feature_subsampling_tuning(
    csv_path: Path,
    train_indices_path: Path,
    test_indices_path: Path,
    output_dir: Path,
) -> None:
    """
    Measure how feature subsampling changes diversity and performance.
    """

    output_dir = Path(output_dir) / "tuning" / "feature_subsampling"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(csv_path)
    prepared = prepare_data(df)
    split, split_validation = reproduce_assignment2_split(
        prepared,
        train_indices_path=train_indices_path,
        test_indices_path=test_indices_path,
    )
    save_json(dataset_audit(df, prepared), output_dir / "data_audit.json")
    save_json(split_validation, output_dir / "split_validation.json")

    model = build_forest_pipeline(
        split.X_train,
        n_estimators=200,
        criterion="entropy",
        max_features="sqrt",
        max_depth=20,
        min_samples_leaf=10,
        class_weight=None,
        n_jobs=-1,
    )
    feature_options: list[object] = ["log2", "sqrt", 0.25, 0.5, 1.0]
    search = GridSearchCV(
        model,
        param_grid={"forest__max_features": feature_options},
        scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
        refit=False,
        cv=StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
        n_jobs=1,
        verbose=2,
        return_train_score=True,
    )
    search.fit(split.X_train, split.y_train, groups=split.groups_train)
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(output_dir / "cv_results.csv", index=False)

    best_row = results.loc[results["mean_test_macro_f1"].idxmax()]
    selected_value = best_row["param_forest__max_features"]
    if isinstance(selected_value, np.generic):
        selected_value = selected_value.item()
    summary = {
        "research_question": (
            "How does the number of candidate predictors at each split affect "
            "tree diversity, validation macro-F1, and computation?"
        ),
        "held_out_test_scored": False,
        "fixed_n_estimators": 200,
        "selection_rule": "Highest mean cross-validation macro-F1",
        "selected_max_features": selected_value,
        "selected_cv_macro_f1_mean": float(best_row["mean_test_macro_f1"]),
        "selected_cv_macro_f1_std": float(best_row["std_test_macro_f1"]),
        "selected_mean_fit_seconds": float(best_row["mean_fit_time"]),
    }
    save_json(summary, output_dir / "summary.json")
    _save_feature_subsampling_plot(
        results, output_dir / "feature_subsampling.png"
    )
    print(json.dumps(summary, indent=2))
    print(f"Feature-subsampling artifacts saved to {output_dir}")


def run_leaf_size_tuning(
    csv_path: Path,
    train_indices_path: Path,
    test_indices_path: Path,
    output_dir: Path,
) -> None:
    """
    Measure how leaf node regularisation affects the random forest model.
    """

    output_dir = Path(output_dir) / "tuning" / "leaf_size"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(csv_path)
    prepared = prepare_data(df)
    split, split_validation = reproduce_assignment2_split(
        prepared,
        train_indices_path=train_indices_path,
        test_indices_path=test_indices_path,
    )
    save_json(dataset_audit(df, prepared), output_dir / "data_audit.json")
    save_json(split_validation, output_dir / "split_validation.json")

    model = build_forest_pipeline(
        split.X_train,
        n_estimators=200,
        criterion="entropy",
        max_features=0.5,
        max_depth=20,
        min_samples_leaf=10,
        class_weight=None,
        n_jobs=-1,
    )
    leaf_sizes = [1, 5, 10, 25]
    search = GridSearchCV(
        model,
        param_grid={"forest__min_samples_leaf": leaf_sizes},
        scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
        refit=False,
        cv=StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
        n_jobs=1,
        verbose=2,
        return_train_score=True,
    )
    search.fit(split.X_train, split.y_train, groups=split.groups_train)
    results = pd.DataFrame(search.cv_results_).sort_values(
        "param_forest__min_samples_leaf"
    )
    results.to_csv(output_dir / "cv_results.csv", index=False)

    best_row = results.loc[results["mean_test_macro_f1"].idxmax()]
    summary = {
        "research_question": (
            "Does allowing smaller terminal nodes improve minority-class "
            "learning enough to justify the added tree complexity?"
        ),
        "held_out_test_scored": False,
        "fixed_n_estimators": 200,
        "fixed_max_features": 0.5,
        "fixed_max_depth": 20,
        "selection_rule": "Highest mean cross-validation macro-F1",
        "selected_min_samples_leaf": int(
            best_row["param_forest__min_samples_leaf"]
        ),
        "selected_cv_macro_f1_mean": float(best_row["mean_test_macro_f1"]),
        "selected_cv_macro_f1_std": float(best_row["std_test_macro_f1"]),
        "selected_mean_fit_seconds": float(best_row["mean_fit_time"]),
    }
    save_json(summary, output_dir / "summary.json")
    _save_leaf_size_plot(results, output_dir / "leaf_size.png")
    print(json.dumps(summary, indent=2))
    print(f"Leaf-size tuning artifacts saved to {output_dir}")


def run_depth_tuning(
    csv_path: Path,
    train_indices_path: Path,
    test_indices_path: Path,
    output_dir: Path,
) -> None:
    """
    Test howe limiting individual-tree depth affects generalisation.
    """

    output_dir = Path(output_dir) / "tuning" / "depth"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(csv_path)
    prepared = prepare_data(df)
    split, split_validation = reproduce_assignment2_split(
        prepared,
        train_indices_path=train_indices_path,
        test_indices_path=test_indices_path,
    )
    save_json(dataset_audit(df, prepared), output_dir / "data_audit.json")
    save_json(split_validation, output_dir / "split_validation.json")

    model = build_forest_pipeline(
        split.X_train,
        n_estimators=200,
        criterion="entropy",
        max_features=0.5,
        max_depth=20,
        min_samples_leaf=1,
        class_weight=None,
        n_jobs=-1,
    )
    depth_options: list[object] = [10, 20, 30, None]
    search = GridSearchCV(
        model,
        param_grid={"forest__max_depth": depth_options},
        scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
        refit=False,
        cv=StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
        n_jobs=1,
        verbose=2,
        return_train_score=True,
    )
    search.fit(split.X_train, split.y_train, groups=split.groups_train)
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(output_dir / "cv_results.csv", index=False)

    best_row = results.loc[results["mean_test_macro_f1"].idxmax()]
    selected_depth = best_row["param_forest__max_depth"]
    if pd.isna(selected_depth):
        selected_depth = None
    else:
        selected_depth = int(selected_depth)
    summary = {
        "research_question": (
            "Can a depth limit retain the benefit of small leaves while "
            "reducing overfitting by individual trees?"
        ),
        "held_out_test_scored": False,
        "fixed_n_estimators": 200,
        "fixed_max_features": 0.5,
        "fixed_min_samples_leaf": 1,
        "selection_rule": "Highest mean cross-validation macro-F1",
        "selected_max_depth": selected_depth,
        "selected_cv_macro_f1_mean": float(best_row["mean_test_macro_f1"]),
        "selected_cv_macro_f1_std": float(best_row["std_test_macro_f1"]),
        "selected_mean_fit_seconds": float(best_row["mean_fit_time"]),
    }
    save_json(summary, output_dir / "summary.json")
    _save_depth_plot(results, output_dir / "depth.png")
    print(json.dumps(summary, indent=2))
    print(f"Depth-tuning artifacts saved to {output_dir}")


def run_class_weight_tuning(
    csv_path: Path,
    train_indices_path: Path,
    test_indices_path: Path,
    output_dir: Path,
) -> None:
    """
    Measure whether class weighting improves minority-class detection.
    """

    output_dir = Path(output_dir) / "tuning" / "class_weight"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(csv_path)
    prepared = prepare_data(df)
    split, split_validation = reproduce_assignment2_split(
        prepared,
        train_indices_path=train_indices_path,
        test_indices_path=test_indices_path,
    )
    save_json(dataset_audit(df, prepared), output_dir / "data_audit.json")
    save_json(split_validation, output_dir / "split_validation.json")

    model = build_forest_pipeline(
        split.X_train,
        n_estimators=200,
        criterion="entropy",
        max_features=0.5,
        max_depth=None,
        min_samples_leaf=1,
        class_weight=None,
        n_jobs=-1,
    )
    minority_recall = make_scorer(
        recall_score,
        labels=[2, 5, 7, 8],
        average="macro",
        zero_division=0,
    )
    weight_options: list[object] = [None, "balanced", "balanced_subsample"]
    search = GridSearchCV(
        model,
        param_grid={"forest__class_weight": weight_options},
        scoring={
            "macro_f1": "f1_macro",
            "accuracy": "accuracy",
            "minority_recall": minority_recall,
        },
        refit=False,
        cv=StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
        n_jobs=1,
        verbose=2,
        return_train_score=True,
    )
    search.fit(split.X_train, split.y_train, groups=split.groups_train)
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(output_dir / "cv_results.csv", index=False)

    best_row = results.loc[results["mean_test_macro_f1"].idxmax()]
    selected_weight = best_row["param_forest__class_weight"]
    if pd.isna(selected_weight):
        selected_weight = None
    summary = {
        "research_question": (
            "Does class weighting improve recall for classes 2, 5, 7, and 8 "
            "without an unacceptable reduction in overall performance?"
        ),
        "held_out_test_scored": False,
        "minority_classes": [2, 5, 7, 8],
        "selection_rule": (
            "Highest mean cross-validation macro-F1, with minority recall "
            "reported as a secondary outcome"
        ),
        "selected_class_weight": selected_weight,
        "selected_cv_macro_f1_mean": float(best_row["mean_test_macro_f1"]),
        "selected_cv_macro_f1_std": float(best_row["std_test_macro_f1"]),
        "selected_cv_minority_recall_mean": float(
            best_row["mean_test_minority_recall"]
        ),
        "selected_cv_accuracy_mean": float(best_row["mean_test_accuracy"]),
    }
    save_json(summary, output_dir / "summary.json")
    _save_class_weight_plot(results, output_dir / "class_weight.png")
    print(json.dumps(summary, indent=2))
    print(f"Class-weight tuning artifacts saved to {output_dir}")


def run_tuning_check(
    csv_path: Path,
    train_indices_path: Path,
    test_indices_path: Path,
    output_dir: Path,
) -> None:
    """
    Check whether the selected configuration against nearby interaction changes.
    """

    output_dir = Path(output_dir) / "tuning" / "confirmation"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(csv_path)
    prepared = prepare_data(df)
    split, split_validation = reproduce_assignment2_split(
        prepared,
        train_indices_path=train_indices_path,
        test_indices_path=test_indices_path,
    )
    save_json(dataset_audit(df, prepared), output_dir / "data_audit.json")
    save_json(split_validation, output_dir / "split_validation.json")

    model = build_forest_pipeline(
        split.X_train,
        n_estimators=200,
        criterion="entropy",
        max_features=0.5,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
    )
    # Each dictionary is one deliberate adaptation of the selected hyperparameters.
    candidate_grid = [
        {
            "forest__max_features": [1.0],
            "forest__max_depth": [None],
            "forest__min_samples_leaf": [1],
            "forest__class_weight": ["balanced"],
        },
        {
            "forest__max_features": [0.5],
            "forest__max_depth": [30],
            "forest__min_samples_leaf": [1],
            "forest__class_weight": ["balanced"],
        },
        {
            "forest__max_features": [0.5],
            "forest__max_depth": [None],
            "forest__min_samples_leaf": [5],
            "forest__class_weight": ["balanced"],
        },
    ]
    minority_recall = make_scorer(
        recall_score,
        labels=[2, 5, 7, 8],
        average="macro",
        zero_division=0,
    )
    search = GridSearchCV(
        model,
        param_grid=candidate_grid,
        scoring={
            "macro_f1": "f1_macro",
            "accuracy": "accuracy",
            "minority_recall": minority_recall,
        },
        refit=False,
        cv=StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
        n_jobs=1,
        verbose=2,
        return_train_score=True,
    )
    search.fit(split.X_train, split.y_train, groups=split.groups_train)
    results = pd.DataFrame(search.cv_results_)
    results.insert(0, "candidate", results.apply(_confirmation_label, axis=1))
    results.to_csv(output_dir / "cv_results.csv", index=False)

    reference_path = (
        Path(output_dir).parent / "class_weight" / "cv_results.csv"
    )
    if not reference_path.is_file():
        raise FileNotFoundError(
            "Class-weight results are required as the selected-model reference: "
            f"{reference_path}"
        )
    class_weight_results = pd.read_csv(reference_path)
    reference = class_weight_results.loc[
        class_weight_results["param_forest__class_weight"].eq("balanced")
    ].iloc[0]
    comparison_rows = [
        {
            "candidate": "selected",
            "max_features": "0.5",
            "max_depth": "None",
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "mean_cv_macro_f1": reference["mean_test_macro_f1"],
            "std_cv_macro_f1": reference["std_test_macro_f1"],
            "mean_minority_recall": reference["mean_test_minority_recall"],
            "mean_accuracy": reference["mean_test_accuracy"],
        }
    ]
    for _, row in results.iterrows():
        comparison_rows.append(
            {
                "candidate": row["candidate"],
                "max_features": row["param_forest__max_features"],
                "max_depth": (
                    "None"
                    if pd.isna(row["param_forest__max_depth"])
                    else int(row["param_forest__max_depth"])
                ),
                "min_samples_leaf": int(row["param_forest__min_samples_leaf"]),
                "class_weight": row["param_forest__class_weight"],
                "mean_cv_macro_f1": row["mean_test_macro_f1"],
                "std_cv_macro_f1": row["std_test_macro_f1"],
                "mean_minority_recall": row["mean_test_minority_recall"],
                "mean_accuracy": row["mean_test_accuracy"],
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values(
        "mean_cv_macro_f1", ascending=False
    )
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    best_mean_row = comparison.iloc[0]
    one_standard_error = float(best_mean_row["std_cv_macro_f1"]) / np.sqrt(3)
    one_standard_error_threshold = (
        float(best_mean_row["mean_cv_macro_f1"]) - one_standard_error
    )
    eligible = comparison.loc[
        comparison["mean_cv_macro_f1"].ge(one_standard_error_threshold)
    ]
    winner = eligible.sort_values(
        "mean_minority_recall", ascending=False
    ).iloc[0]
    summary = {
        "research_question": (
            "Do nearby changes to feature sampling, depth, or leaf size "
            "outperform the sequentially selected configuration when combined "
            "with balanced class weighting?"
        ),
        "held_out_test_scored": False,
        "selection_rule": (
            "Among candidates within one standard error of the highest mean "
            "cross-validation macro-F1, select the highest minority recall"
        ),
        "one_standard_error_threshold": one_standard_error_threshold,
        "selected_candidate": winner["candidate"],
        "selected_cv_macro_f1_mean": float(winner["mean_cv_macro_f1"]),
        "selected_cv_macro_f1_std": float(winner["std_cv_macro_f1"]),
    }
    save_json(summary, output_dir / "summary.json")
    print(comparison.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"Confirmation artifacts saved to {output_dir}")


def run_final_evaluation(
    csv_path: Path,
    train_indices_path: Path,
    test_indices_path: Path,
    output_dir: Path,
    config_path: Path,
) -> None:
    """
    Fit the final chosen model and run the evaluation on th etest partition.
    """

    output_dir = Path(output_dir) / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text())
    if config.get("held_out_test_scored_during_selection") is not False:
        raise ValueError("Selected-parameter record does not confirm test isolation")
    parameters = config["parameters"]

    df = load_dataset(csv_path)
    prepared = prepare_data(df)
    split, split_validation = reproduce_assignment2_split(
        prepared,
        train_indices_path=train_indices_path,
        test_indices_path=test_indices_path,
    )
    save_json(dataset_audit(df, prepared), output_dir / "data_audit.json")
    save_json(split_validation, output_dir / "split_validation.json")

    model = build_forest_pipeline(split.X_train, **parameters)
    fit_started = time.perf_counter()
    model.fit(split.X_train, split.y_train)
    fit_seconds = time.perf_counter() - fit_started

    predict_started = time.perf_counter()
    prediction = model.predict(split.X_test)
    predict_seconds = time.perf_counter() - predict_started

    metrics = calculate_metrics(split.y_test, prediction)
    metrics.update(
        {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "train_rows": len(split.X_train),
            "test_rows": len(split.X_test),
        }
    )
    save_json(metrics, output_dir / "test_metrics.json")
    save_json(config, output_dir / "selected_parameters.json")

    report = classification_report(
        split.y_test, prediction, digits=4, zero_division=0
    )
    (output_dir / "classification_report.txt").write_text(report)
    classification_report_frame(split.y_test, prediction).to_csv(
        output_dir / "classification_report.csv", index=False
    )

    labels = np.sort(split.y_test.unique())
    matrix = confusion_matrix(split.y_test, prediction, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    _save_final_confusion_matrix(
        matrix, labels, output_dir / "confusion_matrix.png"
    )

    pd.DataFrame(
        {
            "row_index": split.X_test.index,
            "actual": split.y_test.to_numpy(),
            "predicted": prediction,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    _save_feature_importance(model, output_dir)
    joblib.dump(model, output_dir / "random_forest_pipeline.joblib")

    final_summary = {
        "model": "RandomForestClassifier",
        "parameters": parameters,
        "selection_cv_macro_f1_mean": config["selection_evidence"][
            "cv_macro_f1_mean"
        ],
        "selection_cv_macro_f1_std": config["selection_evidence"][
            "cv_macro_f1_std"
        ],
        "test_metrics": metrics,
        "test_evaluations_performed": 1,
    }
    save_json(final_summary, output_dir / "final_summary.json")
    print(json.dumps(final_summary, indent=2))
    print(f"Final evaluation artifacts saved to {output_dir}")


def _save_final_confusion_matrix(
    matrix: np.ndarray, labels: np.ndarray, output_path: Path
) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("Random Forest held-out test confusion matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _save_feature_importance(model: Pipeline, output_dir: Path) -> None:
    preprocess = model.named_steps["preprocess"]
    forest = model.named_steps["forest"]
    importance = pd.DataFrame(
        {
            "feature": preprocess.get_feature_names_out(),
            "importance": forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    top = importance.head(20).sort_values("importance")
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Mean decrease in impurity")
    plt.title("Top 20 Random Forest feature importances")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=180)
    plt.close()


def _confirmation_label(row: pd.Series) -> str:
    if float(row["param_forest__max_features"]) == 1.0:
        return "all_features"
    if not pd.isna(row["param_forest__max_depth"]):
        return "depth_30"
    return "leaf_5"


def _smallest_within_one_standard_error(
    results: pd.DataFrame,
) -> tuple[int, float]:
    best_row = results.loc[results["mean_test_macro_f1"].idxmax()]
    standard_error = float(best_row["std_test_macro_f1"]) / np.sqrt(3)
    threshold = float(best_row["mean_test_macro_f1"]) - standard_error
    eligible = results.loc[results["mean_test_macro_f1"].ge(threshold)].copy()
    selected = int(eligible["param_forest__n_estimators"].astype(int).min())
    return selected, threshold


def _save_forest_size_plot(results: pd.DataFrame, output_path: Path) -> None:
    tree_counts = results["param_forest__n_estimators"].astype(int)
    figure, performance_axis = plt.subplots(figsize=(8, 5))
    performance_axis.errorbar(
        tree_counts,
        results["mean_test_macro_f1"],
        yerr=results["std_test_macro_f1"],
        marker="o",
        capsize=4,
        label="Validation macro-F1",
    )
    performance_axis.plot(
        tree_counts,
        results["mean_train_macro_f1"],
        marker="s",
        linestyle="--",
        label="Training macro-F1",
    )
    performance_axis.set_xlabel("Number of trees")
    performance_axis.set_ylabel("Macro-F1")
    performance_axis.set_xticks(tree_counts)
    performance_axis.grid(alpha=0.25)

    time_axis = performance_axis.twinx()
    time_axis.plot(
        tree_counts,
        results["mean_fit_time"],
        color="tab:red",
        marker="^",
        label="Mean fit time",
    )
    time_axis.set_ylabel("Mean fit time per fold (seconds)", color="tab:red")
    handles_a, labels_a = performance_axis.get_legend_handles_labels()
    handles_b, labels_b = time_axis.get_legend_handles_labels()
    performance_axis.legend(handles_a + handles_b, labels_a + labels_b, loc="best")
    plt.title("Random Forest size: performance and computational cost")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _save_feature_subsampling_plot(
    results: pd.DataFrame, output_path: Path
) -> None:
    labels = results["param_forest__max_features"].map(str)
    positions = np.arange(len(results))
    figure, performance_axis = plt.subplots(figsize=(8, 5))
    performance_axis.errorbar(
        positions,
        results["mean_test_macro_f1"],
        yerr=results["std_test_macro_f1"],
        marker="o",
        capsize=4,
        label="Validation macro-F1",
    )
    performance_axis.plot(
        positions,
        results["mean_train_macro_f1"],
        marker="s",
        linestyle="--",
        label="Training macro-F1",
    )
    performance_axis.set_xticks(positions, labels)
    performance_axis.set_xlabel("Maximum features considered per split")
    performance_axis.set_ylabel("Macro-F1")
    performance_axis.grid(alpha=0.25)

    time_axis = performance_axis.twinx()
    time_axis.plot(
        positions,
        results["mean_fit_time"],
        color="tab:red",
        marker="^",
        label="Mean fit time",
    )
    time_axis.set_ylabel("Mean fit time per fold (seconds)", color="tab:red")
    handles_a, labels_a = performance_axis.get_legend_handles_labels()
    handles_b, labels_b = time_axis.get_legend_handles_labels()
    performance_axis.legend(handles_a + handles_b, labels_a + labels_b, loc="best")
    plt.title("Random Forest feature subsampling")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _save_leaf_size_plot(results: pd.DataFrame, output_path: Path) -> None:
    leaf_sizes = results["param_forest__min_samples_leaf"].astype(int)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        leaf_sizes,
        results["mean_test_macro_f1"],
        yerr=results["std_test_macro_f1"],
        marker="o",
        capsize=4,
        label="Validation macro-F1",
    )
    axis.plot(
        leaf_sizes,
        results["mean_train_macro_f1"],
        marker="s",
        linestyle="--",
        label="Training macro-F1",
    )
    axis.set_xlabel("Minimum observations per leaf")
    axis.set_ylabel("Macro-F1")
    axis.set_xticks(leaf_sizes)
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title("Random Forest terminal-node regularisation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _save_depth_plot(results: pd.DataFrame, output_path: Path) -> None:
    labels = results["param_forest__max_depth"].map(
        lambda value: "None" if pd.isna(value) else str(int(value))
    )
    positions = np.arange(len(results))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        positions,
        results["mean_test_macro_f1"],
        yerr=results["std_test_macro_f1"],
        marker="o",
        capsize=4,
        label="Validation macro-F1",
    )
    axis.plot(
        positions,
        results["mean_train_macro_f1"],
        marker="s",
        linestyle="--",
        label="Training macro-F1",
    )
    axis.set_xticks(positions, labels)
    axis.set_xlabel("Maximum tree depth")
    axis.set_ylabel("Macro-F1")
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title("Random Forest maximum-depth regularisation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _save_class_weight_plot(results: pd.DataFrame, output_path: Path) -> None:
    labels = results["param_forest__class_weight"].map(
        lambda value: "None" if pd.isna(value) else str(value)
    )
    positions = np.arange(len(results))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        positions - width / 2,
        results["mean_test_macro_f1"],
        width,
        yerr=results["std_test_macro_f1"],
        capsize=4,
        label="Validation macro-F1",
    )
    axis.bar(
        positions + width / 2,
        results["mean_test_minority_recall"],
        width,
        yerr=results["std_test_minority_recall"],
        capsize=4,
        label="Classes 2/5/7/8 mean recall",
    )
    axis.set_xticks(positions, labels)
    axis.set_xlabel("Class weighting")
    axis.set_ylabel("Score")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.set_title("Random Forest class-weighting trade-off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _stratified_training_sample(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    sample_size: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size >= len(X):
        return X, y, groups

    selected, _ = train_test_split(
        np.arange(len(X)),
        train_size=sample_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X.iloc[selected], y.iloc[selected], groups.iloc[selected]


def _save_confusion_matrix(
    matrix: np.ndarray, labels: np.ndarray, output_path: Path
) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("Random Forest smoke-test validation confusion matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-evaluation", action="store_true")
    parser.add_argument(
        "--tune-stage",
        choices=[
            "forest-size",
            "feature-subsampling",
            "leaf-size",
            "depth",
            "class-weight",
            "confirmation",
        ],
        help="Run one full-training tuning stage without scoring the test set.",
    )
    parser.add_argument("--sample-size", type=int, default=30_000)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--train-indices", type=Path, default=DEFAULT_TRAIN_INDICES_PATH
    )
    parser.add_argument(
        "--test-indices", type=Path, default=DEFAULT_TEST_INDICES_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    selected_operations = sum(
        [bool(args.smoke_test), bool(args.tune_stage), bool(args.final_evaluation)]
    )
    if selected_operations > 1:
        parser.error(
            "Choose only one of --tune-stage or "
            "--final-evaluation."
        )
    if args.tune_stage == "forest-size":
        run_forest_size_tuning(
            csv_path=args.data_path,
            train_indices_path=args.train_indices,
            test_indices_path=args.test_indices,
            output_dir=args.output_dir,
        )
    elif args.tune_stage == "feature-subsampling":
        run_feature_subsampling_tuning(
            csv_path=args.data_path,
            train_indices_path=args.train_indices,
            test_indices_path=args.test_indices,
            output_dir=args.output_dir,
        )
    elif args.tune_stage == "leaf-size":
        run_leaf_size_tuning(
            csv_path=args.data_path,
            train_indices_path=args.train_indices,
            test_indices_path=args.test_indices,
            output_dir=args.output_dir,
        )
    elif args.tune_stage == "depth":
        run_depth_tuning(
            csv_path=args.data_path,
            train_indices_path=args.train_indices,
            test_indices_path=args.test_indices,
            output_dir=args.output_dir,
        )
    elif args.tune_stage == "class-weight":
        run_class_weight_tuning(
            csv_path=args.data_path,
            train_indices_path=args.train_indices,
            test_indices_path=args.test_indices,
            output_dir=args.output_dir,
        )
    elif args.tune_stage == "confirmation":
        run_tuning_check(
            csv_path=args.data_path,
            train_indices_path=args.train_indices,
            test_indices_path=args.test_indices,
            output_dir=args.output_dir,
        )
    elif args.final_evaluation:
        run_final_evaluation(
            csv_path=args.data_path,
            train_indices_path=args.train_indices,
            test_indices_path=args.test_indices,
            output_dir=args.output_dir,
            config_path=args.config,
        )
    else:
        parser.error("No operation selected. Start with --smoke-test.")


if __name__ == "__main__":
    main()

