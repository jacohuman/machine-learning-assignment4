"""
Load the Assignment 2 KNN and classification tree
baselines and create standardized comparison tables.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "assignment4-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.common_pipeline import ASSIGNMENT2_ROOT, PROJECT_ROOT


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison"
DEFAULT_FOREST_RESULTS = PROJECT_ROOT / "results" / "random_forest" / "final"
BASELINE_DIRS = {
    "KNN": ASSIGNMENT2_ROOT / "knn_results",
    "Classification tree": ASSIGNMENT2_ROOT / "tree_results",
}


def load_baseline_tables(
    baseline_dirs: dict[str, Path] = BASELINE_DIRS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load overall and class-specific metrics from Assignment 2 outputs.
    """

    overall_rows: list[dict[str, object]] = []
    class_frames: list[pd.DataFrame] = []

    for model_name, result_dir in baseline_dirs.items():
        result_dir = Path(result_dir)
        selection = _read_json(result_dir / "model_selection.json")
        test_metrics = _read_json(result_dir / "test_metrics.json")
        confusion = pd.read_csv(result_dir / "confusion_matrix.csv", index_col=0)

        overall_rows.append(
            {
                "model": model_name,
                "cv_macro_f1_mean": selection["best_cv_macro_f1_mean"],
                "cv_macro_f1_std": selection["best_cv_macro_f1_std"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_cohen_kappa": test_metrics["cohen_kappa"],
                "test_mcc": test_metrics["mcc"],
            }
        )
        class_metrics = metrics_from_confusion_matrix(confusion)
        class_metrics.insert(0, "model", model_name)
        class_frames.append(class_metrics)

    return pd.DataFrame(overall_rows), pd.concat(class_frames, ignore_index=True)


def metrics_from_confusion_matrix(confusion: pd.DataFrame) -> pd.DataFrame:
    """
    Recover per-class precision, recall, F1 and support from the confusion matrix.
    """

    if confusion.shape[0] != confusion.shape[1]:
        raise ValueError("Confusion matrix must be square")
    if confusion.index.astype(str).tolist() != confusion.columns.astype(str).tolist():
        raise ValueError("Confusion-matrix row and column labels do not match")

    values = confusion.to_numpy(dtype=float)
    true_positive = np.diag(values)
    support = values.sum(axis=1)
    predicted = values.sum(axis=0)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    return pd.DataFrame(
        {
            "class": confusion.index.astype(str),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )


def save_baseline_tables(output_dir: Path) -> tuple[Path, Path]:
    """
    Write standardised baseline tables and return their paths.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overall, class_metrics = load_baseline_tables()
    overall_path = output_dir / "assignment2_overall_metrics.csv"
    class_path = output_dir / "assignment2_class_metrics.csv"
    overall.to_csv(overall_path, index=False)
    class_metrics.to_csv(class_path, index=False)
    return overall_path, class_path


def save_final_comparison(
    forest_results: Path, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine the finalised forest with both Assignment 2 baselines.
    """

    forest_results = Path(forest_results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overall, class_metrics = load_baseline_tables()

    forest_summary = _read_json(forest_results / "final_summary.json")
    test_metrics = forest_summary["test_metrics"]
    forest_overall = pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "cv_macro_f1_mean": forest_summary[
                    "selection_cv_macro_f1_mean"
                ],
                "cv_macro_f1_std": forest_summary["selection_cv_macro_f1_std"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_cohen_kappa": test_metrics["cohen_kappa"],
                "test_mcc": test_metrics["mcc"],
            }
        ]
    )
    overall = pd.concat([overall, forest_overall], ignore_index=True)

    forest_confusion = pd.read_csv(
        forest_results / "confusion_matrix.csv", index_col=0
    )
    forest_classes = metrics_from_confusion_matrix(forest_confusion)
    forest_classes.insert(0, "model", "Random Forest")
    class_metrics = pd.concat(
        [class_metrics, forest_classes], ignore_index=True
    )

    expected_support = class_metrics.groupby("model")["support"].sum()
    if not expected_support.eq(expected_support.iloc[0]).all():
        raise ValueError(
            "Models do not have identical test support; comparison is not paired"
        )

    overall.to_csv(output_dir / "model_overall_comparison.csv", index=False)
    class_metrics.to_csv(output_dir / "model_class_comparison.csv", index=False)
    _save_overall_plot(overall, output_dir / "model_overall_comparison.png")
    _save_recall_plot(class_metrics, output_dir / "model_class_recall.png")

    tree = overall.loc[overall["model"].eq("Classification tree")].iloc[0]
    forest = overall.loc[overall["model"].eq("Random Forest")].iloc[0]
    differences = {
        metric: float(forest[metric] - tree[metric])
        for metric in [
            "test_accuracy",
            "test_macro_f1",
            "test_cohen_kappa",
            "test_mcc",
        ]
    }
    (output_dir / "forest_minus_tree.json").write_text(
        json.dumps(differences, indent=2) + "\n"
    )
    return overall, class_metrics


def _save_overall_plot(overall: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        "test_accuracy",
        "test_macro_f1",
        "test_cohen_kappa",
        "test_mcc",
    ]
    long = overall.melt(
        id_vars="model", value_vars=metrics, var_name="metric", value_name="score"
    )
    long["metric"] = long["metric"].map(
        {
            "test_accuracy": "Accuracy",
            "test_macro_f1": "Macro-F1",
            "test_cohen_kappa": "Cohen's kappa",
            "test_mcc": "MCC",
        }
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=long, x="metric", y="score", hue="model")
    plt.ylim(0, 1)
    plt.xlabel("")
    plt.ylabel("Held-out test score")
    plt.title("Overall model comparison on the common test partition")
    plt.legend(title="Model", loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _save_recall_plot(class_metrics: pd.DataFrame, output_path: Path) -> None:
    plot_data = class_metrics.copy()
    plot_data["class"] = plot_data["class"].astype(int)
    plt.figure(figsize=(11, 6))
    sns.barplot(data=plot_data, x="class", y="recall", hue="model")
    plt.ylim(0, 1)
    plt.xlabel("Cyberattack class")
    plt.ylabel("Held-out test recall")
    plt.title("Class-specific recall on the common test partition")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Assignment 2 result not found: {path}")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--forest-results", type=Path, default=DEFAULT_FOREST_RESULTS
    )
    parser.add_argument("--final-comparison", action="store_true")
    args = parser.parse_args()
    if args.final_comparison:
        overall, _ = save_final_comparison(args.forest_results, args.output_dir)
        print(overall.to_string(index=False))
        print(f"Saved final comparison artifacts to {args.output_dir}")
    else:
        overall_path, class_path = save_baseline_tables(args.output_dir)
        print(f"Saved overall baseline metrics to {overall_path}")
        print(f"Saved class-level baseline metrics to {class_path}")


if __name__ == "__main__":
    main()

