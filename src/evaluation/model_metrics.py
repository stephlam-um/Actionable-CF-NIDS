import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from src.data.loader import load_config, load_processed


def compute_metrics(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tag: str = "",
) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    return {
        "tag": tag,
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"),
        "per_class_f1": f1_score(y_test, y_pred, average=None).tolist(),
    }


def compare_models(results: list[dict], config: dict) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "model": r["tag"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
            "macro_precision": r["macro_precision"],
            "macro_recall": r["macro_recall"],
            "roc_auc": r["roc_auc"],
        })
    df = pd.DataFrame(rows)

    tables_dir = Path(config["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "model_comparison.csv", index=False)
    print(df.to_string(index=False))
    return df


def plot_f1_comparison(comparison_df: pd.DataFrame, config: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(comparison_df["model"], comparison_df["macro_f1"])
    ax.set_ylabel("Macro F1")
    ax.set_title("Model Comparison: Macro F1")
    ax.set_ylim(0, 1)
    fig_dir = Path(config["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "model_f1_comparison.png", bbox_inches="tight")
    plt.close(fig)
