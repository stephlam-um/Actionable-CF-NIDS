import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.data.loader import load_config, load_processed


def evaluate(
    model: XGBClassifier,
    config: dict,
    feature_cols: list[str] | None = None,
    tag: str = "full",
) -> dict:
    _, test_df = load_processed(config)
    target = config["preprocessing"]["target_column"]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    if feature_cols is not None:
        X_test = X_test[feature_cols]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    metrics = {
        "tag": tag,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc": roc_auc,
        "per_class": report,
    }

    print(f"\n=== Evaluation [{tag}] ===")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(classification_report(y_test, y_pred))

    _save_confusion_matrix(y_test, y_pred, config, tag)
    return metrics


def _save_confusion_matrix(y_test, y_pred, config: dict, tag: str) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix [{tag}]")
    fig_dir = Path(config["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"confusion_matrix_{tag}.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    from src.model.train import load_model
    config = load_config()
    model = load_model(config, "model_full")
    evaluate(model, config)
