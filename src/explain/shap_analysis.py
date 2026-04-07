import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier

from src.data.loader import load_config, load_processed


def compute_shap_values(
    model: XGBClassifier,
    X_test: pd.DataFrame,
) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return shap_values  # shape: (n_samples, n_features) or (n_classes, n_samples, n_features)


def global_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    if shap_values.ndim == 3:
        # multiclass: average across classes
        mean_abs = np.mean(np.abs(shap_values), axis=(0, 1))
    else:
        mean_abs = np.mean(np.abs(shap_values), axis=0)

    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def per_class_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    class_labels: list[str],
) -> dict[str, pd.DataFrame]:
    # shap_values: (n_classes, n_samples, n_features)
    result = {}
    for i, label in enumerate(class_labels):
        mean_abs = np.mean(np.abs(shap_values[i]), axis=0)
        df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        result[label] = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return result


def plot_global_importance(importance_df: pd.DataFrame, config: dict, top_k: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(8, top_k * 0.35 + 1))
    top = importance_df.head(top_k)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Global Feature Importance (top {top_k})")
    fig_dir = Path(config["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "shap_global_importance.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SHAP importance plot.")


def run_shap_analysis(model: XGBClassifier, config: dict) -> tuple[np.ndarray, pd.DataFrame]:
    _, test_df = load_processed(config)
    target = config["preprocessing"]["target_column"]
    X_test = test_df.drop(columns=[target])

    shap_values = compute_shap_values(model, X_test)
    importance = global_importance(
        shap_values if shap_values.ndim == 2 else shap_values,
        list(X_test.columns),
    )
    plot_global_importance(importance, config)

    tables_dir = Path(config["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(tables_dir / "shap_global_importance.csv", index=False)
    print(importance.head(20).to_string(index=False))
    return shap_values, importance


if __name__ == "__main__":
    from src.model.train import load_model
    config = load_config()
    model = load_model(config, "model_full")
    run_shap_analysis(model, config)
