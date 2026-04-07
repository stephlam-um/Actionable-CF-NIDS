import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.data.loader import load_config, load_processed
from src.explain.shap_analysis import run_shap_analysis, global_importance
from src.model.train import train, load_model


def sweep(
    importance_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    top_k_values = config["feature_selection"]["top_k_values"]
    _, test_df = load_processed(config)
    target = config["preprocessing"]["target_column"]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    ranked_features = importance_df["feature"].tolist()
    records = []

    for k in top_k_values:
        features = ranked_features[:k]
        model_k = train(config, feature_cols=features, model_name=f"model_top{k}")
        y_pred = model_k.predict(X_test[features])
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        records.append({"top_k": k, "macro_f1": macro_f1, "weighted_f1": weighted_f1})
        print(f"top-{k:2d}: macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}")

    results = pd.DataFrame(records)
    _save_results(results, config)
    _plot_curve(results, config)
    return results


def select_best_k(sweep_results: pd.DataFrame, baseline_macro_f1: float, config: dict) -> int:
    threshold = config["feature_selection"]["f1_drop_threshold"]
    candidates = sweep_results[sweep_results["macro_f1"] >= baseline_macro_f1 - threshold]
    best_k = int(candidates["top_k"].min())
    print(f"Selected top-{best_k} features (F1 drop < {threshold:.0%} from baseline {baseline_macro_f1:.4f})")
    return best_k


def _plot_curve(results: pd.DataFrame, config: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(results["top_k"], results["macro_f1"], marker="o", label="Macro F1")
    ax.plot(results["top_k"], results["weighted_f1"], marker="s", label="Weighted F1")
    ax.set_xlabel("Number of features")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 vs Feature Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_dir = Path(config["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "f1_vs_feature_count.png", bbox_inches="tight")
    plt.close(fig)


def _save_results(results: pd.DataFrame, config: dict) -> None:
    tables_dir = Path(config["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(tables_dir / "feature_sweep_results.csv", index=False)


if __name__ == "__main__":
    config = load_config()
    model = load_model(config, "model_full")
    _, importance = run_shap_analysis(model, config)
    sweep(importance, config)
