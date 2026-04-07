import dice_ml
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

from src.data.loader import load_config, load_processed, load_glossary


def build_dice_explainer(
    model: XGBClassifier,
    train_df: pd.DataFrame,
    continuous_cols: list[str],
    target: str,
    method: str = "random",
) -> dice_ml.Dice:
    data = dice_ml.Data(
        dataframe=train_df,
        continuous_features=continuous_cols,
        outcome_name=target,
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    return dice_ml.Dice(data, dice_model, method=method)


def generate_counterfactuals(
    explainer: dice_ml.Dice,
    query_df: pd.DataFrame,
    n_cfs: int = 3,
    target_class: int | str = 0,
    feature_ranges: dict | None = None,
    immutable_features: list[str] | None = None,
) -> list[dice_ml.counterfactual_explanations.CounterfactualExplanations]:
    kwargs: dict = {"total_CFs": n_cfs, "desired_class": target_class}
    if feature_ranges:
        kwargs["permitted_range"] = feature_ranges
    if immutable_features:
        kwargs["features_to_vary"] = [
            c for c in query_df.columns if c not in immutable_features
        ]

    results = []
    for i, row in query_df.iterrows():
        try:
            cf = explainer.generate_counterfactuals(
                row.to_frame().T, **kwargs
            )
            results.append(cf)
        except Exception as e:
            print(f"CF generation failed for index {i}: {e}")
            results.append(None)
    return results


def build_feature_ranges(glossary: dict, feature_cols: list[str]) -> dict:
    ranges = {}
    for feat in feature_cols:
        if feat in glossary and glossary[feat].get("type") == "continuous":
            entry = glossary[feat]
            ranges[feat] = [entry["min"], entry["max"]]
    return ranges


def run_counterfactual_generation(
    model: XGBClassifier,
    config: dict,
    feature_cols: list[str],
) -> list:
    train_df, test_df = load_processed(config)
    glossary = load_glossary(config)
    target = config["preprocessing"]["target_column"]
    cfg_cf = config["counterfactuals"]

    X_train = train_df[feature_cols + [target]]
    continuous_cols = [
        c for c in feature_cols
        if glossary.get(c, {}).get("type", "continuous") == "continuous"
    ]
    feature_ranges = build_feature_ranges(glossary, feature_cols)

    explainer = build_dice_explainer(
        model, X_train, continuous_cols, target, method=cfg_cf["method"]
    )

    # Sample alerts per class
    samples = (
        test_df[test_df[target] != 0]  # exclude benign
        .groupby(target)
        .apply(lambda g: g.sample(min(len(g), cfg_cf["n_samples_per_class"]), random_state=42))
        .reset_index(drop=True)
    )

    query_df = samples[feature_cols]
    cfs = generate_counterfactuals(
        explainer,
        query_df,
        n_cfs=cfg_cf["n_cfs_per_sample"],
        target_class=cfg_cf["target_class"],
        feature_ranges=feature_ranges or None,
    )
    print(f"Generated CFs for {len(cfs)} alerts.")
    return cfs, samples


if __name__ == "__main__":
    from src.model.train import load_model
    config = load_config()
    # Load reduced model — update model_name and feature_cols as needed
    model = load_model(config, "model_full")
    _, test_df = load_processed(config)
    target = config["preprocessing"]["target_column"]
    feature_cols = [c for c in test_df.columns if c != target]
    run_counterfactual_generation(model, config, feature_cols)
