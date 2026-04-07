import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

from src.data.loader import load_config, load_processed


def train(
    config: dict,
    feature_cols: list[str] | None = None,
    model_name: str = "model_full",
) -> XGBClassifier:
    train_df, _ = load_processed(config)
    target = config["preprocessing"]["target_column"]

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    if feature_cols is not None:
        X_train = X_train[feature_cols]

    n_classes = y_train.nunique()
    params = config["model"]

    model = XGBClassifier(
        objective=params["objective"],
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=params["random_state"],
        n_jobs=params["n_jobs"],
        num_class=n_classes,
    )

    sample_weights = _compute_sample_weights(y_train)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    _save_model(model, config, model_name)
    print(f"Model '{model_name}' trained on {X_train.shape[1]} features.")
    return model


def _compute_sample_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    weights = y.map(lambda c: 1.0 / counts[c])
    return weights.to_numpy()


def _save_model(model: XGBClassifier, config: dict, name: str) -> None:
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"{name}.joblib")
    print(f"Saved model to {model_dir / name}.joblib")


def load_model(config: dict, model_name: str = "model_full") -> XGBClassifier:
    model_dir = Path(config["paths"]["model_dir"])
    return joblib.load(model_dir / f"{model_name}.joblib")


if __name__ == "__main__":
    config = load_config()
    train(config)
