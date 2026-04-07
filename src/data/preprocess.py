import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from src.data.loader import load_config, load_raw


def preprocess(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_raw(config)
    cfg = config["preprocessing"]
    target = cfg["target_column"]

    df = _drop_duplicates(df)
    df = _handle_missing(df)
    df, label_encoder = _encode_labels(df, target)
    df, scaler = _scale_features(df, target, cfg["scaler"])

    train, test = train_test_split(
        df,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=df[target],
    )

    _save(train, test, config)
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    print(f"Class distribution:\n{train[target].value_counts()}")
    return train, test


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows.")
    return df


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df)} rows with NaN/inf.")
    return df


def _encode_labels(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return df, le


def _scale_features(
    df: pd.DataFrame, target: str, method: str
) -> tuple[pd.DataFrame, object]:
    feature_cols = [c for c in df.columns if c != target]
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler


def _save(train: pd.DataFrame, test: pd.DataFrame, config: dict) -> None:
    dataset = config["dataset"]
    train_path = config["paths"][dataset]["processed_train"]
    test_path = config["paths"][dataset]["processed_test"]
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"Saved processed data to {train_path} and {test_path}")


if __name__ == "__main__":
    config = load_config()
    preprocess(config)
