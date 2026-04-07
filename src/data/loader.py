import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw(config: dict) -> pd.DataFrame:
    dataset = config["dataset"]
    path = config["paths"][dataset]["raw"]
    df = pd.read_csv(path)
    _validate(df, config)
    return df


def load_processed(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = config["dataset"]
    train_path = config["paths"][dataset]["processed_train"]
    test_path = config["paths"][dataset]["processed_test"]
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def load_glossary(config: dict) -> dict:
    path = config["paths"]["feature_glossary"]
    with open(path) as f:
        entries = yaml.safe_load(f) or []
    return {e["name"]: e for e in entries} if entries else {}


def _validate(df: pd.DataFrame, config: dict) -> None:
    target = config["preprocessing"]["target_column"]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")
    if df.empty:
        raise ValueError("Loaded dataframe is empty.")
