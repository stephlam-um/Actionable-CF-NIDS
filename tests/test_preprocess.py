import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


MOCK_CONFIG = {
    "dataset": "roedunet",
    "paths": {
        "roedunet": {
            "raw": "data/raw/roedunet.csv",
            "processed_train": "data/processed/roedunet_train.csv",
            "processed_test": "data/processed/roedunet_test.csv",
        },
        "feature_glossary": "data/feature_glossary.yaml",
    },
    "preprocessing": {
        "target_column": "label",
        "test_size": 0.2,
        "random_state": 42,
        "scaler": "standard",
    },
}


def _make_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feat_a": rng.random(n),
        "feat_b": rng.random(n),
        "feat_c": rng.integers(0, 5, n).astype(float),
        "label": rng.choice(["Benign", "DDoS", "PortScan"], n),
    })


def test_drop_duplicates():
    from src.data.preprocess import _drop_duplicates
    df = _make_df(50)
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    result = _drop_duplicates(df)
    assert len(result) == 50


def test_handle_missing():
    from src.data.preprocess import _handle_missing
    df = _make_df(20)
    df.loc[0, "feat_a"] = np.nan
    df.loc[1, "feat_b"] = np.inf
    result = _handle_missing(df)
    assert result.isna().sum().sum() == 0
    assert not np.isinf(result.select_dtypes("number").values).any()


def test_encode_labels():
    from src.data.preprocess import _encode_labels
    df = _make_df(30)
    result, le = _encode_labels(df, "label")
    assert result["label"].dtype in (int, "int64", "int32")
    assert set(le.classes_) == {"Benign", "DDoS", "PortScan"}


def test_scale_features_standard():
    from src.data.preprocess import _scale_features
    df = _make_df(50)
    df["label"] = 0  # already encoded
    result, scaler = _scale_features(df, "label", "standard")
    numeric = result[["feat_a", "feat_b", "feat_c"]]
    assert abs(numeric.mean().mean()) < 0.5  # roughly centered
