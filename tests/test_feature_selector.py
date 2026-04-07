import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


def _make_importance(features: list[str]) -> pd.DataFrame:
    scores = np.linspace(1.0, 0.1, len(features))
    return pd.DataFrame({"feature": features, "mean_abs_shap": scores})


def test_select_best_k_picks_minimum():
    from src.explain.feature_selector import select_best_k

    sweep = pd.DataFrame({
        "top_k": [5, 10, 15, 20],
        "macro_f1": [0.88, 0.91, 0.92, 0.92],
        "weighted_f1": [0.89, 0.91, 0.92, 0.92],
    })
    config = {"feature_selection": {"f1_drop_threshold": 0.03}}
    best = select_best_k(sweep, baseline_macro_f1=0.92, config=config)
    # 0.92 - 0.03 = 0.89 → top_k=10 is the first with f1 >= 0.89
    assert best == 10


def test_select_best_k_respects_threshold():
    from src.explain.feature_selector import select_best_k

    sweep = pd.DataFrame({
        "top_k": [5, 10, 15],
        "macro_f1": [0.80, 0.85, 0.90],
        "weighted_f1": [0.80, 0.85, 0.90],
    })
    config = {"feature_selection": {"f1_drop_threshold": 0.02}}
    # baseline=0.90, threshold=0.02, need f1 >= 0.88 → top_k=15
    best = select_best_k(sweep, baseline_macro_f1=0.90, config=config)
    assert best == 15
