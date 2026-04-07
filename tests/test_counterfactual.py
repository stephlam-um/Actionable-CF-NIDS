import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


def _make_cf_result(feature_values: dict, outcome: int) -> MagicMock:
    """Build a mock DiCE CF result with one counterfactual row."""
    cf_df = pd.DataFrame([{**feature_values, "label": outcome}])
    example = MagicMock()
    example.final_cfs_df = cf_df
    result = MagicMock()
    result.cf_examples_list = [example]
    return result


def test_validity_all_valid():
    from src.evaluation.cf_metrics import validity
    cfs = [_make_cf_result({"feat_a": 0.1}, outcome=0) for _ in range(5)]
    assert validity(cfs, target_class=0) == pytest.approx(1.0)


def test_validity_none_valid():
    from src.evaluation.cf_metrics import validity
    cfs = [_make_cf_result({"feat_a": 0.1}, outcome=1) for _ in range(5)]
    assert validity(cfs, target_class=0) == pytest.approx(0.0)


def test_sparsity_no_change():
    from src.evaluation.cf_metrics import sparsity
    orig = pd.DataFrame([{"feat_a": 0.5, "feat_b": 0.5}])
    cfs = [_make_cf_result({"feat_a": 0.5, "feat_b": 0.5}, outcome=0)]
    result = sparsity(cfs, orig)
    assert result == pytest.approx(0.0)


def test_sparsity_one_change():
    from src.evaluation.cf_metrics import sparsity
    orig = pd.DataFrame([{"feat_a": 0.5, "feat_b": 0.5}])
    cfs = [_make_cf_result({"feat_a": 0.1, "feat_b": 0.5}, outcome=0)]
    result = sparsity(cfs, orig)
    assert result == pytest.approx(1.0)


def test_plausibility_in_range():
    from src.evaluation.cf_metrics import plausibility
    ranges = {"feat_a": [0.0, 1.0]}
    cfs = [_make_cf_result({"feat_a": 0.5}, outcome=0)]
    assert plausibility(cfs, ranges) == pytest.approx(1.0)


def test_plausibility_out_of_range():
    from src.evaluation.cf_metrics import plausibility
    ranges = {"feat_a": [0.0, 1.0]}
    cfs = [_make_cf_result({"feat_a": 5.0}, outcome=0)]
    assert plausibility(cfs, ranges) == pytest.approx(0.0)
