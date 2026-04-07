import numpy as np
import pandas as pd
from pathlib import Path


def validity(cf_results: list, target_class: int | str) -> float:
    """Fraction of CFs where the model actually predicts target_class."""
    valid = 0
    total = 0
    for cf in cf_results:
        if cf is None:
            continue
        try:
            cf_df = cf.cf_examples_list[0].final_cfs_df
            if cf_df is not None and len(cf_df) > 0:
                # DiCE stores the predicted class in the outcome column
                outcome_col = cf_df.columns[-1]
                valid += (cf_df[outcome_col] == target_class).sum()
                total += len(cf_df)
        except Exception:
            continue
    return valid / total if total > 0 else 0.0


def proximity(
    cf_results: list,
    originals: pd.DataFrame,
    feature_ranges: dict[str, list],
) -> float:
    """Mean normalized L1 distance between original and CF."""
    distances = []
    for i, cf in enumerate(cf_results):
        if cf is None:
            continue
        try:
            cf_df = cf.cf_examples_list[0].final_cfs_df
            if cf_df is None or len(cf_df) == 0:
                continue
            orig = originals.iloc[i]
            feature_cols = [c for c in orig.index if c in feature_ranges]
            for _, cf_row in cf_df[feature_cols].iterrows():
                dists = []
                for col in feature_cols:
                    r = feature_ranges[col]
                    span = r[1] - r[0]
                    if span > 0:
                        dists.append(abs(orig[col] - cf_row[col]) / span)
                if dists:
                    distances.append(np.mean(dists))
        except Exception:
            continue
    return float(np.mean(distances)) if distances else float("nan")


def sparsity(cf_results: list, originals: pd.DataFrame) -> float:
    """Mean number of features changed per CF."""
    changed_counts = []
    for i, cf in enumerate(cf_results):
        if cf is None:
            continue
        try:
            cf_df = cf.cf_examples_list[0].final_cfs_df
            if cf_df is None or len(cf_df) == 0:
                continue
            orig = originals.iloc[i]
            feature_cols = [c for c in orig.index if c in cf_df.columns]
            for _, cf_row in cf_df[feature_cols].iterrows():
                n_changed = (cf_row != orig[feature_cols]).sum()
                changed_counts.append(n_changed)
        except Exception:
            continue
    return float(np.mean(changed_counts)) if changed_counts else float("nan")


def plausibility(cf_results: list, feature_ranges: dict[str, list]) -> float:
    """Fraction of CFs where all feature values fall within observed training range."""
    plausible = 0
    total = 0
    for cf in cf_results:
        if cf is None:
            continue
        try:
            cf_df = cf.cf_examples_list[0].final_cfs_df
            if cf_df is None or len(cf_df) == 0:
                continue
            for _, row in cf_df.iterrows():
                in_range = all(
                    feature_ranges[col][0] <= row[col] <= feature_ranges[col][1]
                    for col in feature_ranges
                    if col in row.index
                )
                plausible += int(in_range)
                total += 1
        except Exception:
            continue
    return plausible / total if total > 0 else float("nan")


def compute_all_metrics(
    cf_results: list,
    originals: pd.DataFrame,
    feature_ranges: dict[str, list],
    target_class: int | str,
    config: dict,
    tag: str = "",
) -> dict:
    metrics = {
        "tag": tag,
        "validity": validity(cf_results, target_class),
        "proximity": proximity(cf_results, originals, feature_ranges),
        "sparsity": sparsity(cf_results, originals),
        "plausibility": plausibility(cf_results, feature_ranges),
    }

    print(f"\n=== CF Metrics [{tag}] ===")
    for k, v in metrics.items():
        if k != "tag":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    tables_dir = Path(config["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(tables_dir / f"cf_metrics_{tag}.csv", index=False)
    return metrics
