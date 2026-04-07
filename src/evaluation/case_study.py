import pandas as pd
from pathlib import Path

from src.explain.templates import get_template, render_cf_narrative, BENIGN_LABEL


def render_brief(
    alert: pd.Series,
    shap_values: list[tuple[str, float]],
    cf_changes: list[tuple[str, float]],
    predicted_class: str,
    confidence: float,
    label_map: dict[int, str],
    config: dict,
    alert_id: str = "ALERT-001",
) -> str:
    template = get_template(predicted_class)
    cf_narrative = render_cf_narrative(template, cf_changes)

    top_shap = "\n".join(
        f"  - {feat}: SHAP={val:+.4f}" for feat, val in shap_values[:5]
    )
    cf_section = "\n".join(
        f"  - {feat}: {delta:+.4f}" for feat, delta in cf_changes
    ) or "  No changes identified."

    brief = f"""
========================================
ALERT BRIEF — {alert_id}
========================================
Classification : {predicted_class} (confidence: {confidence:.1%})
MITRE Technique: {template['mitre_id']} — {template['mitre_name']}

SUMMARY
{template['summary']}

TOP SHAP CONTRIBUTORS (local explanation)
{top_shap}

COUNTERFACTUAL EXPLANATION
{cf_narrative}

  Feature changes that would flip verdict to '{BENIGN_LABEL}':
{cf_section}

RECOMMENDED ANALYST ACTION
{template['analyst_action']}
========================================
""".strip()

    _save_brief(brief, config, alert_id)
    return brief


def _save_brief(brief: str, config: dict, alert_id: str) -> None:
    out_dir = Path(config["paths"]["case_studies_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{alert_id}.md"
    path.write_text(brief)
    print(f"Saved brief: {path}")
