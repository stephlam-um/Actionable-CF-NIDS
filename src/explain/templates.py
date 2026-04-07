import yaml
from pathlib import Path

# Attack-class templates keyed by class label string.
# Populate/extend per dataset. Values in braces are filled at render time.
TEMPLATES: dict[str, dict] = {
    "DDoS": {
        "summary": "Volumetric denial-of-service traffic detected — high byte/packet rates targeting destination.",
        "cf_narrative": "Model decision hinges on: {changed_features}. Reducing {top_feature} by {top_delta} would shift classification to benign.",
        "mitre_id": "T1498",
        "mitre_name": "Network Denial of Service",
        "analyst_action": "Confirm source IPs. Check for amplification patterns. Consider rate-limiting or blackholing.",
    },
    "PortScan": {
        "summary": "Reconnaissance activity detected — systematic probing of multiple destination ports.",
        "cf_narrative": "Model decision hinges on: {changed_features}. {cf_detail}",
        "mitre_id": "T1046",
        "mitre_name": "Network Service Discovery",
        "analyst_action": "Identify scanning source. Check for follow-up exploitation attempts. Block source if external.",
    },
    "BruteForce": {
        "summary": "Brute-force authentication attack detected — repeated failed login attempts.",
        "cf_narrative": "Model decision hinges on: {changed_features}. {cf_detail}",
        "mitre_id": "T1110",
        "mitre_name": "Brute Force",
        "analyst_action": "Lock affected accounts. Enforce MFA. Review authentication logs for compromise.",
    },
    "DoS": {
        "summary": "Denial-of-service traffic detected — high-rate flow exhausting target resources.",
        "cf_narrative": "Model decision hinges on: {changed_features}. {cf_detail}",
        "mitre_id": "T1499",
        "mitre_name": "Endpoint Denial of Service",
        "analyst_action": "Isolate affected host. Apply upstream rate limiting. Engage upstream ISP if volumetric.",
    },
    "Botnet": {
        "summary": "Botnet C2 communication pattern detected — periodic beaconing behavior.",
        "cf_narrative": "Model decision hinges on: {changed_features}. {cf_detail}",
        "mitre_id": "T1071",
        "mitre_name": "Application Layer Protocol (C2)",
        "analyst_action": "Isolate host. Capture full PCAP for C2 IOC extraction. Check for lateral movement.",
    },
    "Heartbleed": {
        "summary": "Heartbleed exploitation attempt detected — malformed TLS heartbeat request.",
        "cf_narrative": "Model decision hinges on: {changed_features}. {cf_detail}",
        "mitre_id": "T1190",
        "mitre_name": "Exploit Public-Facing Application",
        "analyst_action": "Patch OpenSSL immediately. Rotate all TLS certificates and private keys on affected host.",
    },
    "WebAttack": {
        "summary": "Web application attack detected — injection or traversal pattern in HTTP flow.",
        "cf_narrative": "Model decision hinges on: {changed_features}. {cf_detail}",
        "mitre_id": "T1190",
        "mitre_name": "Exploit Public-Facing Application",
        "analyst_action": "Review WAF logs. Check application error logs for successful exploitation. Patch if applicable.",
    },
    "Infiltration": {
        "summary": "Infiltration/exfiltration activity detected — unusual outbound data transfer pattern.",
        "cf_narrative": "Model decision hinges on: {changed_features}. {cf_detail}",
        "mitre_id": "T1041",
        "mitre_name": "Exfiltration Over C2 Channel",
        "analyst_action": "Block outbound connection. Identify data scope. Initiate incident response procedure.",
    },
}

BENIGN_LABEL = "Benign"


def get_template(attack_class: str) -> dict:
    if attack_class not in TEMPLATES:
        raise KeyError(f"No template for class '{attack_class}'. Add it to templates.py.")
    return TEMPLATES[attack_class]


def render_cf_narrative(template: dict, changed_features: list[tuple[str, float]]) -> str:
    if not changed_features:
        return "No counterfactual features identified."

    features_str = ", ".join(f"{f} (Δ={d:+.3f})" for f, d in changed_features)
    top_feature, top_delta = changed_features[0]
    cf_detail = f"Changing {top_feature} by {top_delta:+.3f} is the primary lever."

    return template["cf_narrative"].format(
        changed_features=features_str,
        top_feature=top_feature,
        top_delta=f"{top_delta:+.3f}",
        cf_detail=cf_detail,
    )
