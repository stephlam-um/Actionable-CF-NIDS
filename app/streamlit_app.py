import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Actionable-CF-NIDS", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar — config / dataset selection
# ---------------------------------------------------------------------------
st.sidebar.title("Actionable-CF-NIDS")
page = st.sidebar.radio("View", ["Feature Pruning", "Alert Explorer", "Metrics Dashboard"])


# ---------------------------------------------------------------------------
# Helpers — lazy load config + artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def get_config():
    from src.data.loader import load_config
    return load_config()


@st.cache_resource
def get_data(config):
    from src.data.loader import load_processed
    return load_processed(config)


@st.cache_resource
def get_model(config, name="model_full"):
    from src.model.train import load_model
    return load_model(config, name)


@st.cache_data
def get_shap_importance(config):
    from pathlib import Path
    import pandas as pd
    path = Path(config["paths"]["tables_dir"]) / "shap_global_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def get_sweep_results(config):
    from pathlib import Path
    path = Path(config["paths"]["tables_dir"]) / "feature_sweep_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ---------------------------------------------------------------------------
# Page: Feature Pruning
# ---------------------------------------------------------------------------
if page == "Feature Pruning":
    st.title("Feature Pruning via SHAP")
    config = get_config()

    importance = get_shap_importance(config)
    sweep = get_sweep_results(config)

    if importance is None or sweep is None:
        st.warning("Run SHAP analysis and feature selector first (`python -m src.explain.shap_analysis` and `python -m src.explain.feature_selector`).")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Global Feature Importance")
            top_k = st.slider("Show top N features", 5, len(importance), 20)
            top = importance.head(top_k)
            fig, ax = plt.subplots(figsize=(5, top_k * 0.35 + 1))
            ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
            ax.set_xlabel("Mean |SHAP value|")
            st.pyplot(fig)

        with col2:
            st.subheader("F1 vs Feature Count")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.plot(sweep["top_k"], sweep["macro_f1"], marker="o", label="Macro F1")
            ax2.plot(sweep["top_k"], sweep["weighted_f1"], marker="s", label="Weighted F1")
            ax2.set_xlabel("Number of features")
            ax2.set_ylabel("F1 Score")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            st.dataframe(sweep)


# ---------------------------------------------------------------------------
# Page: Alert Explorer
# ---------------------------------------------------------------------------
elif page == "Alert Explorer":
    st.title("Alert Explorer")
    config = get_config()

    try:
        _, test_df = get_data(config)
        target = config["preprocessing"]["target_column"]

        attack_classes = test_df[test_df[target] != 0][target].unique().tolist()
        selected_class = st.selectbox("Filter by attack class", ["All"] + [str(c) for c in attack_classes])

        if selected_class != "All":
            pool = test_df[test_df[target] == int(selected_class)]
        else:
            pool = test_df[test_df[target] != 0]

        idx = st.selectbox("Select alert index", pool.index.tolist()[:100])
        alert = pool.loc[idx]
        st.subheader("Alert Feature Values")
        st.dataframe(alert.to_frame().T)

        st.info("Connect SHAP waterfall plot and CF diff by wiring up the model and explainer here once artifacts are generated.")
    except Exception as e:
        st.warning(f"Could not load data: {e}. Run preprocessing first.")


# ---------------------------------------------------------------------------
# Page: Metrics Dashboard
# ---------------------------------------------------------------------------
elif page == "Metrics Dashboard":
    st.title("Metrics Dashboard")
    config = get_config()
    tables_dir = config["paths"]["tables_dir"]

    from pathlib import Path
    tables = list(Path(tables_dir).glob("*.csv")) if Path(tables_dir).exists() else []

    if not tables:
        st.warning("No results tables found. Run the full pipeline first.")
    else:
        selected = st.selectbox("Table", [t.name for t in tables])
        df = pd.read_csv(tables_dir + "/" + selected)
        st.dataframe(df)

        numeric_cols = df.select_dtypes("number").columns.tolist()
        if numeric_cols and "model" in df.columns:
            metric = st.selectbox("Plot metric", numeric_cols)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(df["model"], df[metric])
            ax.set_ylabel(metric)
            ax.set_title(metric)
            st.pyplot(fig)
