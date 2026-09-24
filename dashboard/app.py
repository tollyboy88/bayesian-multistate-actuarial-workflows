from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"

st.set_page_config(page_title="Bayesian Workflow Laboratory", page_icon="∿", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f5faf9; color: #173b3f; }
    [data-testid="stMetric"] { background: white; border: 1px solid #cfe3df; border-radius: 10px; padding: 14px; }
    h1, h2, h3 { color: #145c63; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_tables():
    return {
        "transition": pd.read_csv(TABLES / "transition_posterior.csv"),
        "scores": pd.read_csv(TABLES / "held_out_model_comparison.csv"),
        "recovery": pd.read_csv(TABLES / "parameter_recovery_summary.csv"),
        "controls": pd.read_csv(TABLES / "control_optimisation.csv", dtype={"placement": str}),
        "recursion": pd.read_csv(TABLES / "recursion_validation.csv"),
    }


tables = load_tables()
transition = tables["transition"]

st.title("Bayesian Workflow Laboratory")
st.caption("Posterior uncertainty, validation and stage-level control placement for automated actuarial workflows")

scores = tables["scores"].set_index("model")
controls = tables["controls"]
selected = controls.loc[controls.selected.astype(str).str.lower().isin(["true", "1"])].iloc[0]
metrics = st.columns(4)
metrics[0].metric("Transition observations", "6,144")
metrics[1].metric("Hierarchical log score", f"{scores.loc['hierarchical', 'log_score']:.3f}", f"{scores.loc['global', 'log_score'] - scores.loc['hierarchical', 'log_score']:.3f} better")
metrics[2].metric("Selected placement", str(selected.placement).zfill(6))
metrics[3].metric("Estimated P(active error)", f"{selected.posterior_mean_silent_probability:.4f}")

tab_transition, tab_decision, tab_validation = st.tabs(["Transition posterior", "Control decision", "Validation"])

with tab_transition:
    left, right, third = st.columns(3)
    stage = left.selectbox("Stage", sorted(transition.stage.unique(), key=lambda value: int(value[1])))
    fault = right.selectbox("Fault class", sorted(transition.fault_type.unique()))
    topology = third.selectbox("Control topology", sorted(transition.topology.unique()))
    view = transition.loc[
        (transition.stage == stage)
        & (transition.fault_type == fault)
        & (transition.topology == topology)
        & (transition.state_before == "E")
    ].copy()
    fig = go.Figure()
    fig.add_bar(
        x=view.state_after,
        y=view.posterior_mean,
        error_y={"type": "data", "symmetric": False, "array": view.ci_95 - view.posterior_mean, "arrayminus": view.posterior_mean - view.ci_05},
        marker_color=["#2a9d8f", "#e76f51", "#4b7bec", "#6c757d"],
    )
    fig.update_layout(title="Destination-state probability from an active error", yaxis_title="Posterior probability", xaxis_title="Destination state", yaxis_range=[0, 1], paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(view[["state_after", "posterior_mean", "ci_05", "ci_95", "cell_n"]], width="stretch", hide_index=True)

with tab_decision:
    feasible = controls.loc[controls.feasible.astype(str).str.lower().isin(["true", "1"])].copy()
    feasible["placement"] = feasible.placement.str.zfill(6)
    fig = px.scatter(feasible, x="annual_cost", y="tail_risk_index", color="pareto", hover_data=["placement", "posterior_mean_silent_probability"], color_discrete_map={True: "#e76f51", False: "#87b8b2"}, log_y=True)
    fig.update_layout(title="Cost-risk frontier across feasible stage-control placements", paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(feasible.sort_values(["objective", "annual_cost"]).head(12), width="stretch", hide_index=True)

with tab_validation:
    left, right = st.columns(2)
    score_long = tables["scores"].melt(id_vars=["model", "n"], value_vars=["log_score", "brier_score"], var_name="score", value_name="value")
    left.plotly_chart(px.bar(score_long, x="model", y="value", color="score", barmode="group", title="Held-out predictive scores; lower is better", color_discrete_sequence=["#145c63", "#e9a23b"]), width="stretch")
    recovery = tables["recovery"].melt(id_vars=["topology"], value_vars=["coverage_90", "coverage_95"], var_name="interval", value_name="coverage")
    right.plotly_chart(px.bar(recovery, x="topology", y="coverage", color="interval", barmode="group", title="Known-parameter interval coverage", color_discrete_sequence=["#2a9d8f", "#4b7bec"], range_y=[0, 1]), width="stretch")
    st.subheader("Analytic recursion check")
    st.dataframe(tables["recursion"], width="stretch", hide_index=True)

st.caption("Shared data archive: https://doi.org/10.5281/zenodo.22821051 · Controlled-study evidence, not production failure telemetry")
