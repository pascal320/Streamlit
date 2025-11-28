import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mental Health & Social Media Dashboard", layout="wide")
st.title("Mental Health & Social Media Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("mental_health_social_media_dataset.csv")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

df = load_data()

# ---------------- Sidebar ----------------
st.sidebar.header("Filters")
platforms = ["All"] + sorted(df["platform"].dropna().unique())
selected_platform = st.sidebar.selectbox("Platform", platforms)

metric_mapping = {
    "Anxiety Level": "anxiety_level",
    "Stress Level": "stress_level",
    "Mood Level": "mood_level",
}
metric_label = st.sidebar.selectbox("Metric for Social Media Time", list(metric_mapping.keys()))
selected_metric = metric_mapping[metric_label]

data = df.copy() if selected_platform == "All" else df[df["platform"] == selected_platform]

# ---------------- Data Preview ----------------
st.subheader("Data Preview")
st.dataframe(df.head())

# ---------------- 1. Platform Comparison ----------------
st.header("1. Platform Comparison")

platform_stats = (
    df.groupby("platform")[["anxiety_level", "stress_level", "mood_level"]]
    .mean()
    .reset_index()
    .sort_values("anxiety_level", ascending=False)
)

fig_platform = go.Figure()
fig_platform.add_trace(go.Bar(x=platform_stats["platform"], y=platform_stats["anxiety_level"], name="Avg Anxiety"))
fig_platform.add_trace(go.Bar(x=platform_stats["platform"], y=platform_stats["stress_level"], name="Avg Stress"))
fig_platform.add_trace(go.Bar(x=platform_stats["platform"], y=platform_stats["mood_level"], name="Avg Mood"))

fig_platform.update_layout(
    barmode="group",
    xaxis_title="Platform",
    yaxis_title="Average Score",
    title="Average Anxiety, Stress & Mood by Platform",
)

st.plotly_chart(fig_platform, width="stretch")

worst = platform_stats.iloc[0]
best = platform_stats.iloc[-1]
st.markdown(
    f"**Observation:** {worst['platform']} shows the highest anxiety and lowest mood. "
    f"{best['platform']} shows the lowest anxiety and highest mood."
)

# ---------------- 2. Mental State Distribution ----------------
st.header("2. Mental State Distribution")

counts = data["mental_state"].value_counts().sort_index()
fig_state = go.Figure(data=[go.Bar(x=counts.index, y=counts.values)])
fig_state.update_layout(
    xaxis_title="Mental State",
    yaxis_title="Count",
    title=f"Mental State Distribution ({selected_platform})",
)

st.plotly_chart(fig_state, width="stretch")

# ---------------- 3. Social Media Time vs Metric ----------------
st.header(f"3. Social Media Time vs {metric_label}")

fig_scatter = go.Figure()
fig_scatter.add_trace(
    go.Scatter(
        x=data["social_media_time_min"],
        y=data[selected_metric],
        mode="markers",
        opacity=0.5,
        marker=dict(size=6),
        name=metric_label,
    )
)
fig_scatter.update_layout(
    xaxis_title="Social Media Time (min)",
    yaxis_title=metric_label,
    title=f"Social Media Time vs {metric_label}",
)

st.plotly_chart(fig_scatter, width="stretch")

# ---------------- 4. Sleep & Activity vs Mood ----------------
st.header("4. Sleep & Physical Activity vs Mood")

def sleep_cat(x):
    if x < 6: return "<6h"
    elif x < 7: return "6–7h"
    elif x < 8: return "7–8h"
    return "≥8h"

def activity_cat(x):
    if x < 30: return "<30min"
    elif x < 60: return "30–59min"
    elif x < 90: return "60–89min"
    return "≥90min"

temp = df.copy()
temp["sleep_cat"] = temp["sleep_hours"].apply(sleep_cat)
temp["activity_cat"] = temp["physical_activity_min"].apply(activity_cat)

sleep_stats = temp.groupby("sleep_cat")["mood_level"].mean()
activity_stats = temp.groupby("activity_cat")["mood_level"].mean()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Mood by Sleep")
    fig_sleep = go.Figure([go.Bar(x=sleep_stats.index, y=sleep_stats.values)])
    fig_sleep.update_layout(xaxis_title="Sleep", yaxis_title="Mood")
    st.plotly_chart(fig_sleep, width="stretch")

with col2:
    st.subheader("Mood by Physical Activity")
    fig_act = go.Figure([go.Bar(x=activity_stats.index, y=activity_stats.values)])
    fig_act.update_layout(xaxis_title="Activity", yaxis_title="Mood")
    st.plotly_chart(fig_act, width="stretch")

# ---------------- 5. Correlation Heatmap (Seaborn) ----------------
st.header("5. Interaction & Mental Health Correlations")

subset = df[
    [
        "negative_interactions_count",
        "positive_interactions_count",
        "anxiety_level",
        "stress_level",
        "mood_level",
    ]
]

corr = subset.corr()

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    corr,
    annot=True,
    cmap="RdBu_r",
    linewidths=.5,
    square=True,
    fmt=".2f",
    ax=ax
)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.title("Correlation Heatmap", fontsize=14)

st.pyplot(fig)