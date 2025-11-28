from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = "COV_VAC_UPTAKE_2024.csv"


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    # Nicer Labels für Population groups
    group_labels = {
        "hcw": "Health & care workers",
        "old": "Older adults",
        "ad_chronic": "Adults with chronic conditions",
        "pw": "Pregnant women",
        "all": "All populations",
        "female": "Females",
        "male": "Males",
        "other": "Other target groups",
    }
    df["GROUP_LABEL"] = df["GROUP"].map(group_labels).fillna(df["GROUP"])

    # Administered doses: fehlend -> 0
    df["COVID_VACCINE_ADM_1D"] = df["COVID_VACCINE_ADM_1D"].fillna(0)
    return df


def filter_data(
    df: pd.DataFrame,
    groups: List[str],
    countries: List[str],
    date_range: Tuple[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    start, end = date_range
    mask = (df["DATE"] >= start) & (df["DATE"] <= end)
    if groups:
        mask &= df["GROUP"].isin(groups)
    if countries:
        mask &= df["COUNTRY"].isin(countries)
    return df.loc[mask].copy()


def aggregate_trend(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggregiert für das Trend-Chart.

    Dosen: Summe pro Datum.
    Coverage: populations-gewichteter Mittelwert (wo POPULATION vorhanden ist),
    sonst einfacher Mittelwert über alle Zeilen mit Coverage.
    """
    if metric == "COVID_VACCINE_ADM_1D":
        return df.groupby("DATE", as_index=False)[metric].sum()

    cov = df.dropna(subset=["COVID_VACCINE_COV_1D"]).copy()
    if cov.empty:
        return cov.assign(**{metric: []})[["DATE", metric]]

    # Nur dort gewichten, wo POPULATION bekannt ist
    cov = cov[cov["POPULATION"].notna()]
    if not cov.empty:
        cov["weighted_cov"] = cov["COVID_VACCINE_COV_1D"] * cov["POPULATION"]
        agg = cov.groupby("DATE", as_index=False)[["weighted_cov", "POPULATION"]].sum()
        agg[metric] = agg["weighted_cov"] / agg["POPULATION"]
        return agg[["DATE", metric]]

    # Fallback: einfacher Mittelwert
    return df.groupby("DATE", as_index=False)[metric].mean()


def plot_trend(agg: pd.DataFrame, metric: str) -> px.line:
    titles = {
        "COVID_VACCINE_ADM_1D": "Vaccine doses administered over time",
        "COVID_VACCINE_COV_1D": "Vaccination coverage over time",
    }
    fig = px.line(
        agg,
        x="DATE",
        y=metric,
        title=titles.get(metric, metric),
        height=420,
    )
    fig.update_layout(
        margin=dict(l=60, r=30, t=60, b=40),
        title_x=0.0,
    )
    if metric == "COVID_VACCINE_ADM_1D":
        fig.update_yaxes(title="Doses (1st dose)", tickformat=",")
    else:
        ymax = float(agg[metric].max()) if not agg.empty else 1.0
        fig.update_yaxes(title="Coverage (%)", range=[0, max(5, ymax * 1.1)])
    return fig


def plot_top_countries(df: pd.DataFrame, metric: str, top_n: int):
    """Top-Länder Chart: für Coverage wird die letzte bekannte Coverage pro Land genutzt."""
    if metric == "COVID_VACCINE_ADM_1D":
        country_metric = (
            df.groupby("COUNTRY", as_index=False)["COVID_VACCINE_ADM_1D"].sum()
        )
        value_col = "COVID_VACCINE_ADM_1D"
        title = f"Top {top_n} countries by doses"
        x_label = "Total doses (1st dose)"
    else:
        cov = df.dropna(subset=["COVID_VACCINE_COV_1D"]).copy()
        if cov.empty:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "No coverage data for this selection", ha="center", va="center")
            ax.axis("off")
            return fig
        cov = cov.sort_values("DATE")
        latest = cov.groupby("COUNTRY").tail(1)
        country_metric = latest[["COUNTRY", "COVID_VACCINE_COV_1D"]]
        value_col = "COVID_VACCINE_COV_1D"
        title = f"Top {top_n} countries by coverage"
        x_label = "Coverage (%)"

    top = country_metric.sort_values(value_col, ascending=False).head(top_n)
    top = top.sort_values(value_col)  # kleinste oben, größte unten

    height = min(360, 40 * len(top) + 80)
    fig, ax = plt.subplots(figsize=(6.5, height / 100))
    sns.barplot(data=top, x=value_col, y="COUNTRY", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Country")
    plt.tight_layout()
    return fig


def plot_group_comparison(df: pd.DataFrame, metric: str):
    """Vergleicht Population groups als Zeitreihe."""
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data for selected filters", ha="center", va="center")
        ax.axis("off")
        return fig

    if metric == "COVID_VACCINE_ADM_1D":
        agg = (
            df.groupby(["DATE", "GROUP_LABEL"], as_index=False)["COVID_VACCINE_ADM_1D"]
            .sum()
        )
        y_label = "Doses (1st dose)"
        value_col = "COVID_VACCINE_ADM_1D"
    else:
        cov = df.dropna(subset=["COVID_VACCINE_COV_1D"]).copy()
        if cov.empty:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "No coverage data for this selection", ha="center", va="center")
            ax.axis("off")
            return fig
        agg = (
            cov.groupby(["DATE", "GROUP_LABEL"], as_index=False)[
                "COVID_VACCINE_COV_1D"
            ]
            .mean()
        )
        y_label = "Coverage (%)"
        value_col = "COVID_VACCINE_COV_1D"

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for group, sub in agg.groupby("GROUP_LABEL"):
        sub = sub.sort_values("DATE")
        ax.plot(sub["DATE"], sub[value_col], label=group, linewidth=1.6)
    ax.set_title(
        "Doses by population group" if metric == "COVID_VACCINE_ADM_1D" else "Coverage by population group"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def format_big_number(n: float) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f} B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f} M"
    if n >= 1_000:
        return f"{n/1_000:.1f} K"
    return f"{n:.0f}"


def metric_card(label: str, value: str, small: str = "") -> str:
    small_html = f"<div class='metric-small'>{small}</div>" if small else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {small_html}
    </div>
    """


def main() -> None:
    st.set_page_config(page_title="COVID-19 Vaccine Uptake Dashboard", layout="wide")

    # Light Theme / Layout
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f6f7f9;
        }
        .block-container {
            max-width: none;
            width: 100%;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e5e7eb;
        }
        .dashboard-title {
            font-size: 2rem;
            font-weight: 650;
            color: #1f2937;
            margin-bottom: 0.2rem;
        }
        .dashboard-subtitle {
            font-size: 0.95rem;
            color: #4b5563;
            margin-bottom: 0.6rem;
        }
        .metric-card {
            background-color: #ffffff;
            border-radius: 0.9rem;
            padding: 0.8rem 1rem;
            border: 1px solid #e5e7eb;
            box-shadow: 0 6px 14px rgba(15, 23, 42, 0.05);
        }
        .metric-label {
            font-size: 0.75rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 650;
            margin-top: 0.2rem;
            color: #111827;
        }
        .metric-small {
            font-size: 0.78rem;
            color: #6b7280;
            margin-top: 0.1rem;
        }
        .stMultiSelect [data-baseweb="tag"] {
            background-color: #eef2ff !important;
            color: #111827 !important;
            border-radius: 999px !important;
            border: 1px solid #c7d2fe !important;
        }
        .stMultiSelect [data-baseweb="tag"] svg {
            fill: #4f46e5 !important;
        }
        .stRadio > div > label > div {
            background-color: #f9fafb !important;
            border-radius: 999px !important;
            border: 1px solid #e5e7eb !important;
            padding: 0.25rem 0.7rem !important;
            font-size: 0.85rem;
        }
        .stRadio label[data-baseweb="radio"]:has(input[checked]) > div {
            background-color: #111827 !important;
            border-color: #111827 !important;
            color: #f9fafb !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()

    st.markdown(
        "<div class='dashboard-title'>COVID-19 vaccine uptake dashboard</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='dashboard-subtitle'>Explore the uptake of COVID-19 vaccines by country and target population groups in 2024.</div>",
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("## Filters")

        groups = sorted(df["GROUP"].unique().tolist())
        default_groups = [g for g in ["hcw", "old"] if g in groups]
        selected_groups = st.multiselect(
            "Population groups", groups, default=default_groups
        )

        available_countries = (
            df if not selected_groups else df[df["GROUP"].isin(selected_groups)]
        )
        countries = sorted(available_countries["COUNTRY"].unique().tolist())
        selected_countries = st.multiselect("Countries (optional)", countries)

        st.markdown("### Time")
        min_date, max_date = df["DATE"].min(), df["DATE"].max()
        start_date, end_date = st.slider(
            "Date range",
            min_value=min_date.to_pydatetime(),
            max_value=max_date.to_pydatetime(),
            value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
            format="YYYY-MM-DD",
        )
        date_range = (pd.Timestamp(start_date), pd.Timestamp(end_date))

        st.markdown("### Metric")
        metric = st.radio(
            "Metric",
            options=["COVID_VACCINE_ADM_1D", "COVID_VACCINE_COV_1D"],
            format_func=lambda x: "Doses administered"
            if x == "COVID_VACCINE_ADM_1D"
            else "Coverage (%)",
        )
        top_n = st.number_input(
            "Top N countries", min_value=3, max_value=20, value=10, step=1
        )

    # Daten filtern
    filtered = filter_data(df, selected_groups, selected_countries, date_range)
    if filtered.empty:
        st.warning("No data for these filters.")
        return

    # KPIs
    latest_date = filtered["DATE"].max()
    total_doses = float(filtered["COVID_VACCINE_ADM_1D"].sum())
    mean_cov = float(filtered["COVID_VACCINE_COV_1D"].mean())
    num_countries = int(filtered["COUNTRY"].nunique())

    total_doses_disp = format_big_number(total_doses)
    mean_cov_disp = f"{mean_cov:.1f}%" if not pd.isna(mean_cov) else "N/A"
    latest_date_disp = latest_date.date().isoformat()

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        metric_card("Total doses (1st dose)", total_doses_disp, f"as of {latest_date_disp}"),
        unsafe_allow_html=True,
    )
    col2.markdown(
        metric_card("Mean coverage", mean_cov_disp),
        unsafe_allow_html=True,
    )
    col3.markdown(
        metric_card("Countries", str(num_countries)),
        unsafe_allow_html=True,
    )

    # Kleines Insight-Snippet
    if metric == "COVID_VACCINE_ADM_1D":
        country_metric = (
            filtered.groupby("COUNTRY", as_index=False)["COVID_VACCINE_ADM_1D"].sum()
        )
        if not country_metric.empty:
            top_country = country_metric.sort_values(
                "COVID_VACCINE_ADM_1D", ascending=False
            ).iloc[0]
            st.markdown(
                f"**Insight** – In the current selection, `{top_country['COUNTRY']}` administered the highest number of first doses."
            )
    else:
        cov = filtered.dropna(subset=["COVID_VACCINE_COV_1D"]).copy()
        if not cov.empty:
            cov_latest = cov.sort_values("DATE").groupby("COUNTRY").tail(1)
            top_country = cov_latest.sort_values(
                "COVID_VACCINE_COV_1D", ascending=False
            ).iloc[0]
            st.markdown(
                f"**Insight** – In the current selection, `{top_country['COUNTRY']}` shows the highest latest coverage."
            )

    st.markdown("---")

    # Charts
    agg_trend = aggregate_trend(filtered, metric)
    st.subheader("Trend over time")
    st.plotly_chart(plot_trend(agg_trend, metric), use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Top 10 countries")
        fig_top = plot_top_countries(filtered, metric, int(top_n))
        st.pyplot(fig_top)
    with col_right:
        st.subheader("Population groups compared")
        fig_grp = plot_group_comparison(filtered, metric)
        st.pyplot(fig_grp)

    # Daten-Tabelle
    with st.expander("Show data"):
        st.dataframe(
            filtered.sort_values(["DATE", "COUNTRY", "GROUP"]).reset_index(drop=True)
        )


if __name__ == "__main__":
    main()