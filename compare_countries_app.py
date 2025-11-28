"""
Streamlit dashboard for comparing COVID‑19 statistics between two countries
----------------------------------------------------------------------

This application builds upon an existing WHO COVID‑19 dashboard and
streamlines it for side‑by‑side comparison of two nations.  Users can
choose a pair of countries, a date range (by exact dates or years) and
select which metric to analyse: new cases, new deaths, cumulative cases
or cumulative deaths.  The dashboard then displays key figures for
each country and overlays their trends on a single chart to highlight
differences in the progression of the pandemic.

The design employs a light, colour‑blind friendly palette and clean
interfaces.  To run the app locally, install Streamlit and run:

    streamlit run compare_countries_app.py

Ensure that `WHO-COVID-19-global-data.csv` is in the same directory.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Use a light, colour‑blind friendly style
sns.set_theme(style="whitegrid", palette="Set2")

DATA_PATH = "WHO-COVID-19-global-data.csv"


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    """Load the WHO global COVID‑19 dataset.

    The data includes country name, WHO region and weekly reported
    new/cumulative cases and deaths.  Dates are parsed into datetime
    objects and missing numeric values are filled with zeros.
    """
    df = pd.read_csv(DATA_PATH)
    df["Date_reported"] = pd.to_datetime(df["Date_reported"], errors="coerce")
    for col in ["New_cases", "New_deaths", "Cumulative_cases", "Cumulative_deaths"]:
        df[col] = df[col].fillna(0)
    return df


def filter_by_countries_and_dates(
    df: pd.DataFrame, countries: List[str], date_range: Tuple[pd.Timestamp, pd.Timestamp]
) -> pd.DataFrame:
    """Return rows for the selected countries and within the date range.

    Parameters
    ----------
    df : pd.DataFrame
        The full WHO dataset.
    countries : list of str
        Two ISO country names to compare.
    date_range : tuple(pd.Timestamp, pd.Timestamp)
        Inclusive start and end date.
    """
    start, end = date_range
    mask = (df["Date_reported"] >= start) & (df["Date_reported"] <= end)
    if countries:
        mask &= df["Country"].isin(countries)
    return df.loc[mask].copy()


def compute_latest_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the latest total cases, deaths and fatality ratio for each country.

    The function takes the filtered DataFrame (already limited to two
    countries) and finds the latest entry for each.  It returns a
    summary DataFrame with one row per country.
    """
    latest_rows = df.sort_values("Date_reported").groupby("Country").tail(1)
    summary = latest_rows[[
        "Country",
        "Cumulative_cases",
        "Cumulative_deaths",
    ]].copy()
    summary["Fatality_ratio"] = (
        summary["Cumulative_deaths"] / summary["Cumulative_cases"] * 100
    ).replace([float("inf"), float("nan")], 0)
    return summary.reset_index(drop=True)


def line_comparison_chart(df: pd.DataFrame, metric: str) -> px.line:
    """Create a Plotly line chart comparing two countries over time.

    The metric must be one of the four numeric columns in the data.  The
    resulting figure uses a discrete colour palette friendly to users
    with colour‑vision deficiencies.
    """
    fig = px.line(
        df,
        x="Date_reported",
        y=metric,
        color="Country",
        title=f"{metric.replace('_', ' ').title()} comparison",
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={
            "Date_reported": "Date",
            metric: metric.replace("_", " ").title(),
            "Country": "Country",
        },
        height=450,
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=70, r=25, t=60, b=40),
        title_x=0.02,
    )
    fig.update_yaxes(tickformat=",", title=None)
    return fig


def normalized_trend_chart(df: pd.DataFrame, metric: str) -> px.line:
    """Return a Plotly line chart with each country's metric normalised.

    Each series is divided by its maximum value over the selected period
    so that curves can be compared independent of absolute scale.  The
    y‑axis is labelled from 0 to 1.
    """
    # Compute max for each country to normalise
    df_norm = df.copy()
    max_values = df_norm.groupby("Country")[metric].transform("max")
    df_norm["normalised"] = df_norm[metric] / max_values.replace(0, 1)
    fig = px.line(
        df_norm,
        x="Date_reported",
        y="normalised",
        color="Country",
        title=f"Normalised {metric.replace('_', ' ').title()}",
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={
            "Date_reported": "Date",
            "normalised": "Normalised (0–1)",
            "Country": "Country",
        },
        height=450,
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=70, r=25, t=60, b=40),
        title_x=0.02,
    )
    fig.update_yaxes(range=[0, 1], title=None)
    return fig


def fatality_ratio_chart(df: pd.DataFrame) -> px.line:
    """Return a Plotly line chart of the case fatality ratio over time.

    The ratio is computed as cumulative deaths divided by cumulative
    cases.  Values are expressed as percentages on the y‑axis.  If
    cumulative cases are zero the ratio is set to zero to avoid
    division by zero errors.
    """
    df_ratio = df.copy()
    # Avoid division by zero by replacing zero cumulative cases with NaN then fill
    df_ratio["fatality_ratio"] = (
        df_ratio["Cumulative_deaths"] / df_ratio["Cumulative_cases"] * 100
    ).replace([float("inf"), float("nan")], 0)
    fig = px.line(
        df_ratio,
        x="Date_reported",
        y="fatality_ratio",
        color="Country",
        title="Case fatality ratio over time",
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={
            "Date_reported": "Date",
            "fatality_ratio": "Fatality ratio (%)",
            "Country": "Country",
        },
        height=450,
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=70, r=25, t=60, b=40),
        title_x=0.02,
    )
    fig.update_yaxes(range=[0, df_ratio["fatality_ratio"].max() * 1.05], title=None)
    return fig


def metric_card(label: str, value: str, colour: str = "#f8f9fa") -> str:
    """Render a simple KPI card as HTML.

    Parameters
    ----------
    label : str
        The text describing the metric.
    value : str
        The formatted value to display.
    colour : str
        Background colour for the card.  Defaults to a light grey.
    """
    return f"""
    <div style="background-color: {colour}; border-radius: 0.6rem; padding: 0.7rem 1rem; border: 1px solid #e5e5e5;">
        <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase;">{label}</div>
        <div style="font-size: 1.4rem; font-weight: 600; margin-top: 0.1rem;">{value}</div>
    </div>
    """


def format_big_int(n: float) -> str:
    """Format large numbers for human readability.

    Numbers are expressed using K for thousands, M for millions and B
    for billions.  Smaller values are returned unchanged.
    """
    n = float(n)
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f} B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n/1_000:.2f} K"
    return f"{int(n)}"


def main() -> None:
    st.set_page_config(
        page_title="COVID‑19 Country Comparison",
        layout="wide",
    )

    df = load_data()

    # Heading
    st.markdown(
        "<h1 style='margin-bottom:0.2rem;'>COVID‑19 Country Comparison Dashboard</h1>"
        "<p style='color:#6c757d;'>Compare how two nations have experienced the COVID‑19 pandemic.</p>",
        unsafe_allow_html=True,
    )

    # Sidebar for filters
    with st.sidebar:
        st.header("Filters")

        countries = sorted(df["Country"].dropna().unique().tolist())
        # Attempt to preselect Sweden and Germany if present
        default_pair = [c for c in ["Sweden", "Germany"] if c in countries]
        selected = st.multiselect(
            "Select two countries", options=countries, default=default_pair, help="Choose exactly two countries to compare."
        )
        if len(selected) != 2:
            st.warning("Please select exactly two countries.")

        # Metric selection
        metric = st.radio(
            "Metric",
            ["New_cases", "New_deaths", "Cumulative_cases", "Cumulative_deaths"],
            index=2,
            help="Choose which metric to visualise."
        )

        # Time range selection
        min_date, max_date = df["Date_reported"].min(), df["Date_reported"].max()
        min_year, max_year = int(min_date.year), int(max_date.year)
        time_mode = st.radio("Filter by", ("Date range", "Year range"), horizontal=True)

        if time_mode == "Date range":
            start_dt, end_dt = st.slider(
                "Date range",
                min_value=min_date.to_pydatetime(),
                max_value=max_date.to_pydatetime(),
                value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
                format="YYYY-MM-DD",
            )
            date_range = (pd.Timestamp(start_dt), pd.Timestamp(end_dt))
        else:
            year_start, year_end = st.slider("Year range", min_year, max_year, (min_year, max_year))
            date_range = (pd.Timestamp(year_start, 1, 1), pd.Timestamp(year_end, 12, 31))

    # If two countries are selected, proceed
    if len(selected) == 2:
        filtered = filter_by_countries_and_dates(df, selected, date_range)
        if filtered.empty:
            st.warning("No data available for the selected countries and date range.")
            return

        # Compute latest metrics for both countries
        summary = compute_latest_metrics(filtered)

        # Display KPI cards side by side
        st.markdown("<h2 style='margin-top:0;'>Key figures</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        for idx, row in summary.iterrows():
            country_col = col1 if idx == 0 else col2
            cases = format_big_int(row["Cumulative_cases"])
            deaths = format_big_int(row["Cumulative_deaths"])
            fatality = f"{row['Fatality_ratio']:.2f}%"
            # Compose cards
            card_html = (
                f"<h3 style='margin-bottom:0.3rem;'>{row['Country']}</h3>"
                + metric_card("Cumulative cases", cases)
                + metric_card("Cumulative deaths", deaths)
                + metric_card("Fatality rate", fatality)
            )
            country_col.markdown(card_html, unsafe_allow_html=True)

        st.markdown("---")

        # Plot comparison chart for selected metric
        st.subheader("Trend comparison")
        st.markdown(
            """
            The line chart below plots the selected metric for both countries over
            the chosen time period.  Hover over the lines to see specific values.
            """,
        )
        fig = line_comparison_chart(filtered, metric)
        st.plotly_chart(fig, use_container_width=True)

        # Normalised trend comparison to see relative progression independent of scale
        st.subheader("Normalised trend comparison")
        st.markdown(
            """
            Each series below is normalised by its maximum value during the selected
            period.  This highlights differences in timing and growth patterns, even
            when absolute numbers differ greatly due to population size or other
            factors.
            """
        )
        fig_norm = normalized_trend_chart(filtered, metric)
        st.plotly_chart(fig_norm, use_container_width=True)

        # Case fatality ratio chart
        st.subheader("Case fatality ratio over time")
        st.markdown(
            """
            The case fatality ratio (CFR) is computed as cumulative deaths divided
            by cumulative cases.  It provides a sense of how deadly the
            outbreak has been in each country.  Because population data are not
            available in this environment, CFR offers a population‑independent
            measure for comparison.
            """
        )
        fig_cfr = fatality_ratio_chart(filtered)
        st.plotly_chart(fig_cfr, use_container_width=True)

        # Show raw data if requested
        with st.expander("Show raw data"):
            st.dataframe(
                filtered.sort_values(["Country", "Date_reported"]).reset_index(drop=True)
            )

    else:
        st.info("Select exactly two countries from the sidebar to begin comparison.")


if __name__ == "__main__":
    main()