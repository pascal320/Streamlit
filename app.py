"""
COVID‑19 Vaccination Uptake Dashboard
====================================

This Streamlit application visualises global COVID‑19 vaccination uptake
between January 2021 and December 2023.  The dashboard is designed for
policy makers, journalists and the general public to explore how
vaccination programmes progressed across countries and over time.  It
offers a light, colour‑blind friendly theme and interactive filters to
customise the view.  Three types of charts—Plotly, Seaborn and
Matplotlib—are used to satisfy course requirements and to demonstrate
different strengths of each library.

To run this application locally, install Streamlit and the necessary
dependencies (`pandas`, `plotly`, `seaborn`, `matplotlib`) and execute

    streamlit run app.py

The application expects the data file `COV_VAC_UPTAKE_2021_2023.csv` to
reside in the same directory.  No external API calls are required.

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="COVID‑19 Vaccination Uptake Dashboard",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Load and cache the data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load the vaccination uptake data from a CSV file.

    The function caches the result to avoid re‑reading the file on every
    interaction.  Dates are parsed into datetime objects for easier
    manipulation.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        A DataFrame with a datetime‑indexed `DATE` column.
    """
    df = pd.read_csv(path)
    # Convert the date column to datetime
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d")
    return df


# Attempt to load the data; if it fails an error message is displayed
try:
    data = load_data("COV_VAC_UPTAKE_2021_2023.csv")
except FileNotFoundError:
    st.error(
        "Data file not found. Please place `COV_VAC_UPTAKE_2021_2023.csv` in the same directory as this script."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_measure_options() -> dict:
    """Return a mapping of human‑friendly indicator names to column names.

    This dictionary is used to populate the indicator selector and ensures
    that users see descriptive labels while the code references the correct
    DataFrame columns.
    """
    return {
        "Total doses administered": "COVID_VACCINE_ADM_TOT_DOSES",
        "Doses administered per 100 people": "COVID_VACCINE_ADM_TOT_DOSES_PER100",
        "First dose administered": "COVID_VACCINE_ADM_TOT_A1D",
        "Completed primary series": "COVID_VACCINE_ADM_TOT_CPS",
        "Booster doses administered": "COVID_VACCINE_ADM_TOT_BOOST",
        "Coverage: at least one dose (%)": "COVID_VACCINE_COV_TOT_A1D",
        "Coverage: completed primary series (%)": "COVID_VACCINE_COV_TOT_CPS",
        "Coverage: booster (%)": "COVID_VACCINE_COV_TOT_BOOST",
    }


def aggregate_global(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Aggregate a given measure globally over time.

    The function groups by date and sums the specified column.  It
    gracefully handles missing values.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    column : str
        Column name to aggregate.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns: `DATE` and the aggregated measure.
    """
    agg = (
        df.groupby("DATE")[column]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={column: "value"})
    )
    return agg


# ---------------------------------------------------------------------------
# Narrative introduction
# ---------------------------------------------------------------------------
st.title("COVID‑19 Vaccination Uptake Dashboard (2021–2023)")

st.markdown(
    """
    **Role:** Public Health Analyst at the World Health Organization (WHO)

    **Stakeholders:** This dashboard is intended for public health officials,
    policy makers, journalists and citizens who want to understand how COVID‑19
    vaccination programmes progressed across the globe.  It provides a
    high‑level overview as well as the ability to drill down into specific
    countries and time periods.

    **Objective:** Use vaccination uptake data from 2021–2023 to explore
    patterns and disparities in COVID‑19 vaccine distribution.  The
    interactive tools below allow you to select different indicators (e.g., total
    doses administered, coverage rates) and dates, compare countries, and
    identify leaders and laggards.  A light colour palette has been chosen
    throughout to ensure accessibility for users with colour‑vision
    deficiencies.
    """
)


# ---------------------------------------------------------------------------
# Sidebar for filtering
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔧 Controls")
    st.write(
        "Adjust the filters below to explore different aspects of the vaccination data."
    )

    # Select indicator
    measure_options = get_measure_options()
    measure_label = st.selectbox(
        "Indicator", options=list(measure_options.keys()), index=0
    )
    measure_col = measure_options[measure_label]

    # Select reporting date using a slider; default to the most recent date in the data
    unique_dates = sorted(data["DATE"].unique())
    selected_date = st.select_slider(
        "Reporting date",
        options=unique_dates,
        value=unique_dates[-1],
        format_func=lambda d: d.strftime("%Y-%m-%d"),
    )

    # Select countries for time series comparison
    countries = sorted(data["COUNTRY"].unique())
    default_countries = [c for c in ["USA", "CHN", "IND", "BRA"] if c in countries]
    selected_countries = st.multiselect(
        "Countries to compare",
        options=countries,
        default=default_countries,
        help="Pick up to 5 countries to compare their vaccination trends",
    )

    # Select how many top countries to display in the rankings
    top_n = st.slider(
        "Number of countries in ranking", 5, 30, value=10, step=1
    )


# ---------------------------------------------------------------------------
# Tabs for different views
# ---------------------------------------------------------------------------
tab_map, tab_trend, tab_rank, tab_about = st.tabs(
    ["🌍 Global map", "📈 Trends", "🏅 Rankings", "📖 About"]
)


with tab_map:
    """Interactive world map view"""
    st.subheader("Global vaccination map")
    st.markdown(
        f"The choropleth below shows **{measure_label.lower()}** as of **{selected_date.strftime('%Y‑%m‑%d')}**. "
        "Hover over a country to see its value.  Use the indicator and date selectors in the sidebar to change what you see."
    )

    # Filter data for the selected date
    df_date = data[data["DATE"] == selected_date]

    # Build choropleth
    fig = px.choropleth(
        df_date,
        locations="COUNTRY",
        locationmode="ISO-3",
        color=measure_col,
        hover_name="COUNTRY",
        color_continuous_scale="cividis",
        title="",
    )
    fig.update_layout(
        coloraxis_colorbar=dict(title=measure_label),
        template="plotly_white",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    # Handle NaNs gracefully by setting them to a very light colour
    fig.update_traces(
        marker=dict(line=dict(color="#FFFFFF", width=0.5)),
        selector=dict(type="choropleth"),
    )
    st.plotly_chart(fig, use_container_width=True)


with tab_trend:
    """Time series trends view"""
    st.subheader("Trends over time")
    st.markdown(
        """
        The line charts below depict how vaccination uptake has evolved over time.

        * The **global** line aggregates the selected indicator across all countries.
        * The **country** lines allow comparison of individual countries.  Select
          different countries in the sidebar to change which lines are displayed.
        """
    )

    # Aggregate global data for the selected measure
    global_trend = aggregate_global(data, measure_col)

    # Build a Matplotlib/Seaborn line chart for global trend
    fig_global, ax_global = plt.subplots(figsize=(10, 4))
    sns.lineplot(
        data=global_trend, x="DATE", y="value", ax=ax_global, color="#1f77b4"
    )
    ax_global.set_title(f"Global {measure_label} over time", fontsize=12)
    ax_global.set_xlabel("Date")
    ax_global.set_ylabel(measure_label)
    ax_global.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.4)
    st.pyplot(fig_global, use_container_width=True)

    # Plot selected countries using Plotly for interactivity
    if selected_countries:
        df_countries = data[data["COUNTRY"].isin(selected_countries)]
        # Plotly express line chart
        fig_countries = px.line(
            df_countries,
            x="DATE",
            y=measure_col,
            color="COUNTRY",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title=f"{measure_label} for selected countries",
            labels={
                "DATE": "Date",
                measure_col: measure_label,
                "COUNTRY": "Country",
            },
        )
        fig_countries.update_layout(
            template="plotly_white",
            legend_title_text="Country",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_countries, use_container_width=True)
    else:
        st.info(
            "Select one or more countries in the sidebar to display their trends."
        )


with tab_rank:
    """Country ranking view"""
    st.subheader("Country rankings")
    st.markdown(
        f"The bar chart below ranks countries by **{measure_label.lower()}** on **{selected_date.strftime('%Y‑%m‑%d')}**. "
        "Adjust the number of countries shown using the slider in the sidebar."
    )

    # Prepare data for ranking at selected date
    df_rank = (
        data[data["DATE"] == selected_date]
        .dropna(subset=[measure_col])
        .sort_values(by=measure_col, ascending=False)
        .head(top_n)
    )

    # Use Matplotlib bar chart
    fig_rank, ax_rank = plt.subplots(figsize=(10, 0.4 * len(df_rank) + 2))
    sns.barplot(
        data=df_rank,
        x=measure_col,
        y="COUNTRY",
        palette="crest",
        ax=ax_rank,
    )
    ax_rank.set_title(
        f"Top {len(df_rank)} countries by {measure_label} on {selected_date.strftime('%Y‑%m‑%d')}",
        fontsize=12,
    )
    ax_rank.set_xlabel(measure_label)
    ax_rank.set_ylabel("Country")
    # Make x axis labels human friendly by using scientific notation for large numbers
    if not df_rank.empty and df_rank[measure_col].max() > 1e6:
        ax_rank.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
        )
        ax_rank.set_xlabel(f"{measure_label} (millions)")
    ax_rank.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
    st.pyplot(fig_rank, use_container_width=True)


with tab_about:
    """About and limitations view"""
    st.subheader("About the data and limitations")
    st.markdown(
        """
        ### Data source
        The dataset comes from the [WHO COVID‑19 data repository](https://data.who.int/dashboards/covid19/data)
        and contains monthly vaccination uptake statistics for 222 countries from
        January 2021 through December 2023.  Each row represents the value
        reported for a country at a given end‑of‑month date.  Indicators include
        the number of doses administered, doses per 100 people, coverage rates
        for at least one dose, completed primary series and booster doses, as
        well as the date each country started reporting.

        ### Limitations
        * **Reporting lags and gaps:** Not all countries report data at the same
          frequency or with the same timeliness.  Some values may be missing
          or outdated, particularly towards the end of the time span.
        * **Inconsistent definitions:** Different countries may have used
          varying definitions for what constitutes a booster dose or a complete
          primary series, especially early in the vaccination campaign.
        * **Population denominators:** Coverage rates (percentages) rely on
          population estimates that can vary by source.  For simplicity this
          dashboard uses the values provided in the WHO dataset without
          adjustment.
        * **Country codes:** The hover labels on the map display ISO‑3
          codes rather than full country names.  This is because the
          underlying dataset does not include official names and external
          packages could not be installed in the current environment.  Despite
          this limitation, the map still conveys relative differences across
          countries.

        ### Recommendations
        * Users should cross‑check critical figures with official national
          dashboards before drawing policy conclusions.
        * Future iterations could incorporate demographic breakdowns (e.g., by
          age group) and vaccine types to provide a more nuanced picture of
          vaccination strategies.
        * Enhancing the data with regional classifications or income groups
          would allow for more targeted comparisons.
        
        ---
        *Created by a WHO Public Health Analyst for educational purposes.*
        """
    )