from __future__ import annotations
from typing import List, Optional

import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Light style for charts
sns.set_theme(style="whitegrid", palette="pastel")

# Global matplotlib font sizes for better readability
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

DATA_PATH = "WHO-COVID-19-global-data.csv"
MIN_CASES_FOR_CFR = 1000  # threshold so early days do not explode the ratio

# ---- East / West Europe country sets ----
EAST_EUROPE = {
    "Poland", "Czechia", "Czech Republic", "Slovakia", "Hungary",
    "Romania", "Bulgaria", "Croatia", "Slovenia",
    "Estonia", "Latvia", "Lithuania",
    "Serbia", "Bosnia and Herzegovina", "North Macedonia",
    "Albania", "Montenegro", "Moldova", "Ukraine", "Belarus"
}

WEST_EUROPE = {
    "Germany", "France", "Italy", "Spain", "Portugal",
    "Netherlands", "Belgium", "Luxembourg", "Austria",
    "Switzerland", "United Kingdom", "Ireland",
    "Norway", "Sweden", "Finland", "Denmark", "Iceland",
    "Greece", "Cyprus", "Malta"
}


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date_reported"] = pd.to_datetime(df["Date_reported"], errors="coerce")
    df["New_cases"] = df["New_cases"].fillna(0)
    df["New_deaths"] = df["New_deaths"].fillna(0)
    return df


def make_region_mask(df: pd.DataFrame, regions: List[str]) -> pd.Series:
    """Return boolean mask for rows whose region falls in the selected list.

    Supports normal WHO_region values plus pseudo-regions
    'West Europe' and 'East Europe' (subsets of EURO).
    """
    if not regions:
        return pd.Series(True, index=df.index)

    real_regions = set(df["WHO_region"].dropna().unique())
    mask = pd.Series(False, index=df.index)

    for r in regions:
        if r in real_regions:
            mask |= df["WHO_region"].eq(r)
        elif r == "West Europe":
            mask |= df["WHO_region"].eq("EURO") & df["Country"].isin(WEST_EUROPE)
        elif r == "East Europe":
            mask |= df["WHO_region"].eq("EURO") & df["Country"].isin(EAST_EUROPE)

    return mask


def filter_data(df, regions, countries, date_range):
    start, end = date_range
    mask = (df["Date_reported"] >= start) & (df["Date_reported"] <= end)

    if regions:
        mask &= make_region_mask(df, regions)

    if countries:
        mask &= df["Country"].isin(countries)

    return df[mask].copy()


def aggregate_metric(df, metric, by_country: bool = False):
    """
    Aggregate metric by date, optionally split by country.

    For Fatality_rate we use:
    cumulative_deaths / cumulative_cases * 100,
    and we drop dates where cumulative_cases < MIN_CASES_FOR_CFR.
    """
    if metric == "Fatality_rate":
        if by_country:
            group_cols = ["Country", "Date_reported"]
        else:
            group_cols = ["Date_reported"]

        agg = (
            df.groupby(group_cols, as_index=False)[
                ["Cumulative_cases", "Cumulative_deaths"]
            ].sum()
        )

        agg = agg[agg["Cumulative_cases"] >= MIN_CASES_FOR_CFR].copy()
        agg["Fatality_rate"] = (
            agg["Cumulative_deaths"] / agg["Cumulative_cases"] * 100.0
        )

        return agg

    # all other metrics just summed
    group_cols = ["Date_reported"]
    if by_country:
        group_cols.append("Country")

    grouped = df.groupby(group_cols, as_index=False)
    return grouped[metric].sum()


def aggregate_by_region(df, metric, selected_regions: Optional[List[str]] = None):
    """
    Aggregate metric by region and date.

    Normally region = WHO_region. If the user selected 'West Europe'
    and/or 'East Europe', we split EURO into those blocks for charts.
    """
    df = df.copy()
    region_group = df["WHO_region"].astype(str)

    selected_regions = selected_regions or []

    if "West Europe" in selected_regions:
        mask_w = df["WHO_region"].eq("EURO") & df["Country"].isin(WEST_EUROPE)
        region_group[mask_w] = "West Europe"

    if "East Europe" in selected_regions:
        mask_e = df["WHO_region"].eq("EURO") & df["Country"].isin(EAST_EUROPE)
        region_group[mask_e] = "East Europe"

    df["Region_group"] = region_group

    if metric == "Fatality_rate":
        region = (
            df.groupby(["Region_group", "Date_reported"], as_index=False)[
                ["Cumulative_cases", "Cumulative_deaths"]
            ].sum()
        )
        region = region[region["Cumulative_cases"] >= MIN_CASES_FOR_CFR].copy()
        region["Fatality_rate"] = (
            region["Cumulative_deaths"] / region["Cumulative_cases"] * 100.0
        )
        region = region.rename(columns={"Region_group": "WHO_region"})
        return region

    region = (
        df.groupby(["Region_group", "Date_reported"], as_index=False)[metric].sum()
    ).rename(columns={"Region_group": "WHO_region"})

    return region


def plot_trend(agg, metric, by_country: bool = False):
    titles = {
        "New_cases": "New cases over time",
        "New_deaths": "New deaths over time",
        "Cumulative_cases": "Cumulative cases over time",
        "Cumulative_deaths": "Cumulative deaths over time",
        "Fatality_rate": "Fatality rate over time",
    }

    # We avoid setting a figure title here so the Streamlit subheader
    # remains the single visible heading (prevents duplicate headings).
    # Use a moderate height so the page shows all charts without scrolling.
    common_kwargs = dict(
        x="Date_reported",
        y=metric,
        height=480,
    )

    if by_country:
        fig = px.line(agg, color="Country", **common_kwargs)
    else:
        fig = px.line(agg, **common_kwargs)

    fig.update_layout(
        # Ensure there is no figure title shown (Streamlit subheader is used)
        title_text="",
        margin=dict(l=70, r=25, t=18, b=48),
        title_x=0.02,
        font=dict(size=17),
    )

    fig.update_xaxes(title_font=dict(size=14), tickfont=dict(size=12))
    if metric == "Fatality_rate":
        fig.update_yaxes(ticksuffix=" %", tickformat=".2f", title_font=dict(size=14), tickfont=dict(size=12))
    else:
        fig.update_yaxes(tickformat=",", title_font=dict(size=14), tickfont=dict(size=12))

    return fig


def plot_top_countries(df, metric, top_n):
    latest = df.sort_values("Date_reported").groupby("Country").tail(1).copy()

    if metric == "Fatality_rate":
        latest = latest[latest["Cumulative_cases"] >= MIN_CASES_FOR_CFR].copy()
        latest["Fatality_rate"] = (
            latest["Cumulative_deaths"] / latest["Cumulative_cases"] * 100.0
        )
        rank_metric = "Fatality_rate"
    else:
        rank_metric = "Cumulative_cases" if "cases" in metric else "Cumulative_deaths"

    top = (
        latest.sort_values(rank_metric, ascending=False)
        .head(top_n)
        .sort_values(rank_metric)
    )

    height = min(320, 40 * len(top) + 80)
    fig, ax = plt.subplots(figsize=(6.5, height / 100))
    sns.barplot(data=top, x=rank_metric, y="Country", ax=ax)

    if rank_metric == "Fatality_rate":
        ax.set_xlabel("Fatality rate (%)", fontsize=11)
    # Title is handled by Streamlit subheader to avoid duplicate headings
    plt.tight_layout()
    return fig


def plot_region_trends(region_df, metric, country_df=None):
    """
    Plot regions and, if provided, also selected countries.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    # Regions
    for region, group in region_df.groupby("WHO_region"):
        group = group.sort_values("Date_reported")
        ax.plot(
            group["Date_reported"],
            group[metric],
            label=f"{region} (region)",
            linewidth=1.8,
        )

    # Countries (if any)
    if country_df is not None and not country_df.empty:
        for country, group in country_df.groupby("Country"):
            group = group.sort_values("Date_reported")
            ax.plot(
                group["Date_reported"],
                group[metric],
                linestyle="--",
                linewidth=1.5,
                label=f"{country} (country)",
            )

    # Title is handled by Streamlit subheader to avoid duplicate headings
    if metric == "Fatality_rate":
        ax.set_ylabel("Fatality rate (%)", fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def plot_top_countries_plotly(df, metric, top_n, height_px: int = 420):
    """Plot top countries as a horizontal bar chart using Plotly.

    Returns a Plotly Figure with the requested pixel height so it can be
    aligned with other Plotly charts.
    """
    latest = df.sort_values("Date_reported").groupby("Country").tail(1).copy()

    if metric == "Fatality_rate":
        latest = latest[latest["Cumulative_cases"] >= MIN_CASES_FOR_CFR].copy()
        latest["Fatality_rate"] = (
            latest["Cumulative_deaths"] / latest["Cumulative_cases"] * 100.0
        )
        rank_metric = "Fatality_rate"
    else:
        rank_metric = "Cumulative_cases" if "cases" in metric else "Cumulative_deaths"

    top = (
        latest.sort_values(rank_metric, ascending=False)
        .head(top_n)
        .sort_values(rank_metric)
    )

    if top.empty:
        return None

    fig = px.bar(
        top,
        x=rank_metric,
        y="Country",
        orientation="h",
        labels={
            "Country": "Country",
            rank_metric: "Cumulative cases" if rank_metric == "Cumulative_cases" else rank_metric,
        },
        height=height_px,
    )

    fig.update_layout(margin=dict(l=80, r=10, t=18, b=30), showlegend=False)
    fig.update_xaxes(tickformat=",")
    return fig


def plot_region_trends_plotly(region_df, metric, country_df=None, height_px: int = 420):
    """Plot WHO regions (and optionally selected countries) as a Plotly line chart.

    Uses the same `height_px` as the top-countries chart so both line up.
    """
    if region_df.empty:
        return None

    # Ensure Date_reported is datetime
    df = region_df.copy()
    df = df.sort_values("Date_reported")

    fig = px.line(df, x="Date_reported", y=metric, color="WHO_region", height=height_px)

    # Add country lines (dashed) if provided
    if country_df is not None and not country_df.empty:
        for country, group in country_df.groupby("Country"):
            group = group.sort_values("Date_reported")
            fig.add_scatter(
                x=group["Date_reported"],
                y=group[metric],
                mode="lines",
                name=f"{country} (country)",
                line=dict(dash="dash"),
            )

    fig.update_layout(margin=dict(l=40, r=10, t=20, b=30), legend=dict(x=1.02, y=1), font=dict(size=13))
    if metric == "Fatality_rate":
        fig.update_yaxes(ticksuffix=" %", tickformat=".2f")
    else:
        fig.update_yaxes(tickformat=",")

    return fig


def plot_map_bubbles(latest_df: pd.DataFrame):
    """
    World map:
    - bubble size = cumulative cases
    - bubble color = fatality rate (%), more red = higher rate

    Uses a clipped fatality rate (0–5%) so high values are clearly red.
    """
    if latest_df.empty:
        return None

    map_df = latest_df.copy()

    # Fatality rate based on cumulative values
    has_enough_cases = map_df["Cumulative_cases"] >= MIN_CASES_FOR_CFR
    map_df["Fatality_rate"] = 0.0
    map_df.loc[has_enough_cases, "Fatality_rate"] = (
        map_df.loc[has_enough_cases, "Cumulative_deaths"]
        / map_df.loc[has_enough_cases, "Cumulative_cases"]
        * 100.0
    )

    # Clip for visualization: everything above 5% gets the deepest red
    map_df["Fatality_rate_plot"] = map_df["Fatality_rate"].clip(upper=5.0)

    custom_red_scale = [
        [0.0, "#ffecec"],
        [0.15, "#ffc9c9"],
        [0.35, "#ff9999"],
        [0.60, "#ff4d4f"],
        [0.85, "#e03131"],
        [1.0, "#990000"],
    ]

    fig = px.scatter_geo(
        map_df,
        locations="Country",
        locationmode="country names",
        size="Cumulative_cases",
        color="Fatality_rate_plot",
        hover_name="Country",
        hover_data={
            "Cumulative_cases": ":,",
            "Cumulative_deaths": ":,",
            "Fatality_rate": ":.2f",  # true (unclipped) rate
        },
        size_max=50,
        projection="natural earth",
        color_continuous_scale=custom_red_scale,
    )

    fig.update_coloraxes(
        cmin=0,
        cmax=5,
        colorbar_title="Fatality rate (%)\n(clipped at 5%)",
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=56, b=12),
        title_x=0.02,
        title="Cumulative cases (bubble size) and fatality rate (color)",
        height=460,
        font=dict(size=15),
    )

    fig.update_xaxes(title_font=dict(size=14), tickfont=dict(size=12))
    fig.update_yaxes(title_font=dict(size=14), tickfont=dict(size=12))

    return fig


def plot_country_fatality_scatter(
    latest_df: pd.DataFrame,
    selected_regions: List[str],
    selected_countries: Optional[List[str]] = None,
):
    """
    Scatter of countries under the map:
    x = cumulative cases, y = fatality rate (%)
    bubble size = cumulative cases

    - Always shows all countries in the filtered data.
    - If selected_countries is not empty, those points are highlighted in red.
    - If no countries are selected, East Europe vs West Europe use different colors.
    """
    if latest_df.empty:
        return None

    df = latest_df.copy()
    df = df[df["Cumulative_cases"] >= MIN_CASES_FOR_CFR]
    if df.empty:
        return None

    # Compute fatality rate
    df["Fatality_rate"] = (
        df["Cumulative_deaths"] / df["Cumulative_cases"] * 100.0
    )

    # Build Region_group = WHO_region, but split EURO into East / West when those are selected
    region_group = df["WHO_region"].astype(str)

    if "West Europe" in selected_regions:
        mask_w = df["WHO_region"].eq("EURO") & df["Country"].isin(WEST_EUROPE)
        region_group[mask_w] = "West Europe"

    if "East Europe" in selected_regions:
        mask_e = df["WHO_region"].eq("EURO") & df["Country"].isin(EAST_EUROPE)
        region_group[mask_e] = "East Europe"

    df["Region_group"] = region_group

    # --- Case 1: highlighted selection ---
    if selected_countries:
        df["Selected"] = df["Country"].isin(selected_countries)
        df["Highlight"] = df["Selected"].map({True: "Selected", False: "Other"})
        df["Size"] = df["Cumulative_cases"]

        fig = px.scatter(
            df,
            x="Cumulative_cases",
            y="Fatality_rate",
            size="Size",
            color="Highlight",
            hover_name="Country",
            hover_data={
                "Region_group": True,
                "Cumulative_cases": ":,",
                "Cumulative_deaths": ":,",
                "Fatality_rate": ":.2f",
            },
            labels={
                "Cumulative_cases": "Cumulative cases",
                "Fatality_rate": "Fatality rate (%)",
                "Highlight": "",
            },
            title="Countries (selected highlighted): fatality rate vs cases",
            color_discrete_map={
                "Other": "#d0d4da",
                "Selected": "#e03131",
            },
            size_max=40,
        )
        fig.update_traces(marker=dict(opacity=0.4))
        for i, trace in enumerate(fig.data):
            if trace.name == "Selected":
                fig.data[i].marker.opacity = 0.9

    # --- Case 2: no selection -> color by Region_group (East vs West visible) ---
    else:
        fig = px.scatter(
            df,
            x="Cumulative_cases",
            y="Fatality_rate",
            size="Cumulative_cases",
            color="Region_group",
            hover_name="Country",
            hover_data={
                "Cumulative_cases": ":,",
                "Cumulative_deaths": ":,",
                "Fatality_rate": ":.2f",
            },
            labels={
                "Cumulative_cases": "Cumulative cases",
                "Fatality_rate": "Fatality rate (%)",
                "Region_group": "Region",
            },
            title="All countries: fatality rate vs cases",
            color_discrete_map={
                "East Europe": "#1f77b4",   # blue
                "West Europe": "#ff7f0e",   # orange
            },
            size_max=40,
        )

    fig.update_xaxes(tickformat=",", title_font=dict(size=14), tickfont=dict(size=12))
    fig.update_yaxes(ticksuffix=" %", tickformat=".2f", title_font=dict(size=14), tickfont=dict(size=12))
    fig.update_layout(height=420, margin=dict(l=40, r=10, t=60, b=40), font=dict(size=15))
    return fig



def plot_region_cases_deaths(
    latest_df: pd.DataFrame,
    selected_regions: List[str],
    selected_countries: Optional[List[str]] = None,
):
    """
    Scatter of regions:
    x = cumulative cases (sum of countries),
    y = cumulative deaths (sum of countries)

    - Uses East / West Europe split if those pseudo-regions are selected.
    - Always shows all regions.
    - Regions that contain at least one selected country are highlighted.
    """
    if latest_df.empty:
        return None

    df = latest_df.copy()
    region_group = df["WHO_region"].astype(str)

    if "West Europe" in selected_regions:
        mask_w = df["WHO_region"].eq("EURO") & df["Country"].isin(WEST_EUROPE)
        region_group[mask_w] = "West Europe"

    if "East Europe" in selected_regions:
        mask_e = df["WHO_region"].eq("EURO") & df["Country"].isin(EAST_EUROPE)
        region_group[mask_e] = "East Europe"

    df["Region_group"] = region_group

    # Determine which regions should be highlighted based on selected countries
    highlight_regions: set[str] = set()
    if selected_countries:
        sel = df[df["Country"].isin(selected_countries)].copy()
        highlight_regions = set(sel["Region_group"].unique())

    reg = (
        df.groupby("Region_group", as_index=False)[
            ["Cumulative_cases", "Cumulative_deaths"]
        ].sum()
    )
    reg = reg[reg["Cumulative_cases"] > 0]
    if reg.empty:
        return None

    if highlight_regions:
        reg["Selected"] = reg["Region_group"].isin(highlight_regions)
        reg["Highlight"] = reg["Selected"].map({True: "Selected", False: "Other"})
        reg["Size"] = reg["Selected"].map({True: 18, False: 10})

        fig = px.scatter(
            reg,
            x="Cumulative_cases",
            y="Cumulative_deaths",
            color="Highlight",
            size="Size",
            hover_name="Region_group",
            labels={
                "Cumulative_cases": "Cumulative cases",
                "Cumulative_deaths": "Cumulative deaths",
                "Region_group": "Region",
                "Highlight": "",
            },
            title="Regions: cases vs deaths (region of selected country highlighted)",
            color_discrete_map={
                "Other": "#d0d4da",
                "Selected": "#e03131",
            },
        )
        fig.update_traces(marker=dict(opacity=0.45))
        for i, trace in enumerate(fig.data):
            if trace.name == "Selected":
                fig.data[i].marker.opacity = 0.9
    else:
        # No selected country – just color by region normally
        fig = px.scatter(
            reg,
            x="Cumulative_cases",
            y="Cumulative_deaths",
            hover_name="Region_group",
            color="Region_group",
            labels={
                "Cumulative_cases": "Cumulative cases",
                "Cumulative_deaths": "Cumulative deaths",
                "Region_group": "Region",
            },
            title="Regions: cases vs deaths",
        )

    fig.update_xaxes(tickformat=",", title_font=dict(size=14), tickfont=dict(size=12))
    fig.update_yaxes(tickformat=",", title_font=dict(size=14), tickfont=dict(size=12))
    fig.update_layout(height=420, margin=dict(l=40, r=10, t=60, b=40), font=dict(size=15))
    return fig


# KPI helpers
def format_big_int(n):
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f} B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n/1_000:.2f} K"
    return f"{n}"


def metric_card(label, value):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def main():
    st.set_page_config(
    page_title="COVID Dashboard",
    layout="centered"
)

    # ---- Global Styling ----
    st.markdown(
        """
        <style>

        /* ====== GLOBAL FONT SIZE ====== */
        html, body, [class*="css"]  {
            font-size: 18px;
        }

        /* Sidebar width and base font - make widgets and labels larger */
        section[data-testid="stSidebar"] {
            min-width: 380px;
            font-size: 18px;
        }

        /* Sidebar section titles */
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        /* Generic widget labels (select, multiselect, slider, radio etc.) */
        section[data-testid="stSidebar"] label {
            font-size: 1.15rem;
            font-weight: 700;
            color: #343a40;
        }

        /* Slightly more space between sidebar widgets */
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            padding-bottom: 0.7rem;
        }

        /* ---- Multiselect & select display text bigger ---- */
        /* Selected chips */
        .stMultiSelect [data-baseweb="tag"] {
            background-color: #f1f3f5 !important;
            color: #333 !important;
            border-radius: 6px !important;
            border: 1px solid #dee2e6 !important;
            font-size: 1.1rem !important;
        }
        .stMultiSelect [data-baseweb="tag"]:hover {
            background-color: #e9ecef !important;
        }

        /* Value area of selects and multiselects */
        .stMultiSelect div[role="combobox"] span,
        .stSelectbox div[role="button"] span {
            font-size: 1.12rem !important;
        }

        /* ---- Radio buttons ---- */
        .stRadio > div > button {
            background-color: #f8f9fa !important;
            border: 1px solid #d0d7de !important;
            color: #333 !important;
            border-radius: 6px !important;
            padding: 9px 16px !important;
            font-size: 1.12rem !important;
        }
        .stRadio > div > button[aria-checked="true"] {
            background-color: #e9ecef !important;
            border: 1px solid #adb5bd !important;
            color: black !important;
        }

        /* Main content: make subheaders consistent and slightly larger */
        div[data-testid="stApp"] h2,
        div[data-testid="stApp"] h3 {
            font-size: 1.25rem;
            font-weight: 700;
            margin-top: 0.35rem;
            margin-bottom: 0.30rem;
        }

        /* Reduce top padding of the main container so content sits higher on the page */
        div.block-container {
            padding-top: 1.2rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        /* KPI Cards */
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 0.6rem;
            padding: 0.9rem 1.0rem;
            border: 1px solid #e5e5e5;
        }
        .metric-label {
            font-size: 0.95rem;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 2.0rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }

        /* Page title stays large */
        .dashboard-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-top: 0.6rem;
            margin-bottom: 1.0rem;
        }

        /* Tabs text a bit larger */
        .stTabs [data-baseweb="tab"] p {
            font-size: 1.05rem;
            font-weight: 650;
        }

        /* Make sidebar stretch to the viewport height and use available space */
        section[data-testid="stSidebar"] {
            min-height: calc(100vh - 2rem);
            position: sticky;
            top: 1rem;
            display: flex;
            flex-direction: column;
            overflow: auto;
        }

        /* Allow the inner sidebar wrapper to take full height so we can push the
           last block down and reduce the visual empty area at the bottom. */
        section[data-testid="stSidebar"] > div {
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
        }

        /* Push the final vertical block to the bottom so the space is used */
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:last-child {
            margin-top: auto;
        }

        </style>
    """,
        unsafe_allow_html=True,
    )

    # Constrain the main content width so the dashboard doesn't span the
    # entire browser on very wide screens. Adjust `max-width` as desired.
    st.markdown(
        """
        <style>
            .main > div {
                max-width: 1400px;   /* increase this number as you like */
                margin-left: auto;
                margin-right: auto;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()

    # ---- Session state for country filter shared between sidebar and map ----
    if "country_filter" not in st.session_state:
        st.session_state["country_filter"] = []

    # ---- Heading ----
    st.markdown(
        "<div class='dashboard-title'>COVID-19 Deaths – KPI Dashboard</div>",
        unsafe_allow_html=True,
    )

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("## Filters")

        st.markdown("### Geography")
        base_regions = sorted(df["WHO_region"].dropna().unique().tolist())
        regions_extended = base_regions + ["East Europe", "West Europe"]

        selected_regions = st.multiselect(
            "WHO regions",
            regions_extended,
            default=base_regions,
        )

        # Available countries depend on region mask (including East/West)
        region_mask_for_countries = make_region_mask(df, selected_regions)
        available_countries = df[region_mask_for_countries]["Country"]
        countries = sorted(available_countries.unique().tolist())

        # Multiselect is bound to session_state["country_filter"]
        selected_countries = st.multiselect(
            "Countries (optional)",
            countries,
            key="country_filter",
        )

        st.markdown("---")

        st.markdown("### Time")
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
            year_start, year_end = st.slider(
                "Year range", min_year, max_year, (min_year, max_year)
            )
            date_range = (pd.Timestamp(year_start, 1, 1), pd.Timestamp(year_end, 12, 31))

        st.markdown("---")

        st.markdown("### Metrics")
        metric = st.selectbox(
            "Metric",
            [
                "New_cases",
                "New_deaths",
                "Cumulative_cases",
                "Cumulative_deaths",
                "Fatality_rate",
            ],
        )
        top_n = st.number_input("Top N countries", 3, 20, 10)

    # Use the session_state country filter as the single source of truth
    selected_countries = st.session_state["country_filter"]

    # ---- Filter Data ----
    filtered = filter_data(df, selected_regions, selected_countries, date_range)
    if filtered.empty:
        st.warning("No data for these filters.")
        return

    # latest per country within filters (used for KPIs, map, scatters)
    latest = (
        filtered.sort_values("Date_reported")
        .groupby("Country", as_index=False)
        .tail(1)
    )

    total_cases = int(latest["Cumulative_cases"].sum())
    total_deaths = int(latest["Cumulative_deaths"].sum())
    fatality_ratio = (total_deaths / total_cases * 100) if total_cases else 0

    # ---- KPI Cards ----
    st.markdown("<div style='margin-top: 0.6rem'></div>", unsafe_allow_html=True)

    total_cases_disp = format_big_int(total_cases)
    total_deaths_disp = format_big_int(total_deaths)
    fatality_disp = f"{fatality_ratio:.2f}%"

    col1, col2, col3 = st.columns(3)
    col1.markdown(metric_card("Total cases", total_cases_disp), unsafe_allow_html=True)
    col2.markdown(
        metric_card("Total deaths", total_deaths_disp), unsafe_allow_html=True
    )
    col3.markdown(metric_card("Fatality rate", fatality_disp), unsafe_allow_html=True)

    st.markdown("---")

    # ---- Tabs ----
    tab_trend, tab_map = st.tabs(["Trends & Regions", "Map"])

    # ==== TAB 1: Trends & regions ====
    with tab_trend:
        by_country = bool(selected_countries)
        agg = aggregate_metric(filtered, metric, by_country=by_country)

        if agg.empty:
            st.warning(
                "Not enough data to compute this metric over time "
                "(try a later date range or different metric)."
            )
        else:
            st.subheader("Trend over time")
            st.markdown("<div style='margin-top: 0.35rem'></div>", unsafe_allow_html=True)
            st.plotly_chart(
                plot_trend(agg, metric, by_country=by_country),
                use_container_width=True,
            )

        region_data = aggregate_by_region(filtered, metric, selected_regions)
        country_region_data = (
            aggregate_metric(filtered, metric, by_country=True)
            if selected_countries
            else None
        )

        col_left, col_right = st.columns(2)

        # Compute a shared pixel height so both charts align vertically while
        # keeping the overall layout compact enough to avoid scrolling/cutoff.
        # Formula scales with `top_n` but clamps between sensible bounds.
        # Raised the minimum and scaling slightly so charts can be a bit larger.
        shared_height = min(700, max(420, 35 * int(top_n) + 110))

        with col_left:
            st.subheader(f"Top {int(top_n)} countries")
            bar_fig = plot_top_countries_plotly(filtered, metric, int(top_n), height_px=shared_height)
            if bar_fig is not None:
                st.plotly_chart(bar_fig, use_container_width=True)
            else:
                st.info("Not enough data to show top countries.")

        with col_right:
            st.subheader("WHO regions compared")
            if region_data.empty:
                st.warning("Not enough data to show regions for this metric.")
            else:
                line_fig = plot_region_trends_plotly(region_data, metric, country_region_data, height_px=shared_height)
                if line_fig is not None:
                    st.plotly_chart(line_fig, use_container_width=True)
                else:
                    st.info("Not enough data to plot regions.")

        with st.expander("Show data"):
            st.dataframe(
                filtered.sort_values(["Date_reported", "Country"]).reset_index(
                    drop=True
                )
            )

    # ==== TAB 2: Map ====
    with tab_map:
        st.subheader("Global map: cases and fatality rate")
        st.markdown(
            "Bubble size = cumulative cases, color = fatality rate (CFR, % – clipped at 5%).\n\n"
            "Click one or more bubbles to select countries. "
            "The sidebar filter and the scatter below will update. "
            "Clear selection on the map or via the filter to deselect."
        )

        map_fig = plot_map_bubbles(latest)

        # Callback to sync map selection -> sidebar filter
        def _update_selected_countries_from_map():
            event = st.session_state.get("world_map")
            if not event:
                return
            selection = event.get("selection", {})
            points = selection.get("points", [])

            countries_clicked = []
            for p in points:
                loc = p.get("hovertext") or p.get("location")
                if loc:
                    countries_clicked.append(loc)

            # Set filter to exactly the selected countries (empty selection = show all)
            st.session_state["country_filter"] = sorted(set(countries_clicked))

        if map_fig is None:
            st.warning("No data available for the map.")
        else:
            st.plotly_chart(
                map_fig,
                use_container_width=True,
                key="world_map",
                on_select=_update_selected_countries_from_map,
                selection_mode="points",
            )

            # Show which countries are currently “active” from either map or sidebar
            active_countries = st.session_state["country_filter"]
            if active_countries:
                st.markdown(
                    f"**Active country filter:** {', '.join(active_countries)}"
                )
            else:
                st.markdown("**Active country filter:** All countries in selected regions")

            st.markdown("---")
            st.subheader("Fatality rate context")

            col_countries, col_regions = st.columns(2)

            with col_countries:
                country_fig = plot_country_fatality_scatter(
                    latest, selected_regions, active_countries
                )
                if country_fig is not None:
                    st.plotly_chart(country_fig, use_container_width=True)
                else:
                    st.info("Not enough data to plot country fatality vs cases.")

            with col_regions:
                region_fig = plot_region_cases_deaths(
                    latest, selected_regions, active_countries
                )
                if region_fig is not None:
                    st.plotly_chart(region_fig, use_container_width=True)
                else:
                    st.info("Not enough data to plot region cases vs deaths.")


if __name__ == "__main__":
    main()