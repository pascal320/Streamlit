from __future__ import annotations
from typing import List, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Light style for charts
sns.set_theme(style="whitegrid", palette="pastel")

DATA_PATH = "WHO-COVID-19-global-data.csv"

@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date_reported"] = pd.to_datetime(df["Date_reported"], errors="coerce")
    df["New_cases"] = df["New_cases"].fillna(0)
    df["New_deaths"] = df["New_deaths"].fillna(0)
    return df


def filter_data(df, regions, countries, date_range):
    start, end = date_range
    mask = (df["Date_reported"] >= start) & (df["Date_reported"] <= end)
    if regions:
        mask &= df["WHO_region"].isin(regions)
    if countries:
        mask &= df["Country"].isin(countries)
    return df[mask].copy()


def aggregate_metric(df, metric):
    return df.groupby("Date_reported", as_index=False)[metric].sum()


def plot_trend(agg, metric):
    titles = {
        "New_cases": "New cases over time",
        "New_deaths": "New deaths over time",
        "Cumulative_cases": "Cumulative cases over time",
        "Cumulative_deaths": "Cumulative deaths over time",
    }
    fig = px.line(
        agg,
        x="Date_reported",
        y=metric,
        title=titles.get(metric, metric),
        height=420,
    )
    fig.update_layout(
        margin=dict(l=70, r=25, t=90, b=50),
        title_x=0.02,
    )
    fig.update_yaxes(tickformat=",")
    return fig


def plot_top_countries(df, metric, top_n):
    rank_metric = "Cumulative_cases" if "cases" in metric else "Cumulative_deaths"
    latest = df.sort_values("Date_reported").groupby("Country").tail(1)
    top = latest.sort_values(rank_metric, ascending=False).head(top_n)
    top = top.sort_values(rank_metric)

    height = min(280, 40 * len(top) + 60)
    fig, ax = plt.subplots(figsize=(6, height / 100))
    sns.barplot(data=top, x=rank_metric, y="Country", ax=ax)
    ax.set_title(f"Top {len(top)} countries")
    plt.tight_layout()
    return fig


def plot_region_trends(df, metric):
    fig, ax = plt.subplots(figsize=(6, 3))
    for region, group in df.groupby("WHO_region"):
        group = group.sort_values("Date_reported")
        ax.plot(group["Date_reported"], group[metric], label=region, linewidth=1.5)
        ax.set_title(f"{metric} by WHO region")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
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
    st.set_page_config(page_title="WHO COVID-19 Dashboard", layout="wide")

    # ---- Global Styling ----
    st.markdown("""
        <style>

        /* ---- Clean White/Grey Multiselect Chips ---- */
        .stMultiSelect [data-baseweb="tag"] {
            background-color: #f1f3f5 !important;
            color: #333 !important;
            border-radius: 6px !important;
            border: 1px solid #dee2e6 !important;
        }
        .stMultiSelect [data-baseweb="tag"]:hover {
            background-color: #e9ecef !important;
        }
        .stMultiSelect [data-baseweb="tag"] svg {
            fill: #666 !important;
        }

        /* ---- Radio buttons clean look ---- */
        .stRadio > div > button {
            background-color: #f8f9fa !important;
            border: 1px solid #d0d7de !important;
            color: #333 !important;
            border-radius: 6px !important;
            padding: 4px 10px !important;
        }
        .stRadio > div > button[aria-checked="true"] {
            background-color: #e9ecef !important;
            border: 1px solid #adb5bd !important;
            color: black !important;
        }

        /* KPI Cards */
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 0.6rem;
            padding: 0.7rem 1rem;
            border: 1px solid #e5e5e5;
        }
        .metric-label {
            font-size: 0.75rem;
            color: #6c757d;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 1.4rem;
            font-weight: 600;
            margin-top: 0.1rem;
        }

        /* Page title */
        .dashboard-title {
            font-size: 2rem;
            font-weight: 600;
            margin-top: 0.5rem;
            margin-bottom: 1.2rem;
        }

        </style>
    """, unsafe_allow_html=True)

    df = load_data()

    # All countries (independent of filters)
    all_countries = sorted(df["Country"].dropna().unique().tolist())

    # ---- Heading ----
    st.markdown("<div class='dashboard-title'>COVID-19 Deaths – KPI Dashboard</div>", unsafe_allow_html=True)

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("## Filters")

        st.markdown("### Geography")
        regions = sorted(df["WHO_region"].dropna().unique().tolist())
        selected_regions = st.multiselect("WHO regions", regions, default=regions)

        # Countries filter depends on selected regions, but if none selected, show all countries
        if selected_regions:
            available_countries = sorted(
                df[df["WHO_region"].isin(selected_regions)]["Country"].dropna().unique().tolist()
            )
        else:
            available_countries = all_countries

        selected_countries = st.multiselect("Countries (optional)", available_countries)

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
            year_start, year_end = st.slider("Year range", min_year, max_year, (min_year, max_year))
            date_range = (pd.Timestamp(year_start, 1, 1), pd.Timestamp(year_end, 12, 31))

        st.markdown("### Metrics")
        metric = st.selectbox("Metric", ["New_cases", "New_deaths", "Cumulative_cases", "Cumulative_deaths"])
        top_n = st.number_input("Top N countries", 3, 20, 10)

        # ---- Storytelling comparison controls ----
        st.markdown("### Storytelling comparison")
        compare_mode = st.radio("Compare", ("None", "Countries", "WHO regions"))

        comp_country_1 = comp_country_2 = None
        comp_region_1 = comp_region_2 = None

        if compare_mode == "Countries":
            comp_options = all_countries
            if comp_options:
                default_b = 1 if len(comp_options) > 1 else 0
                comp_country_1 = st.selectbox("Country A", comp_options, index=0)
                comp_country_2 = st.selectbox("Country B", comp_options, index=default_b)
        elif compare_mode == "WHO regions":
            comp_region_options = regions
            if comp_region_options:
                default_b = 1 if len(comp_region_options) > 1 else 0
                comp_region_1 = st.selectbox("Region A", comp_region_options, index=0)
                comp_region_2 = st.selectbox("Region B", comp_region_options, index=default_b)

    # ---- Filter Data for main dashboard ----
    filtered = filter_data(df, selected_regions, selected_countries, date_range)
    if filtered.empty:
        st.warning("No data for these filters.")
        return

    latest = filtered.sort_values("Date_reported").groupby("Country").tail(1)
    total_cases = int(latest["Cumulative_cases"].sum())
    total_deaths = int(latest["Cumulative_deaths"].sum())
    fatality_ratio = (total_deaths / total_cases * 100) if total_cases else 0

    # ---- KPI Cards ----
    st.markdown("<div style='margin-top: 1rem'></div>", unsafe_allow_html=True)

    total_cases_disp = format_big_int(total_cases)
    total_deaths_disp = format_big_int(total_deaths)
    fatality_disp = f"{fatality_ratio:.2f}%"

    col1, col2, col3 = st.columns(3)
    col1.markdown(metric_card("Total cases", total_cases_disp), unsafe_allow_html=True)
    col2.markdown(metric_card("Total deaths", total_deaths_disp), unsafe_allow_html=True)
    col3.markdown(metric_card("Fatality rate", fatality_disp), unsafe_allow_html=True)

    # ---- Storytelling comparison block (countries OR regions) ----
    if compare_mode != "None":
        st.markdown("---")
        st.subheader("Country or region comparison")

        # Compare two countries (independent of region filters)
        if compare_mode == "Countries" and comp_country_1 and comp_country_2 and comp_country_1 != comp_country_2:
            group_col = "Country"
            df_comp = df[
                (df["Country"].isin([comp_country_1, comp_country_2]))
                & (df["Date_reported"] >= date_range[0])
                & (df["Date_reported"] <= date_range[1])
            ].copy()

            if df_comp.empty:
                st.warning("No data available for this date range and these countries.")
            else:
                latest_comp = df_comp.sort_values("Date_reported").groupby(group_col).tail(1)

                col_a, col_b = st.columns(2)
                for entity, col in zip([comp_country_1, comp_country_2], [col_a, col_b]):
                    row = latest_comp[latest_comp[group_col] == entity]
                    if not row.empty:
                        cases = int(row["Cumulative_cases"].iloc[0])
                        deaths = int(row["Cumulative_deaths"].iloc[0])
                        fr = f"{(deaths / cases * 100):.2f}%" if cases else "0.00%"
                        with col:
                            st.markdown(f"#### {entity}")
                            k1, k2, k3 = st.columns(3)
                            k1.markdown(metric_card("Cases", format_big_int(cases)), unsafe_allow_html=True)
                            k2.markdown(metric_card("Deaths", format_big_int(deaths)), unsafe_allow_html=True)
                            k3.markdown(metric_card("Fatality rate", fr), unsafe_allow_html=True)

                agg_comp = (
                    df_comp.groupby([group_col, "Date_reported"], as_index=False)[metric]
                    .sum()
                )
                fig = px.line(
                    agg_comp,
                    x="Date_reported",
                    y=metric,
                    color=group_col,
                    title=f"{metric.replace('_', ' ')} – {comp_country_1} vs {comp_country_2}",
                    height=420,
                )
                fig.update_layout(
                    margin=dict(l=70, r=25, t=80, b=40),
                    title_x=0.02,
                    legend_title="",
                )
                fig.update_yaxes(tickformat=",")
                st.plotly_chart(fig, use_container_width=True)

        # Compare two WHO regions (whole regions)
        elif compare_mode == "WHO regions" and comp_region_1 and comp_region_2 and comp_region_1 != comp_region_2:
            group_col = "WHO_region"
            df_comp = df[
                (df["WHO_region"].isin([comp_region_1, comp_region_2]))
                & (df["Date_reported"] >= date_range[0])
                & (df["Date_reported"] <= date_range[1])
            ].copy()

            if df_comp.empty:
                st.warning("No data available for this date range and these regions.")
            else:
                latest_comp = df_comp.sort_values("Date_reported").groupby(group_col).tail(1)

                col_a, col_b = st.columns(2)
                for entity, col in zip([comp_region_1, comp_region_2], [col_a, col_b]):
                    row = latest_comp[latest_comp[group_col] == entity]
                    if not row.empty:
                        cases = int(row["Cumulative_cases"].iloc[0])
                        deaths = int(row["Cumulative_deaths"].iloc[0])
                        fr = f"{(deaths / cases * 100):.2f}%" if cases else "0.00%"
                        with col:
                            st.markdown(f"#### {entity}")
                            k1, k2, k3 = st.columns(3)
                            k1.markdown(metric_card("Cases", format_big_int(cases)), unsafe_allow_html=True)
                            k2.markdown(metric_card("Deaths", format_big_int(deaths)), unsafe_allow_html=True)
                            k3.markdown(metric_card("Fatality rate", fr), unsafe_allow_html=True)

                agg_comp = (
                    df_comp.groupby([group_col, "Date_reported"], as_index=False)[metric]
                    .sum()
                )
                fig = px.line(
                    agg_comp,
                    x="Date_reported",
                    y=metric,
                    color=group_col,
                    title=f"{metric.replace('_', ' ')} – {comp_region_1} vs {comp_region_2}",
                    height=420,
                )
                fig.update_layout(
                    margin=dict(l=70, r=25, t=80, b=40),
                    title_x=0.02,
                    legend_title="",
                )
                fig.update_yaxes(tickformat=",")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

    # ---- Main charts ----
    agg = aggregate_metric(filtered, metric)

    st.subheader("Trend over time")
    st.markdown("<div style='margin-top: 0.5rem'></div>", unsafe_allow_html=True)
    st.plotly_chart(plot_trend(agg, metric), use_container_width=True)

    region_data = (
        filtered.groupby(["WHO_region", "Date_reported"], as_index=False)[metric].sum()
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(f"Top {int(top_n)} countries")
        st.pyplot(plot_top_countries(filtered, metric, int(top_n)))

    with col_right:
        st.subheader("WHO regions compared")
        if region_data.empty:
            st.warning("No regional data available for this selection.")
        else:
            st.pyplot(plot_region_trends(region_data, metric))

    with st.expander("Show data"):
        st.dataframe(
            filtered.sort_values(["Date_reported", "Country"]).reset_index(drop=True)
        )


if __name__ == "__main__":
    main()