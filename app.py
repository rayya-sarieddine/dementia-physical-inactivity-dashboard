
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Dementia & Physical Inactivity Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# Custom CSS for cleaner layout
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        max-width: 1500px;
    }

    h1 {
        font-size: 2.1rem !important;
        margin-bottom: 0.2rem;
    }

    h2, h3 {
        margin-top: 0.5rem;
    }

    .stMetric {
        background-color: #f7f9fc;
        border: 1px solid #e5e8ef;
        padding: 12px;
        border-radius: 12px;
    }

    .small-note {
        font-size: 0.85rem;
        color: #5f6b7a;
        line-height: 1.35;
    }

    .priority-box {
        background-color: #f7f9fc;
        border-left: 5px solid #4c78a8;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Load data
# ============================================================

@st.cache_data
def load_data():
    file_path = Path("final_dementia_physical_inactivity_clean.csv")

    if not file_path.exists():
        st.error(
            "The clean dataset file was not found. Please make sure "
            "`final_dementia_physical_inactivity_clean.csv` is in the same folder as app.py."
        )
        st.stop()

    df = pd.read_csv(file_path)

    required_cols = [
        "country", "iso3", "who_region", "year", "sex",
        "dementia_prevalence_rate", "dementia_incidence_rate", "dementia_death_rate",
        "physical_inactivity_pct",
        "pct_change_dementia_prevalence_since_2000",
        "pct_change_dementia_incidence_since_2000",
        "pct_change_dementia_death_since_2000",
        "pct_change_physical_inactivity_since_2000",
        "global_rank_dementia_prevalence",
        "global_rank_dementia_incidence",
        "global_rank_dementia_death",
        "regional_rank_dementia_prevalence",
        "regional_rank_dementia_incidence",
        "regional_rank_dementia_death",
        "priority_category_prevalence",
        "priority_category_incidence",
        "priority_category_death"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"The dataset is missing these required columns: {missing_cols}")
        st.stop()

    return df


df = load_data()

# ============================================================
# Helper dictionaries and functions
# ============================================================

measure_config = {
    "Prevalence": {
        "column": "dementia_prevalence_rate",
        "label": "Dementia prevalence rate",
        "pct_change_column": "pct_change_dementia_prevalence_since_2000",
        "global_rank_column": "global_rank_dementia_prevalence",
        "regional_rank_column": "regional_rank_dementia_prevalence",
        "priority_column": "priority_category_prevalence"
    },
    "Incidence": {
        "column": "dementia_incidence_rate",
        "label": "Dementia incidence rate",
        "pct_change_column": "pct_change_dementia_incidence_since_2000",
        "global_rank_column": "global_rank_dementia_incidence",
        "regional_rank_column": "regional_rank_dementia_incidence",
        "priority_column": "priority_category_incidence"
    },
    "Deaths": {
        "column": "dementia_death_rate",
        "label": "Dementia death rate",
        "pct_change_column": "pct_change_dementia_death_since_2000",
        "global_rank_column": "global_rank_dementia_death",
        "regional_rank_column": "regional_rank_dementia_death",
        "priority_column": "priority_category_death"
    }
}

sex_order = ["Both", "Male", "Female"]

sex_color_map = {
    "Both": "#6A5ACD",
    "Male": "#0072B2",
    "Female": "#E69F00"
}

priority_color_map = {
    "Above-average dementia + above-average inactivity": "#0072B2",
    "Above-average dementia + below-average inactivity": "#E69F00",
    "Below-average dementia + above-average inactivity": "#009E73",
    "Below-average dementia + below-average inactivity": "#999999",
    "Insufficient data": "#CCCCCC"
}

def format_number(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"

def format_pct(value, decimals=1):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}%"

def calculate_correlation(dataframe, x_col, y_col):
    temp = dataframe[[x_col, y_col]].dropna()
    if len(temp) < 2:
        return np.nan
    return temp[x_col].corr(temp[y_col])

# ============================================================
# Dashboard title and description
# ============================================================

st.title("Dementia Burden and Insufficient Physical Activity Dashboard")

st.markdown(
    """
    Dementia burden is a major global public-health issue, while insufficient physical activity remains a widespread
    and modifiable lifestyle risk factor. This dashboard compares country-level dementia burden rates with the
    prevalence of insufficient physical activity among adults to support public-health awareness, geographic
    prioritization, and evidence-informed decision-making.

    <span class="small-note">
    This dashboard explores population-level associations only. It does not prove that insufficient physical activity
    directly causes dementia. Dementia burden is measured using age-standardized rates to improve comparability across countries.
    </span>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Sidebar filters
# ============================================================

st.sidebar.header("Dashboard Filters")

year_options = sorted(df["year"].dropna().unique().astype(int).tolist())
default_year = 2022 if 2022 in year_options else max(year_options)
selected_year = st.sidebar.selectbox(
    "Year",
    options=year_options,
    index=year_options.index(default_year)
)

selected_measure_display = st.sidebar.selectbox(
    "Dementia measure",
    options=list(measure_config.keys()),
    index=0
)

selected_sex = st.sidebar.selectbox(
    "Sex",
    options=sex_order,
    index=0
)

region_options = ["All regions"] + sorted(df["who_region"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox(
    "WHO region",
    options=region_options,
    index=0
)

# Country dropdown is affected by the selected region.
if selected_region == "All regions":
    country_options = sorted(df["country"].dropna().unique().tolist())
else:
    country_options = sorted(
        df[df["who_region"] == selected_region]["country"].dropna().unique().tolist()
    )

default_country = "Lebanon"
default_country_index = country_options.index(default_country) if default_country in country_options else 0

selected_country = st.sidebar.selectbox(
    "Country",
    options=country_options,
    index=default_country_index
)

# Selected measure metadata
selected_measure_col = measure_config[selected_measure_display]["column"]
selected_measure_label = measure_config[selected_measure_display]["label"]
selected_pct_change_col = measure_config[selected_measure_display]["pct_change_column"]
selected_priority_col = measure_config[selected_measure_display]["priority_column"]

if selected_region == "All regions":
    selected_rank_col = measure_config[selected_measure_display]["global_rank_column"]
    rank_scope_label = "global"
else:
    selected_rank_col = measure_config[selected_measure_display]["regional_rank_column"]
    rank_scope_label = "regional"

# ============================================================
# Filtered dataframes
# ============================================================

comparison_df = df[
    (df["year"] == selected_year) &
    (df["sex"] == selected_sex)
].copy()

if selected_region != "All regions":
    comparison_df = comparison_df[comparison_df["who_region"] == selected_region].copy()

selected_country_row = df[
    (df["country"] == selected_country) &
    (df["year"] == selected_year) &
    (df["sex"] == selected_sex)
].copy()

country_trend_df = df[
    (df["country"] == selected_country) &
    (df["sex"] == selected_sex)
].sort_values("year").copy()

sex_comparison_df = df[
    (df["country"] == selected_country) &
    (df["year"] == selected_year)
].copy()

sex_comparison_df["sex"] = pd.Categorical(
    sex_comparison_df["sex"],
    categories=sex_order,
    ordered=True
)

sex_comparison_df = sex_comparison_df.sort_values("sex")

# ============================================================
# KPI cards
# ============================================================

if selected_country_row.empty:
    st.warning("No data is available for the selected country, year, and sex group.")
    st.stop()

selected_row = selected_country_row.iloc[0]

correlation_value = calculate_correlation(
    comparison_df,
    "physical_inactivity_pct",
    selected_measure_col
)

comparison_country_count = comparison_df["country"].nunique()

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

with kpi1:
    st.metric(
        selected_measure_label,
        format_number(selected_row[selected_measure_col], 2)
    )

with kpi2:
    st.metric(
        "Physical inactivity",
        format_pct(selected_row["physical_inactivity_pct"], 1)
    )

with kpi3:
    st.metric(
        "Dementia change since 2000",
        format_pct(selected_row[selected_pct_change_col], 1)
    )

with kpi4:
    st.metric(
        "Inactivity change since 2000",
        format_pct(selected_row["pct_change_physical_inactivity_since_2000"], 1)
    )

with kpi5:
    st.metric(
        f"{rank_scope_label.title()} dementia rank",
        f"#{int(selected_row[selected_rank_col])} of {comparison_country_count}"
    )

with kpi6:
    st.metric(
        "Association r",
        "N/A" if pd.isna(correlation_value) else f"{correlation_value:.2f}"
    )

priority_value = selected_row[selected_priority_col]

st.markdown(
    f"""
    <div class="priority-box">
    <b>Selected country:</b> {selected_country} |
    <b>Year:</b> {selected_year} |
    <b>Sex:</b> {selected_sex} |
    <b>Priority category:</b> {priority_value}
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Row 1 visuals: map and two trend charts
# ============================================================

row1_col1, row1_col2, row1_col3 = st.columns([1.15, 1, 1])

with row1_col1:
    st.subheader("Geographical Map of Dementia Burden")

    fig_map = px.choropleth(
        comparison_df,
        locations="iso3",
        color=selected_measure_col,
        hover_name="country",
        hover_data={
            "iso3": False,
            "who_region": True,
            selected_measure_col: ":.2f",
            "physical_inactivity_pct": ":.2f"
        },
        color_continuous_scale="Viridis",
        projection="natural earth",
        labels={
            selected_measure_col: selected_measure_label,
            "who_region": "WHO region",
            "physical_inactivity_pct": "Physical inactivity (%)"
        }
    )

    fig_map.update_geos(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="LightGray"
    )

    fig_map.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(title=selected_measure_label)
    )

    st.plotly_chart(fig_map, use_container_width=True)

with row1_col2:
    st.subheader("Dementia Burden Trend")

    fig_dementia_trend = px.line(
        country_trend_df,
        x="year",
        y=selected_measure_col,
        markers=True,
        labels={
            "year": "Year",
            selected_measure_col: selected_measure_label
        }
    )

    fig_dementia_trend.add_vline(
        x=selected_year,
        line_width=1,
        line_dash="dash",
        line_color="gray"
    )

    fig_dementia_trend.update_traces(
        line=dict(width=3),
        marker=dict(size=6)
    )

    fig_dementia_trend.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Year",
        yaxis_title=selected_measure_label
    )

    st.plotly_chart(fig_dementia_trend, use_container_width=True)

with row1_col3:
    st.subheader("Physical Inactivity Trend")

    fig_inactivity_trend = px.line(
        country_trend_df,
        x="year",
        y="physical_inactivity_pct",
        markers=True,
        labels={
            "year": "Year",
            "physical_inactivity_pct": "Physical inactivity (%)"
        }
    )

    fig_inactivity_trend.add_vline(
        x=selected_year,
        line_width=1,
        line_dash="dash",
        line_color="gray"
    )

    fig_inactivity_trend.update_traces(
        line=dict(width=3),
        marker=dict(size=6)
    )

    fig_inactivity_trend.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Year",
        yaxis_title="Physical inactivity (%)"
    )

    st.plotly_chart(fig_inactivity_trend, use_container_width=True)

# ============================================================
# Row 2 visuals: scatterplot, top 10, sex comparison
# ============================================================

row2_col1, row2_col2, row2_col3 = st.columns([1.1, 1, 0.9])

with row2_col1:
    st.subheader("Physical Inactivity vs Dementia Burden")

    fig_scatter = px.scatter(
        comparison_df,
        x="physical_inactivity_pct",
        y=selected_measure_col,
        color=selected_priority_col,
        hover_name="country",
        hover_data={
            "who_region": True,
            "physical_inactivity_pct": ":.2f",
            selected_measure_col: ":.2f",
            selected_priority_col: True
        },
        color_discrete_map=priority_color_map,
        labels={
            "physical_inactivity_pct": "Physical inactivity (%)",
            selected_measure_col: selected_measure_label,
            selected_priority_col: "Priority category",
            "who_region": "WHO region"
        }
    )

    # Add a simple trend line if enough data points exist.
    scatter_temp = comparison_df[["physical_inactivity_pct", selected_measure_col]].dropna()

    if len(scatter_temp) >= 2:
        x = scatter_temp["physical_inactivity_pct"].values
        y = scatter_temp[selected_measure_col].values

        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        fig_scatter.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Trend line",
                line=dict(color="black", dash="dash", width=2)
            )
        )

    fig_scatter.update_layout(
        height=390,
        margin=dict(l=0, r=0, t=10, b=0),
        legend_title_text="Priority category",
        xaxis_title="Physical inactivity (%)",
        yaxis_title=selected_measure_label
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown(
        '<p class="small-note">This visual shows population-level association only, not causation.</p>',
        unsafe_allow_html=True
    )

with row2_col2:
    st.subheader("Top 10 Countries")

    top10_df = comparison_df.nlargest(10, selected_measure_col).copy()
    top10_df = top10_df.sort_values(selected_measure_col, ascending=False)

    fig_top10 = go.Figure()

    fig_top10.add_trace(
        go.Bar(
            x=top10_df["country"],
            y=top10_df[selected_measure_col],
            name=selected_measure_label,
            marker_color="#4C78A8",
            yaxis="y1"
        )
    )

    fig_top10.add_trace(
        go.Scatter(
            x=top10_df["country"],
            y=top10_df["physical_inactivity_pct"],
            name="Physical inactivity (%)",
            mode="lines+markers",
            marker=dict(size=8, color="#E69F00"),
            line=dict(width=3, color="#E69F00"),
            yaxis="y2"
        )
    )

    fig_top10.update_layout(
        height=390,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="Country", tickangle=-35),
        yaxis=dict(title=selected_measure_label, side="left"),
        yaxis2=dict(
            title="Physical inactivity (%)",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    st.plotly_chart(fig_top10, use_container_width=True)

with row2_col3:
    st.subheader("Physical Inactivity by Sex")

    fig_sex = px.bar_polar(
        sex_comparison_df,
        r="physical_inactivity_pct",
        theta="sex",
        color="sex",
        color_discrete_map=sex_color_map,
        labels={
            "physical_inactivity_pct": "Physical inactivity (%)",
            "sex": "Sex"
        }
    )

    max_inactivity = sex_comparison_df["physical_inactivity_pct"].max()

    fig_sex.update_layout(
        height=390,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=True,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(10, max_inactivity + 10)]
            )
        )
    )

    st.plotly_chart(fig_sex, use_container_width=True)

    st.markdown(
        '<p class="small-note">These are physical inactivity rates by sex. They are not additive shares.</p>',
        unsafe_allow_html=True
    )

# ============================================================
# Footer notes
# ============================================================

st.markdown("---")

st.markdown(
    """
    <span class="small-note">
    <b>Data sources:</b> IHME/GBD dementia burden data and WHO physical inactivity data.
    Dementia measures are age-standardized rates. Physical inactivity represents the percentage of adults aged 18+
    who are insufficiently physically active. This dashboard supports exploratory public-health analysis and should
    not be interpreted as causal evidence.
    </span>
    """,
    unsafe_allow_html=True
)
