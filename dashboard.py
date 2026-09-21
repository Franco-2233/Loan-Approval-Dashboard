import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import f_oneway, chi2_contingency

st.set_page_config(
    page_title="Loan Approval EDA Dashboard",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Executive Banking visual theme
BANK_RED = "#e61a2d"
BANK_GREEN = "#00a44d"
NAVY = "#102033"
SLATE = "#536273"
INK = "#17212b"
MUTED = "#7a8794"
SURFACE = "#F5EBEB"
BG = "#BBBBBB"
BORDER = "#e3e7eb"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: {BG};
        color: {INK};
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stSidebar"] {{
        background: {NAVY};
        border-right: 0;
    }}

    [data-testid="stSidebar"] * {{
        color: #f4f7fa !important;
    }}

    [data-testid="stSidebar"] .stMarkdown p {{
        color: #b9c5d1 !important;
    }}

    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="multiselect"] > div {{
        background: #1b2c40;
        border-color: #314459;
    }}

    [data-testid="stSidebar"] hr {{
        border-color: #314459;
    }}

    .exec-brand {{
        padding: 8px 0 22px 0;
        border-bottom: 1px solid #314459;
        margin-bottom: 22px;
    }}

    .exec-brand .mark {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: {BANK_RED};
        font-weight: 800;
        font-size: 18px;
        margin-right: 10px;
    }}

    .exec-brand .name {{
        font-size: 16px;
        font-weight: 800;
        letter-spacing: .02em;
    }}

    .exec-brand .sub {{
        color: #9eacba !important;
        font-size: 11px;
        margin-left: 49px;
        margin-top: -8px;
    }}

    h1 {{
        color: {NAVY} !important;
        font-weight: 800 !important;
        letter-spacing: -0.035em;
        margin-bottom: 2px !important;
    }}

    h2, h3 {{
        color: {NAVY} !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }}

    .exec-kicker {{
        color: {BANK_RED};
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }}

    .exec-subtitle {{
        color: {SLATE};
        font-size: 13px;
        margin-bottom: 20px;
    }}

    [data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 17px 18px;
        box-shadow: 0 2px 8px rgba(16,32,51,.04);
        position: relative;
        overflow: hidden;
    }}

    [data-testid="stMetric"]:before {{
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: {BANK_RED};
    }}

    [data-testid="stMetric"] label {{
        color: {MUTED} !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: .06em;
    }}

    [data-testid="stMetricValue"] {{
        color: {NAVY} !important;
        font-size: 25px !important;
        font-weight: 800 !important;
    }}

    [data-testid="stMetricDelta"] {{
        font-size: 11px !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: transparent;
        border-bottom: 1px solid {BORDER};
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {SLATE};
        font-weight: 600;
        font-size: 12px;
        padding: 11px 15px;
    }}

    .stTabs [aria-selected="true"] {{
        color: {BANK_RED} !important;
    }}

    .stTabs [data-baseweb="tab-highlight"] {{
        background: {BANK_RED};
    }}

    .stButton > button {{
        border-radius: 8px;
        border: 1px solid {BORDER};
        font-weight: 600;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        overflow: hidden;
    }}

    .stAlert {{
        border-radius: 10px;
    }}

    footer {{
        visibility: hidden;
    }}

    /* Plotly containers */
    [data-testid="stPlotlyChart"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 6px;
        box-shadow: 0 2px 8px rgba(16,32,51,.03);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


DATA_FILE = "loan_dashboard_data.xlsx"

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_FILE)
    df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
    df["application_year"] = pd.to_numeric(
        df["application_year"], errors="coerce"
    ).astype("Int64")
    return df

df = load_data()

# Missing loan_status is displayed as Review ONLY in the dashboard.
df["loan_status_display"] = df["loan_status"].fillna("Review")

FEATURES = [
    "age", "gender", "marital_status", "education_level", "annual_income",
    "loan_amount", "credit_score", "employment_years", "employment_type",
    "loan_purpose", "home_ownership", "debt_to_income_ratio",
    "existing_loans", "loan_term_months", "interest_rate", "loan_status",
    "application_date", "application_year"
]
NUMERIC = [
    "age", "annual_income", "loan_amount", "credit_score", "employment_years",
    "debt_to_income_ratio", "existing_loans", "loan_term_months", "interest_rate"
]
CATEGORICAL = [
    "gender", "marital_status", "education_level", "employment_type",
    "loan_purpose", "home_ownership", "loan_status_display"
]

st.markdown('<div class="exec-kicker">Executive Banking Intelligence</div>', unsafe_allow_html=True)
st.title("Loan Portfolio Executive Dashboard")
st.markdown(
    '<div class="exec-subtitle">Portfolio performance, applicant quality and credit-risk indicators at a glance.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown(
        '<div class="exec-brand">'
        '<div><span class="mark">B</span><span class="name">BANKING DASHBOARD</span></div>'
        '<div class="sub">LOAN APPLICATIONS</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.header("Portfolio Filters")

    status_options = ["Approved", "Rejected", "Review"]
    selected_status = st.multiselect(
        "Loan Status",
        status_options,
        default=status_options
    )

    year_options = sorted(df["application_year"].dropna().unique().tolist())
    selected_years = st.multiselect(
        "Application Year",
        year_options,
        default=year_options
    )


filtered = df[
    df["loan_status_display"].isin(selected_status)
    & df["application_year"].isin(selected_years)
].copy()

# KPI calculations
total = len(filtered)
approved = (filtered["loan_status_display"] == "Approved").sum()
rejected = (filtered["loan_status_display"] == "Rejected").sum()
review = (filtered["loan_status_display"] == "Review").sum()
known = approved + rejected
approval_rate = approved / known * 100 if known else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Applications", f"{total:,}")
c2.metric("Approved", f"{approved:,}")
c3.metric("Rejected", f"{rejected:,}")
c4.metric("Review", f"{review:,}")
c5.metric("Approval Rate", f"{approval_rate:.1f}%")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    " Overview", " Applicant Profile", " Financial Analysis",
    " Correlations", " Data Quality", " Driver Analysis"
])

def banking_layout(fig, height=None):
    """Apply the executive banking chart language without changing analytical data."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, Arial, sans-serif", color=INK, size=11),
        title=dict(
            font=dict(size=14, color=NAVY),
            x=0.02,
            xanchor="left",
        ),
        margin=dict(l=45, r=25, t=55, b=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=10, color=SLATE),
        ),
        hoverlabel=dict(bgcolor=NAVY, font_color="white"),
        height=height,
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=BORDER,
        tickfont=dict(color=SLATE, size=10),
        title_font=dict(color=SLATE, size=10),
    )
    fig.update_yaxes(
        gridcolor="#edf0f2",
        zeroline=False,
        linecolor=BORDER,
        tickfont=dict(color=SLATE, size=10),
        title_font=dict(color=SLATE, size=10),
    )
    return fig


def apply_status_colors(fig):
    """Use the requested red/green palette for loan outcomes."""
    fig.update_traces(
        marker=dict(
            color=[
                BANK_GREEN if str(v).lower() == "approved"
                else BANK_RED if str(v).lower() == "rejected"
                else "#9aa6b2"
                for v in fig.data[0].x
            ]
        )
    )
    return fig


with tab1:
    st.subheader("Loan Status Overview")

    col1, col2 = st.columns(2)

    status_counts = (
        filtered["loan_status_display"]
        .value_counts()
        .reindex(status_options, fill_value=0)
        .reset_index()
    )
    status_counts.columns = ["Loan Status", "Count"]

    with col1:
        fig = px.bar(
            status_counts, x="Loan Status", y="Count",
            text="Count", title="Applications by Loan Status"
        )
        fig.update_traces(
            textposition="outside",
            marker=dict(color=[BANK_GREEN if s == "Approved" else BANK_RED if s == "Rejected" else "#9aa6b2"
                               for s in status_counts["Loan Status"]])
        )
        banking_layout(fig)
        banking_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            status_counts, names="Loan Status", values="Count",
            hole=0.58, title="Loan Status Distribution",
            color="Loan Status",
            color_discrete_map={
                "Approved": BANK_GREEN,
                "Rejected": BANK_RED,
                "Review": "#9aa6b2",
            }
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        banking_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Application Trends")
    yearly = (
        filtered.groupby(["application_year", "loan_status_display"])
        .size().reset_index(name="Applications")
    )
    fig = px.line(
        yearly, x="application_year", y="Applications",
        color="loan_status_display", markers=True,
        title="Applications by Year and Loan Status"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Applicant Characteristics")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            filtered, x="age", color="loan_status_display",
            nbins=25, marginal="box", barmode="overlay",
            title="Age Distribution by Loan Status"
        )
        banking_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        edu = (
            filtered.groupby(["education_level", "loan_status_display"])
            .size().reset_index(name="Applications")
        )
        fig = px.bar(
            edu, x="education_level", y="Applications",
            color="loan_status_display", barmode="group",
            title="Education Level vs Loan Status"
        )
        banking_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        emp = (
            filtered.groupby(["employment_type", "loan_status_display"])
            .size().reset_index(name="Applications")
        )
        fig = px.bar(
            emp, x="employment_type", y="Applications",
            color="loan_status_display", barmode="group",
            title="Employment Type vs Loan Status"
        )
        banking_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        home = (
            filtered.groupby(["home_ownership", "loan_status_display"])
            .size().reset_index(name="Applications")
        )
        fig = px.bar(
            home, x="home_ownership", y="Applications",
            color="loan_status_display", barmode="group",
            title="Home Ownership vs Loan Status"
        )
        banking_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Financial and Loan Characteristics")

    status_color_map = {
        "Approved": BANK_GREEN,
        "Rejected": BANK_RED,
        "Review": "#9aa6b2",
    }

    def violin_by_status(y_col, title):
        fig = px.violin(
            filtered, x="loan_status_display", y=y_col,
            color="loan_status_display",
            color_discrete_map=status_color_map,
            box=True, points=False,
            title=title
        )
        fig.update_traces(meanline_visible=True, showlegend=False)
        banking_layout(fig)
        return fig

    col1, col2 = st.columns(2)

    with col1:
        fig = violin_by_status("credit_score", "Credit Score by Loan Status")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = violin_by_status("annual_income", "Annual Income by Loan Status")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = violin_by_status("loan_amount", "Loan Amount by Loan Status")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = violin_by_status("debt_to_income_ratio", "Debt-to-Income Ratio by Loan Status")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Credit Score vs Loan Amount")
    plot_df = filtered.dropna(subset=["credit_score", "loan_amount"])
    fig = px.scatter(
        plot_df, x="credit_score", y="loan_amount",
        color="loan_status_display",
        hover_data=["annual_income", "debt_to_income_ratio", "loan_purpose"],
        title="Credit Score vs Loan Amount"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Correlation Analysis")

    corr = filtered[NUMERIC].corr(numeric_only=True)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            zmin=-1, zmax=1,
            colorscale=[
                [0.0, BANK_RED],
                [0.5, "#ffffff"],
                [1.0, BANK_GREEN],
            ],
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{x} × %{y}: %{z:.2f}<extra></extra>"
        )
    )
    fig.update_layout(
        title="Correlation Matrix — Numerical Attributes",
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Interpretation:** Values close to +1 indicate a strong positive relationship, "
        "values close to -1 indicate a strong negative relationship, and values close to "
        "0 indicate a weak linear relationship. Correlation does not establish causation."
    )

    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            pairs.append({
                "Attribute 1": corr.columns[i],
                "Attribute 2": corr.columns[j],
                "Absolute Correlation": abs(corr.iloc[i, j]),
                "Correlation": corr.iloc[i, j]
            })

    pairs_df = pd.DataFrame(pairs).sort_values(
        "Absolute Correlation", ascending=False
    ).head(10)

    st.subheader("Top 10 Strongest Numerical Relationships")
    st.dataframe(
        pairs_df.drop(columns="Absolute Correlation").style.format(
            {"Correlation": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True
    )

with tab5:
    st.subheader("Data Quality")

    missing = (
        df[FEATURES]
        .isna()
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    missing.columns = ["Attribute", "Missing Values"]
    missing["Missing %"] = missing["Missing Values"] / len(df) * 100

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            missing[missing["Missing Values"] > 0],
            x="Missing Values", y="Attribute",
            orientation="h",
            title="Missing Values by Attribute"
        )
        banking_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Rows", f"{len(df):,}")
        st.metric("Columns", f"{len(df.columns):,}")
        st.metric("Duplicate Rows", f"{df.duplicated().sum():,}")
        st.metric("Total Missing Cells", f"{int(df[FEATURES].isna().sum().sum()):,}")

    st.subheader("Missing-Value Summary")
    st.dataframe(
        missing.style.format({"Missing %": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True
    )

with tab6:
    st.subheader("Driver Analysis — Does Any Attribute Predict Loan Outcome?")
    st.markdown(
        "Correlation and chi-square only flag *associations*, not causation. Below, each "
        "attribute is tested against **Loan Status** directly: ANOVA/eta-squared for numeric "
        "fields, chi-square/Cramér's V for categorical fields. A low p-value with a low "
        "effect size means a difference is *detectable* at this sample size but not "
        "*meaningful* for decisions."
    )

    dr_numeric = filtered[NUMERIC].columns.tolist() if False else NUMERIC
    num_results = []
    for col in NUMERIC:
        sub = filtered[[col, "loan_status_display"]].dropna()
        grps = [g[col].values for _, g in sub.groupby("loan_status_display") if len(g) > 1]
        if len(grps) < 2:
            continue
        f_stat, p_val = f_oneway(*grps)
        grand_mean = sub[col].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in grps)
        ss_total = sum((sub[col] - grand_mean) ** 2)
        eta_sq = ss_between / ss_total if ss_total else 0
        num_results.append({
            "Attribute": col, "Test": "ANOVA (F)", "Statistic": f_stat,
            "p-value": p_val, "Effect Size (η²)": eta_sq,
            "Significant (p<0.05)": "Yes" if p_val < 0.05 else "No"
        })
    num_df = pd.DataFrame(num_results).sort_values("Effect Size (η²)", ascending=False)

    cat_results = []
    for col in CATEGORICAL:
        if col == "loan_status_display":
            continue
        ct = pd.crosstab(filtered[col], filtered["loan_status_display"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p_val, dof, _ = chi2_contingency(ct)
        n = ct.sum().sum()
        k = min(ct.shape) - 1
        cramers_v = (chi2 / (n * k)) ** 0.5 if k > 0 and n > 0 else 0
        cat_results.append({
            "Attribute": col, "Test": "Chi-square (χ²)", "Statistic": chi2,
            "p-value": p_val, "Effect Size (Cramér's V)": cramers_v,
            "Significant (p<0.05)": "Yes" if p_val < 0.05 else "No"
        })
    cat_df = pd.DataFrame(cat_results).sort_values("Effect Size (Cramér's V)", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Numeric Attributes vs. Loan Status**")
        st.dataframe(
            num_df.style.format({
                "Statistic": "{:.3f}", "p-value": "{:.4f}", "Effect Size (η²)": "{:.5f}"
            }),
            use_container_width=True, hide_index=True
        )

    with col2:
        st.markdown("**Categorical Attributes vs. Loan Status**")
        if not cat_df.empty:
            st.dataframe(
                cat_df.style.format({
                    "Statistic": "{:.3f}", "p-value": "{:.4f}",
                    "Effect Size (Cramér's V)": "{:.5f}"
                }),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("Not enough category variety in the current filter selection to test.")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=num_df["Attribute"], y=num_df["Effect Size (η²)"],
        name="Numeric (η²)", marker_color=NAVY
    ))
    if not cat_df.empty:
        fig.add_trace(go.Bar(
            x=cat_df["Attribute"], y=cat_df["Effect Size (Cramér's V)"],
            name="Categorical (Cramér's V)", marker_color=BANK_RED
        ))
    fig.update_layout(
        title="Effect Size by Attribute (higher = stronger relationship with Loan Status)",
        yaxis_title="Effect Size",
    )
    banking_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Interpretation guide:** effect sizes are conventionally read as "
        "small (η² / Cramér's V ≈ 0.01), medium (≈ 0.06), and large (≈ 0.14+). "
        "If every bar above sits near zero, it means loan outcome in this dataset is "
        "effectively independent of applicant attributes — consistent with randomly "
        "assigned or synthetic outcome labels rather than real underwriting decisions."
    )

st.divider()
st.markdown("---")
st.markdown(
    '<div style="text-align:right;color:#7a8794;font-size:10px;letter-spacing:.05em;">'
    'EXECUTIVE BANKING INTELLIGENCE • LOAN MONITORING'
    '</div>',
    unsafe_allow_html=True
)
st.caption(
    "Analytical note: customer_id is treated as an identifier and is not included "
    "in numerical correlation analysis. loan_status is the outcome variable."
)
