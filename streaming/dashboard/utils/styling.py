import streamlit as st

PRIMARY_RED = "#E50914"
PANEL_BG = "#F5F5F7"
CARD_BG = "#FFFFFF"

SLICE_COLORS = {
    "eMBB":  "#1f77b4",
    "URLLC": "#d62728",
    "mMTC":  "#9467bd",
}

SEVERITY_COLORS = {
    "Critical": "#E50914",
    "Medium":   "#F5A623",
    "Low":      "#2BB673",
    "Healthy":  "#2BB673",
    "Degraded": "#F5A623",
}

_CSS = """
<style>
/* Red sidebar */
section[data-testid="stSidebar"] {
    background-color: #E50914;
}
section[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

/* Page links as rounded white buttons */
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul {
    padding-top: 1rem;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
    background-color: #FFFFFF !important;
    border-radius: 22px !important;
    padding: 0.55rem 1rem !important;
    margin: 0.35rem 0.5rem !important;
    text-align: center;
    font-weight: 600;
    transition: background-color 0.15s ease;
    border: 2px solid #FFFFFF;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a span {
    color: #E50914 !important;
    font-weight: 700;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
    background-color: #f9d7d9 !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
    background-color: #1F1F2E !important;
    border-color: #1F1F2E !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] span {
    color: #FFFFFF !important;
}

/* Filter boxes in sidebar */
section[data-testid="stSidebar"] [data-testid="stMultiSelect"] label,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #FFFFFF !important;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #1F1F2E !important;
}

/* Panel background */
section.main > div.block-container {
    background-color: #F5F5F7;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    border-radius: 8px;
}

/* KPI cards */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 1rem 1.25rem;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid #EEE;
}
div[data-testid="stMetricLabel"] {
    font-weight: 600;
    color: #555 !important;
}
div[data-testid="stMetricValue"] {
    color: #1F1F2E;
}

/* Plotly chart container cards */
div.element-container:has(> div.js-plotly-plot) {
    background-color: #FFFFFF;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    border: 1px solid #EEE;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 0.5rem;
}

/* Section titles */
h1, h2, h3, h4 { color: #1F1F2E; }

/* Sidebar Refresh button: red bg, white border + text */
section[data-testid="stSidebar"] .stButton > button {
    background-color: #E50914 !important;
    color: #FFFFFF !important;
    border: 2px solid #FFFFFF !important;
    border-radius: 22px !important;
    font-weight: 700 !important;
    padding: 0.45rem 1rem !important;
    transition: background-color 0.15s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #c40810 !important;
    color: #FFFFFF !important;
    border-color: #FFFFFF !important;
}
section[data-testid="stSidebar"] .stButton > button:focus,
section[data-testid="stSidebar"] .stButton > button:active {
    background-color: #c40810 !important;
    color: #FFFFFF !important;
    border-color: #FFFFFF !important;
    box-shadow: none !important;
}

/* Hide streamlit chrome */
#MainMenu, footer { visibility: hidden; }
</style>
"""


def inject_style():
    st.markdown(_CSS, unsafe_allow_html=True)


def section_title(text: str):
    st.markdown(
        f"<h4 style='color:#1F1F2E;margin:0.25rem 0 0.5rem 0;'>{text}</h4>",
        unsafe_allow_html=True,
    )


def slice_color(slice_type: str) -> str:
    return SLICE_COLORS.get(slice_type, "#888888")


def severity_color(severity: str) -> str:
    return SEVERITY_COLORS.get(severity, "#888888")


def style_severity_df(df, col: str = "severity"):
    def _colorize(val):
        color = severity_color(val)
        return f"background-color: {color}; color: white; font-weight: 600; text-align: center; border-radius: 4px;"
    return df.style.applymap(_colorize, subset=[col])
