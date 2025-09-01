import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import requests
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import numpy as np
from dateutil import parser as du


st.set_page_config(layout="wide")


# --------- Light theme (COMC-inspired) ----------
COMC_RED    = "#D63A2F"   # accent
COMC_NAVY   = "#0F2746"   # headings
COMC_BG     = "#FFFFFF"   # page
COMC_PANEL  = "#F5F6F8"   # cards / panels
COMC_TEXT   = "#111111"   # body text
COMC_BORDER = "#E5E7EB"   # borders
COMC_CONTROL= "#F0F0F0"   # inputs

st.markdown(f"""
<style>
:root {{
  --comc-panel: {COMC_PANEL};
  --comc-border: {COMC_BORDER};
}}
/* Global */
html, body, [data-testid="stAppViewContainer"] {{
  background: {COMC_BG};
  color: {COMC_TEXT};
}}
div.block-container {{ padding-top: 1rem; padding-bottom: 2rem; }}
/* Header */
header[data-testid="stHeader"] {{ background: {COMC_BG} !important; border-bottom: none !important; }}
header[data-testid="stHeader"] * {{ color: {COMC_TEXT} !important; }}
/* Titles/text */
h1, h2, h3, h4, h5, h6 {{ color: {COMC_RED} !important; letter-spacing: .2px; }}
p, label, span, div, td, th {{ color: {COMC_TEXT} !important; }}
/* Sidebar */
section[data-testid="stSidebar"] {{ width: 260px !important; background: {COMC_PANEL}; border-right: 1px solid {COMC_BORDER}; }}
@media (min-width: 1024px) {{ section[data-testid="stSidebar"] > div {{ width: 260px !important; }} }}
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stRadio,
section[data-testid="stSidebar"] .stMultiSelect,
section[data-testid="stSidebar"] .stNumberInput,
section[data-testid="stSidebar"] .stDateInput,
section[data-testid="stSidebar"] .stTextInput {{ margin-bottom: .5rem; }}
/* Inputs */
div[data-baseweb="select"] > div,
.stDateInput input[type="text"],
.StNumberInput input, .stTextInput input,
.stMultiSelect, .stSelectbox {{
  background-color: {COMC_CONTROL} !important;
  color: {COMC_TEXT} !important;
  border: 1px solid {COMC_BORDER} !important;
  box-shadow: none !important; outline: none !important;
}}
/* Number input */
[data-testid="stNumberInput"] > div {{ background: {COMC_CONTROL} !important; border: 1px solid {COMC_BORDER} !important; border-radius: 12px !important; box-shadow: none !important; }}
[data-testid="stNumberInput"] input {{ background: {COMC_CONTROL} !important; color: {COMC_TEXT} !important; }}
[data-testid="stNumberInput"] button {{ background: {COMC_CONTROL} !important; color: {COMC_TEXT} !important; border-left: 1px solid {COMC_BORDER} !important; box-shadow: none !important; }}
[data-testid="stNumberInput"] button:hover {{ background: #E9ECEF !important; }}
[data-testid="stNumberInput"] svg {{ fill: {COMC_TEXT} !important; color: {COMC_TEXT} !important; }}
/* Date inputs */
.stDateInput > div, .stDateInput > div:focus, .stDateInput > div:active, .stDateInput > div:focus-within, .stDateInput input:focus {{
  background: {COMC_CONTROL} !important; border: 1px solid {COMC_BORDER} !important; border-radius: 12px !important; box-shadow: none !important; outline: none !important;
}}
.stDateInput > div > div {{ box-shadow: none !important; }}
.stDateInput button {{ background: {COMC_CONTROL} !important; border: 1px solid {COMC_BORDER} !important; box-shadow: none !important; }}
.stDateInput button svg {{ fill: {COMC_TEXT} !important; color: {COMC_TEXT} !important; }}
/* Popovers */
div[data-baseweb="popover"], div[data-baseweb="popover"] * {{ background: {COMC_CONTROL} !important; color: {COMC_TEXT} !important; }}
ul[role="listbox"], li[role="option"] {{ background: {COMC_CONTROL} !important; color: {COMC_TEXT} !important; border-color: {COMC_BORDER} !important; }}
li[role="option"][aria-selected="true"], li[role="option"]:hover {{ background: #E9ECEF !important; color: {COMC_TEXT} !important; }}
body [role="listbox"], body [role="dialog"], body [data-baseweb="menu"] {{ background: {COMC_CONTROL} !important; color: {COMC_TEXT} !important; border: 1px solid {COMC_BORDER} !important; }}
/* Icons */
[data-baseweb="select"] svg, .stMultiSelect svg, .stSelectbox svg {{ fill: {COMC_TEXT} !important; color: {COMC_TEXT} !important; }}
/* Expanders */
div.streamlit-expanderHeader {{ color: {COMC_NAVY} !important; background: {COMC_PANEL} !important; font-weight: 600; border-radius: 8px; }}
div.streamlit-expanderContent {{ background: {COMC_PANEL} !important; border: 1px solid {COMC_BORDER} !important; border-top: 0 !important; border-radius: 0 0 8px 8px !important; }}
/* File uploader */
[data-testid="stFileUploader"] {{ background: {COMC_PANEL} !important; border: 1px solid {COMC_BORDER} !important; border-radius: 12px !important; padding: 1rem !important; color: {COMC_TEXT} !important; }}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"], [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] section {{ background: {COMC_CONTROL} !important; border: 1px dashed {COMC_BORDER} !important; border-radius: 10px !important; color: {COMC_TEXT} !important; }}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] * {{ background: transparent !important; color: {COMC_TEXT} !important; }}
[data-testid="stFileUploader"] button {{ background: {COMC_RED} !important; color: #fff !important; border: 0 !important; border-radius: 10px !important; }}
a[download], .stDownloadButton button, [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {{ background: {COMC_CONTROL} !important; color: {COMC_TEXT} !important; border: 1px solid {COMC_BORDER} !important; border-radius: 10px !important; }}
/* Callout */
.comc-callout {{ background: {COMC_PANEL}; color: {COMC_TEXT}; border: 1px solid {COMC_BORDER}; border-radius: 12px; padding: 1rem 1.25rem; }}
/* Alerts */
[data-testid="stAlertInfo"] {{ background: {COMC_PANEL} !important; color: {COMC_TEXT} !important; border: 1px solid {COMC_BORDER} !important; border-radius: 12px !important; padding: 1rem !important; box-shadow: none !important; }}
[data-testid="stAlertInfo"]::before {{ display: none !important; }}
[data-testid="stAlertInfo"] * {{ color: {COMC_TEXT} !important; }}
[data-testid="stAlertInfo"] svg {{ fill: {COMC_TEXT} !important; color: {COMC_TEXT} !important; }}
/* Checkboxes */
.stCheckbox label {{ display: flex !important; align-items: center !important; white-space: nowrap !important; }}
/* Tables */
[data-testid="stTable"], .stDataFrame {{ background: {COMC_PANEL} !important; }}
.stDataFrame td, .stDataFrame th {{ background: {COMC_PANEL} !important; color: {COMC_TEXT} !important; border-color: {COMC_BORDER} !important; }}
[data-testid="stTable"] table {{ background: {COMC_PANEL} !important; }}
.stDataFrame [role="columnheader"], .stDataFrame [role="gridcell"] {{ background: {COMC_PANEL} !important; color: {COMC_TEXT} !important; border-color: {COMC_BORDER} !important; }}
.stDataFrame [role="gridcell"]:hover {{ background: #E9ECEF !important; }}
/* Scrollbars */
*::-webkit-scrollbar {{ width: 10px; height: 10px; }}
*::-webkit-scrollbar-track {{ background: {COMC_PANEL}; }}
*::-webkit-scrollbar-thumb {{ background: {COMC_BORDER}; border-radius: 10px; }}
* {{ scrollbar-color: {COMC_BORDER} {COMC_PANEL}; }}
/* KPI cards */
.kpi-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin: 1rem 0 1.25rem 0; }}
@media (max-width: 1200px) {{ .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 700px) {{ .kpi-grid {{ grid-template-columns: 1fr; }} }}
.kpi-card {{ background: var(--comc-panel); border: 1px solid var(--comc-border); border-radius: 16px; padding: 18px 20px; position: relative; box-shadow: 0 2px 6px rgba(16,24,40,0.05); }}
.kpi-card:before {{ content: ""; position: absolute; left: 0; top: 0; width: 6px; height: 100%; border-radius: 16px 0 0 16px; background: {COMC_RED}; }}
.kpi-card[data-accent="profit"]:before {{ background: {COMC_RED}; }}
.kpi-card[data-accent="volume"]:before {{ background: {COMC_NAVY}; }}
.kpi-card[data-accent="time"]:before {{ background: #8A8F98; }}
.kpi-title {{ font-size: .95rem; letter-spacing:.2px; margin: 0 0 6px 0; color: {COMC_NAVY}; font-weight: 600; }}
.kpi-value {{ font-size: 2.1rem; line-height: 1.1; font-weight: 700; margin: 0 0 6px 0; }}
.kpi-sub {{ font-size: .85rem; color: #555; margin: 0; }}
.kpi-delta {{ display:inline-block; font-size:.8rem; font-weight:600; padding:.1rem .45rem; border-radius:999px; margin-left:.5rem; background:#EAF7EE; color:#127A2E; }}
.kpi-delta.neg {{ background:#FDECEC; color:#AD1E1E; }}

/* Fullscreen (expand) modal – force light theme */
[data-testid="stModal"],
[data-testid="stModal"] * {{
  background: {COMC_BG} !important;
  color: {COMC_TEXT} !important;
}}
/* Some Streamlit versions render a separate dialog container */
[role="dialog"][aria-modal="true"],
[role="dialog"][aria-modal="true"] * {{
  background: {COMC_BG} !important;
  color: {COMC_TEXT} !important;
}}
/* Backdrop overlay behind the modal */
[data-testid="stModal"] > div:first-child,
[aria-modal="true"] + div[tabindex="-1"],
div[role="presentation"][style*="opacity"] {{
  background: rgba(255,255,255,0.95) !important;
}}
/* Ensure the matplotlib canvas/image rests on white */
[data-testid="stModal"] canvas,
[data-testid="stModal"] img {{
  background: {COMC_BG} !important;
}}
/* Hide the element toolbar / fullscreen across Streamlit versions */
html body [data-testid="stElementToolbar"] {{ display: none !important; }}
html body [data-testid="StyledFullScreenButton"] {{ display: none !important; }}
html body button[title="View fullscreen"] {{ display: none !important; }}
html body button[aria-label="View fullscreen"] {{ display: none !important; }}
html body [role="button"][aria-label="View fullscreen"] {{ display: none !important; }}
html body button[aria-label*="full"] {{ display: none !important; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------- Helpers ----------------------------
REQUIRED = [
    "sport", "set name", "description",
    "purchase price", "sale price", "comc credit",
    "date sold", "acquisition date",
]

SYN_MAP = {
    "sport": "sport",
    "set name": "set name",
    "description": "description",
    "purchase price": "purchase price",
    "sale price": "sale price",
    "comc credit": "comc credit",
    "date sold": "date sold",
    "acquisition date": "acquisition date",
    # variants
    "set": "set name",
    "player": "description", "player name": "description",
    "buy price": "purchase price", "cost": "purchase price",
    "sold price": "sale price", "price sold": "sale price",
    "credit after sale": "comc credit", "net credit": "comc credit",
    "sold date": "date sold",
    "date acquired": "acquisition date", "acquired date": "acquisition date",
}

def normalise_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("_", " ")
        .str.lower()
    )
    for src, dst in SYN_MAP.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)
    return df

def validate_required(df: pd.DataFrame):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        st.error("❌ Missing required columns: " + ", ".join(missing))
        st.stop()

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # UK-style date parsing to avoid warnings and inconsistent parsing
    for col in ["date sold", "acquisition date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    for col in ["purchase price", "sale price", "comc credit"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def extract_year(set_name: str):
    if pd.isna(set_name):
        return None
    m = re.search(r"\b\d{{4}}\b", str(set_name))
    return int(m.group()) if m else None

def extract_name(description: str):
    if pd.isna(description):
        return ""
    s = re.sub(r"\s*\(.*?\)\s*", " ", str(description))
    if "-" in s:
        s = s.split("-")[-1].strip()
    return s.strip()

@st.cache_data(ttl=60 * 60 * 24)
def get_usd_to_gbp() -> float:
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=10)
        r.raise_for_status()
        return float(r.json()["rates"]["GBP"])
    except Exception:
        return 0.78

def fmt_currency(series: pd.Series, symbol: str) -> pd.Series:
    return series.apply(lambda x: f"{symbol}{x:,.2f}" if pd.notna(x) else "")

def parse_mixed_timestamps(series: pd.Series) -> pd.Series:
    """
    Robustly parse mixed timestamp strings:
    - 'm/d/Y H:M(:S) AM/PM'
    - 'm/d/Y HH:MM'
    - 'd/m/Y HH:MM'
    Returns tz-naive pandas datetimes (no timezone yet).
    """
    def _try_parse(x: str):
        if pd.isna(x):
            return pd.NaT
        s = str(x).strip()
        if not s:
            return pd.NaT
        # 1) Try US month/day first (handles AM/PM + 24h)
        try:
            return du.parse(s, dayfirst=False, fuzzy=False)
        except Exception:
            pass
        # 2) Try day-first
        try:
            return du.parse(s, dayfirst=True,  fuzzy=False)
        except Exception:
            return pd.NaT

    parsed = series.apply(_try_parse)
    return pd.to_datetime(parsed, errors="coerce")


# ---- KPI helpers ----
def fmt_int(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return "0"

def fmt_money(n, symbol):
    try:
        return f"{symbol}{n:,.2f}"
    except Exception:
        return f"{symbol}0.00"

def kpi_card_html(title, value, sub=None, accent="volume", delta=None, delta_is_pos=True):
    delta_html = ""
    if delta is not None:
        cls = "" if delta_is_pos else " neg"
        delta_html = f'<span class="kpi-delta{cls}">{delta}</span>'
    sub_html = f'<p class="kpi-sub">{sub}</p>' if sub else ""
    return f"""
    <div class="kpi-card" data-accent="{accent}">
      <p class="kpi-title">{title}</p>
      <p class="kpi-value">{value}{delta_html}</p>
      {sub_html}
    </div>
    """


# ---------------------------- App ----------------------------
def main():
    st.markdown(
        "<h1 style='text-align: center;'>The COMC Seller Dashboard Tool</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 0.85rem;'>"
        "Use the sidebar to set filters and currency."
        "</p>",
        unsafe_allow_html=True
    )

    files = st.file_uploader(
        "Upload one or more COMC Sales History CSV files",
        type=["csv"], accept_multiple_files=True
    )

    if not files:
        with st.sidebar:
            st.header("Filters")
            st.markdown('<div class="comc-callout">Upload CSVs to enable filters.</div>', unsafe_allow_html=True)
        st.markdown('<div class="comc-callout">Upload CSVs to begin.</div>', unsafe_allow_html=True)
        return

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = normalise_headers(df)
    validate_required(df)
    df = coerce_types(df)

    # -------- Sidebar Filters --------
    with st.sidebar:
        st.header("Filters")

        # Currency
        with st.expander("Currency", expanded=True):
            currency = st.radio("Display currency", ["USD ($)", "GBP (£)"], index=0, horizontal=True)
            usd_to_gbp = get_usd_to_gbp()
            if currency.startswith("GBP"):
                st.caption(f"FX (cached daily): 1 USD = {usd_to_gbp:.4f} GBP")
        curr_symbol = "£" if currency.startswith("GBP") else "$"
        convert_to_gbp = currency.startswith("GBP")

        # Sold date range
        with st.expander("Sold Date", expanded=True):
            sold_option = st.selectbox(
                "Range",
                ["Custom", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Last 180 Days"],
                key="sold_date",
            )

        # Acquisition date range
        with st.expander("Acquisition Date", expanded=False):
            acq_option = st.selectbox(
                "Range",
                ["Custom", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Last 180 Days"],
                key="acq_date",
            )

    # -------- Apply filters --------
    filtered = df.copy()

    # Sold date
    sold_min = pd.to_datetime(filtered["date sold"], errors="coerce").min()
    sold_max = pd.to_datetime(filtered["date sold"], errors="coerce").max()
    if pd.isna(sold_min) or pd.isna(sold_max):
        st.error("No valid 'Date Sold' values found.")
        st.stop()

    if sold_option == "Custom":
        with st.sidebar.expander("Sold Date", expanded=True):
            start_sold = st.date_input(
                "Start (Sold)",
                value=sold_min.date(),
                min_value=sold_min.date(),
                max_value=sold_max.date(),
                format="DD/MM/YYYY",
            )
            end_sold = st.date_input(
                "End (Sold)",
                value=sold_max.date(),
                min_value=sold_min.date(),
                max_value=sold_max.date(),
                format="DD/MM/YYYY",
            )
    else:
        end_sold = datetime.now().date()
        days = {"Last 7 Days": 6, "Last 30 Days": 29, "Last 90 Days": 89, "Last 180 Days": 179}[sold_option]
        start_sold = end_sold - timedelta(days=days)

    filtered = filtered[
        (filtered["date sold"].dt.date >= start_sold)
        & (filtered["date sold"].dt.date <= end_sold)
    ]

    # Acquisition date
    acq_min = pd.to_datetime(filtered["acquisition date"], errors="coerce").min()
    acq_max = pd.to_datetime(filtered["acquisition date"], errors="coerce").max()
    if not (pd.isna(acq_min) or pd.isna(acq_max)):
        if acq_option == "Custom":
            with st.sidebar.expander("Acquisition Date", expanded=True):
                start_acq = st.date_input(
                    "Start (Acquisition)",
                    value=acq_min.date(),
                    min_value=acq_min.date(),
                    max_value=acq_max.date(),
                    format="DD/MM/YYYY",
                )
                end_acq = st.date_input(
                    "End (Acquisition)",
                    value=acq_max.date(),
                    min_value=acq_min.date(),
                    max_value=acq_max.date(),
                    format="DD/MM/YYYY",
                )
        else:
            end_acq = datetime.now().date()
            days = {"Last 7 Days": 6, "Last 30 Days": 29, "Last 90 Days": 89, "Last 180 Days": 179}[acq_option]
            start_acq = end_acq - timedelta(days=days)

        filtered = filtered[
            (filtered["acquisition date"].dt.date >= start_acq)
            & (filtered["acquisition date"].dt.date <= end_acq)
        ]
    else:
        st.sidebar.caption("No valid Acquisition dates after Sold filter.")

    # --- Sale Type filter ---
    with st.sidebar.expander("Sale Type", expanded=False):
        c1, c2 = st.columns(2)
        with c1: show_sent_in = st.checkbox("Sent In", value=True)
        with c2: show_flipped = st.checkbox("Resold", value=True)

    if show_sent_in and not show_flipped:
        filtered = filtered[filtered["purchase price"].isna()]
    elif show_flipped and not show_sent_in:
        filtered = filtered[filtered["purchase price"].notna()]
    elif not show_sent_in and not show_flipped:
        st.warning("Select at least one of Sent In or Flipped to continue.")
        st.stop()

    # --- Sports multiselect filter ---
    with st.sidebar.expander("Sports", expanded=True):
        all_sports = sorted([s for s in filtered["sport"].dropna().unique()])
        selected_sports = st.multiselect("Select sports", options=all_sports, default=all_sports)
        if selected_sports:
            filtered = filtered[filtered["sport"].isin(selected_sports)]
        else:
            st.warning("No sports selected; nothing to show.")
            st.stop()

    # --- Heatmap options (sidebar) ---
    with st.sidebar.expander("Heatmap options", expanded=True):
        tz_labels = {"America/Los_Angeles": "Los Angeles", "Europe/London": "London"}
        tz_choice = st.radio(
            "Timezone:",
            list(tz_labels.keys()),
            index=0,
            horizontal=True,
            format_func=lambda k: tz_labels[k],
            key="heatmap_tz",
        )

    # -------- KPI row --------
    added_mask = filtered["purchase price"].isna()
    added_sold = int(added_mask.sum())
    flipped_sold = int((~added_mask).sum())

    flipped_only = filtered.loc[filtered["purchase price"].notna()].copy()
    flipped_median_markup = None
    if not flipped_only.empty:
        valid = (flipped_only["purchase price"] > 0) & flipped_only["sale price"].notna()
        mm_series = ((flipped_only.loc[valid, "sale price"] - flipped_only.loc[valid, "purchase price"])
                     / flipped_only.loc[valid, "purchase price"] * 100)
        if not mm_series.empty:
            flipped_median_markup = float(mm_series.median())

    per_day = filtered["date sold"].dt.date.value_counts()
    median_sales_per_day = int(per_day.median()) if not per_day.empty else None

    nd_kpi = filtered.copy()
    nd_kpi["purchase price"] = nd_kpi["purchase price"].where(
        nd_kpi["purchase price"].notna(),
        nd_kpi["sale price"].apply(lambda x: 2.0 if pd.notna(x) and x > 99 else 0.5),
    )
    nd_kpi["profit"] = nd_kpi["comc credit"] - nd_kpi["purchase price"]

    if convert_to_gbp:
        fx = usd_to_gbp
        nd_kpi[["purchase price","sale price","comc credit","profit"]] *= fx

    total_sold = len(nd_kpi)
    total_sales_amt = nd_kpi["sale price"].sum(skipna=True)
    profit_sum = nd_kpi["profit"].sum(skipna=True)

    nd_kpi["days_to_sale"] = (nd_kpi["date sold"] - nd_kpi["acquisition date"]).dt.days
    median_days = float(nd_kpi["days_to_sale"].median()) if nd_kpi["days_to_sale"].notna().any() else None

    today = pd.Timestamp.today().normalize()
    last30 = today - pd.Timedelta(days=30)
    prev30 = today - pd.Timedelta(days=60)
    sold_last30 = nd_kpi[nd_kpi["date sold"] >= last30].shape[0]
    sold_prev30 = nd_kpi[(nd_kpi["date sold"] < last30) & (nd_kpi["date sold"] >= prev30)].shape[0]
    if sold_prev30 > 0:
        pct = (sold_last30 - sold_prev30) / sold_prev30 * 100
        delta_count, delta_is_pos = f"{pct:+.0f}%", pct >= 0
    else:
        delta_count, delta_is_pos = None, True

    kpis_html = f"""
    <div class="kpi-grid">
      {kpi_card_html("Cards Sold", fmt_int(total_sold), accent="volume")}
      {kpi_card_html("Total Sales", fmt_money(total_sales_amt, curr_symbol), sub="Sum of sale price", accent="volume")}
      {kpi_card_html("Profit", fmt_money(profit_sum, curr_symbol), sub="After assumed costs", accent="profit")}
      {kpi_card_html("Median Days to Sale",
                     fmt_int(median_days) if median_days is not None else "–",
                     sub=f"Median sales per day: {fmt_int(median_sales_per_day) if median_sales_per_day is not None else '–'}",
                     accent="time")}
    </div>
    """
    st.markdown(kpis_html, unsafe_allow_html=True)

    kpis2_html = f"""
    <div class="kpi-grid">
      {kpi_card_html("Sent In Sales", fmt_int(added_sold), accent="volume")}
      {kpi_card_html("Resold Sales", fmt_int(flipped_sold), accent="volume")}
    </div>
    """
    st.markdown(kpis2_html, unsafe_allow_html=True)

    # -------- Derived cols --------
    filtered = filtered.copy()
    filtered["year"] = filtered["set name"].apply(extract_year)

    # Matplotlib text colors for readability
    plt.rcParams.update({
        "axes.edgecolor": COMC_BORDER,
        "axes.labelcolor": COMC_NAVY,
        "xtick.color": COMC_TEXT,
        "ytick.color": COMC_TEXT,
        "text.color":  COMC_TEXT,
        "figure.facecolor": COMC_BG,
        "axes.facecolor": COMC_BG,
    })

    st.markdown("<h4 style='text-align: center;'>Sales by Sport</h4>", unsafe_allow_html=True)

    counts = filtered["sport"].value_counts()
    if not counts.empty:
        fig, ax = plt.subplots(figsize=(16, 6), facecolor=COMC_BG)
        ax.barh(counts.index, counts.values, color=COMC_RED)
        ax.set_xlabel("Total Sales", fontsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.tick_params(axis="x", labelsize=16)
        ax.invert_yaxis()
        st.pyplot(fig, use_container_width=True)
    else:
        st.write("No data after filters.")

    # ---------- Top Selling Players (card layout, 2 per row) ----------
    filtered["player_name"] = filtered["description"].apply(extract_name)
    top = (
        filtered["player_name"]
        .value_counts()
        .head(10)
        .rename_axis("Name")
        .reset_index(name="Items Sold")
    )
    total_items = len(filtered["player_name"])
    top["% of total"] = (top["Items Sold"] / max(total_items, 1) * 100).round(1)

    def player_card_html(rank, name, sold, pct, accent=COMC_RED, border=COMC_BORDER, bg="#FFFFFF"):
        pct = max(min(float(pct), 100), 0)
        return f"""
        <div style="
            border:1px solid {border};
            border-radius:14px;
            padding:14px 16px;
            background:{bg};
            height:120px;
            display:flex;
            flex-direction:column;
            justify-content:space-between;
            gap:6px;
        ">
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="
                font-size:12px; font-weight:600; color:{accent};
                border:1px solid {accent}; border-radius:999px;
                padding:2px 8px; line-height:1;">#{rank}</span>
            <span style="font-weight:700; font-size:16px;">{name}</span>
        </div>

        <div style="display:flex; align-items:baseline; gap:8px;">
            <span style="font-size:34px; font-weight:600; line-height:1;">{int(sold)}</span>
            <span style="font-size:12px; color:#666;">({pct:.1f}% of total)</span>
        </div>

        <div style="height:6px; background:#F1F2F4; border-radius:999px; overflow:hidden;">
            <div style="width:{pct}%; height:100%; background:{accent};"></div>
        </div>
        </div>
        """

    st.markdown("""
    <style>
    /* reduce vertical whitespace in columns */
    .block-container div[data-testid="column"] > div { margin-bottom: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: center;'>Top Selling Players</h4>", unsafe_allow_html=True)

    for i in range(0, len(top), 2):
        c1, c2 = st.columns(2)
        for j, col in enumerate((c1, c2)):
            if i + j < len(top):
                r = top.iloc[i + j]
                html = player_card_html(
                    rank=i + j + 1,
                    name=r["Name"],
                    sold=r["Items Sold"],
                    pct=r["% of total"],
                    accent=COMC_RED,
                    border=COMC_BORDER,
                )
                with col:
                    st.markdown(html, unsafe_allow_html=True)

    # ---------- Currency-aware dataset for charts ----------
    nd = filtered.copy()
    nd["purchase price"] = nd["purchase price"].where(
        nd["purchase price"].notna(),
        nd["sale price"].apply(lambda x: 2.0 if pd.notna(x) and x > 99 else 0.5),
    )
    nd["profit"] = nd["comc credit"] - nd["purchase price"]

    money_cols = ["purchase price", "sale price", "comc credit", "profit"]
    if convert_to_gbp:
        nd[money_cols] = nd[money_cols] * usd_to_gbp

    nd["days_to_sale"] = (nd["date sold"] - nd["acquisition date"]).dt.days
    nd["markup_pct"] = pd.NA
    mask = (nd["purchase price"] > 0)
    nd.loc[mask, "markup_pct"] = (nd.loc[mask, "sale price"] - nd.loc[mask, "purchase price"]) / nd.loc[mask, "purchase price"] * 100

    # -------- Trend Charts --------
    st.markdown("<h4 style='text-align: center;'>Weekly Sales</h4>", unsafe_allow_html=True)

    weekly_sales = nd.groupby(pd.Grouper(key="date sold", freq="W"))["sale price"].size()
    if not weekly_sales.empty:
        fig, ax = plt.subplots(figsize=(16, 6), facecolor=COMC_BG)
        fig.patch.set_alpha(1.0)
        ax.bar(weekly_sales.index.to_pydatetime(), weekly_sales.values, color=COMC_RED)
        ax.set_ylabel("Sales Total", fontsize=12)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        ax.tick_params(axis="x", which="major", labelsize=9)
        ax.tick_params(axis="x", which="minor", length=0)
        ax.xaxis.get_offset_text().set_visible(False)
        ax.margins(x=0.01)
        st.pyplot(fig)

    st.markdown("<h4 style='text-align: center;'>Sales by Day of Week</h4>", unsafe_allow_html=True)

    if not nd.empty:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        nd["dow"] = nd["date sold"].dt.day_name()
        nd["dow"] = pd.Categorical(nd["dow"], categories=dow_order, ordered=True)
        by_dow = nd["dow"].value_counts().sort_index()
        fig, ax = plt.subplots(facecolor=COMC_BG)
        fig.patch.set_alpha(1.0)
        ax.bar(by_dow.index.astype(str), by_dow.values, color=COMC_RED)
        ax.set_ylabel("Total Sales")
        ax.tick_params(axis="x", labelsize=6)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        st.pyplot(fig)

    st.markdown("<h4 style='text-align: center;'>Days to Sale</h4>", unsafe_allow_html=True)

    if nd["days_to_sale"].notna().any():
        dts = nd.loc[nd["days_to_sale"] >= 0, "days_to_sale"].dropna()
        fig, ax = plt.subplots(facecolor=COMC_BG)
        fig.patch.set_alpha(1.0)
        ax.set_xlabel("Day number")
        ax.set_ylabel("Total Sales")
        counts, edges, patches = ax.hist(
            dts, bins=60, density=False, edgecolor=COMC_BORDER, color=COMC_RED
        )
        heights = np.array([p.get_height() for p in patches], dtype=float)
        ymax = float(heights.max()) if heights.size else 1.0
        ax.set_ylim(0, ymax * 1.1)
        ax.set_xlim(left=0)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        st.pyplot(fig)

    # --- Sales Heatmap ---
    st.markdown(
        f"<h4 id='hm-title' style='text-align:center;color:{COMC_RED};'>Sales Heatmap</h4>",
        unsafe_allow_html=True
    )

    if not nd.empty and nd["date sold"].notna().any():
        # Read the sidebar selection; default to LA if not present
        tz_choice = st.session_state.get("heatmap_tz", "America/Los_Angeles")

        # Parse mixed timestamp formats
        s = parse_mixed_timestamps(nd["date sold"])

        # Localize to Seattle (source tz), then convert if UK selected
        s = s.dt.tz_localize("America/Los_Angeles", ambiguous="NaT", nonexistent="NaT")
        if tz_choice == "Europe/London":
            s = s.dt.tz_convert("Europe/London")

        # Prepare day-of-week + hour from converted times
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        nd_h = nd.copy()
        nd_h["dow"]  = s.dt.day_name()
        nd_h["hour"] = s.dt.hour
        nd_h["dow"]  = pd.Categorical(nd_h["dow"], categories=dow_order, ordered=True)

        # Pivot table → 7x24 grid
        hours = list(range(24))
        pivot = (
            nd_h.pivot_table(index="dow", columns="hour", values="sale price",
                             aggfunc="size", fill_value=0)
            .reindex(index=dow_order, columns=hours, fill_value=0)
        )
        Z = pivot.values

        # ---- Plot heatmap ----
        fig, ax = plt.subplots(figsize=(16, 6), facecolor=COMC_BG)
        fig.patch.set_alpha(1.0)

        cmap = mcolors.LinearSegmentedColormap.from_list("comc_red", ["#FFFFFF", COMC_RED])
        im = ax.imshow(Z, aspect="auto", origin="upper", cmap=cmap)

        ax.set_yticks(range(len(dow_order)))
        ax.set_yticklabels(dow_order, fontsize=9)
        xticks = list(range(0, 24, 2))
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{h:02d}:00" for h in xticks], fontsize=9)

        # Grid
        ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        ax.grid(which="minor", color=COMC_BORDER, linewidth=0.6)
        ax.tick_params(which="minor", length=0)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Sales count", rotation=90)

        st.pyplot(fig, use_container_width=True)
    else:
        st.write("No valid timestamps in 'Date Sold' to build a day/hour heatmap.")


if __name__ == "__main__":
    main()
