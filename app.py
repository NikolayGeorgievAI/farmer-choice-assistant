import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Farmer Choice Assistant (Demo)", page_icon="🌾", layout="centered")

# ========== External price fetchers (cached) ==========

@st.cache_data(ttl=24*3600)
def fetch_barley_price_usd_t() -> float | None:
    """
    Global price of Barley (USD/mt) proxy from FRED (PBARLUSDM).
    Returns latest monthly value as float, or None on failure.
    """
    try:
        fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PBARLUSDM"
        df = pd.read_csv(fred_url)
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["DATE"])
        df = df.sort_values("DATE")
        val = df.iloc[-1]["PBARLUSDM"]
        return float(val) if pd.notnull(val) else None
    except Exception:
        return None

@st.cache_data(ttl=24*3600)
def fetch_urea_price_usd_t() -> float | None:
    """
    Urea (bulk, Middle East) monthly price from World Bank CMO workbook.
    Requires an Excel reader (openpyxl). Falls back to None if not available.
    """
    try:
        wb_url = ("https://thedocs.worldbank.org/en/doc/"
                  "5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx")
        # Many Streamlit environments support openpyxl; if not, this will raise and we'll return None.
        raw = pd.read_excel(wb_url, sheet_name="Monthly Prices", header=4, engine="openpyxl")
        # Find the urea column
        urea_cols = [c for c in raw.columns if isinstance(c, str) and "Urea" in c]
        if not urea_cols:
            return None
        ucol = urea_cols[0]
        df = raw[["Date", ucol]].dropna()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        val = df.iloc[-1][ucol]
        return float(val) if pd.notnull(val) else None
    except Exception:
        return None

# ========== Data load (your local CSV) ==========
DATA_PATH = Path("data/barley_scenarios.csv")
df = pd.read_csv(
    DATA_PATH,
    sep=",",
    engine="c",
    encoding="utf-8",
    skip_blank_lines=True
)
df.columns = [c.strip().lower() for c in df.columns]

# ========== UI ==========
st.title("🌾 Malting Barley (Demo)")
st.markdown(
    "This demo shows how AI-style decision support can simplify complex choices. "
    "It **does not replace** agronomists or farmer expertise — it provides quick, neutral baseline options."
)

# Region
region = st.selectbox("Region", sorted(df["region"].unique().tolist()))

# Priorities (malting spec is embedded—no separate option)
priority = st.radio(
    "Your priority",
    ["Balanced", "Maximize profit", "Maximize yield", "Lower cost", "Higher sustainability"],
    index=0
)

# Risk tolerance (kept for potential penalties)
risk_tolerance = st.slider("Risk tolerance (0=low, 1=high)", 0.0, 1.0, 0.3, 0.1)

# Try to fetch external prices
barley_proxy = fetch_barley_price_usd_t()
urea_proxy   = fetch_urea_price_usd_t()

# Defaults (rounded) with graceful fallbacks
base_price_default  = int(round(barley_proxy, 0)) if barley_proxy else 180
urea_price_default  = int(round(urea_proxy, 0))   if urea_proxy   else 400

st.subheader("Market assumptions")
colp1, colp2, colp3, colp4 = st.columns(4)
with colp1:
    base_price = st.number_input("Base barley price (€/t)", min_value=50, max_value=500, value=base_price_default, step=5)
with colp2:
    malting_premium = st.number_input("Malting premium (€/t)", min_value=0, max_value=200, value=25, step=5)
with colp3:
    oos_discount = st.number_input("Out-of-spec discount (€/t)", min_value=0, max_value=200, value=20, step=5)
with colp4:
    urea_price = st.number_input("Urea price (USD/t, proxy)", min_value=100, max_value=1200, value=urea_price_default, step=10)

with st.expander("Price sources (proxies)", expanded=False):
    st.caption(
        "Barley price: FRED global barley price (PBARLUSDM), monthly proxy. "
        "Malting-specific spot varies by region/spec; adjust the malting premium accordingly."
    )
    st.caption(
        "Urea price: World Bank CMO 'Urea (bulk, Middle East)', monthly proxy. "
        "Used here for context only."
    )

st.divider()

# ========== Filter by region ==========
d = df[df["region"] == region].copy()
if d.empty:
    st.warning("No data for this region.")
    st.stop()

# ========== Helpers ==========
def minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

# Normalize basic metrics
d["yield_norm"] = minmax(d["yield_t_ha"])
d["cost_norm"]  = 1 - minmax(d["cost_eur_ha"])  # lower cost = higher score

# ========== Malting spec + economics ==========
# Protein target band for malting acceptance
low_ok, high_ok = 10.0, 11.5  # %
def quality_band(p):
    if low_ok <= p <= high_ok:
        return "malting"      # premium
    if 9.5 <= p < low_ok or high_ok < p <= 12.0:
        return "edge"         # borderline -> base price
    return "oos"              # out of spec -> discount

d["quality"] = d["protein_pct"].apply(quality_band)

# Grain N (for display): protein (%) / 6.25
d["grain_n_pct"] = d["protein_pct"] / 6.25

# Price per tonne by quality band
def price_per_t(row):
    if row["quality"] == "malting":
        return base_price + malting_premium
    if row["quality"] == "oos":
        return max(0, base_price - oos_discount)
    return base_price  # edge

d["price_eur_t"] = d.apply(price_per_t, axis=1)

# Revenue & Profit per hectare
d["revenue_eur_ha"] = d["yield_t_ha"] * d["price_eur_t"]
d["profit_eur_ha"]  = d["revenue_eur_ha"] - d["cost_eur_ha"]

# Normalized for scoring
d["profit_norm"] = minmax(d["profit_eur_ha"])

# Quality score to always bake malting acceptance into ranking
def quality_score(p):
    if low_ok <= p <= high_ok: return 1.0
    if 9.5 <= p < low_ok or high_ok < p <= 12.0: return 0.6
    return 0.15
d["quality_score"] = d["protein_pct"].apply(quality_score)

# Extra hard penalty well outside spec (protect the farmer)
hard_penalty = np.where((d["protein_pct"] < 9.5) | (d["protein_pct"] > 12.0), 0.25 * (1 - risk_tolerance), 0.0)

# ========== Scoring ==========
# Base weights (profit + quality lead; yield & sustainability follow)
w = {"profit": 0.34, "quality": 0.26, "yield": 0.22, "sustain": 0.18}

if priority == "Maximize profit":
    w["profit"] += 0.18; w["yield"] -= 0.08; w["sustain"] -= 0.10
elif priority == "Maximize yield":
    w["yield"]  += 0.18; w["profit"] -= 0.08; w["sustain"] -= 0.10
elif priority == "Lower cost":
    # cost is embedded via profit; nudge sustainability for low-input preference
    w["sustain"] += 0.10; w["profit"] -= 0.05; w["yield"] -= 0.05
elif priority == "Higher sustainability":
    w["sustain"] += 0.18; w["profit"] -= 0.09; w["yield"]  -= 0.09

# Normalize weights
wsum = sum(max(0, v) for v in w.values())
w = {k: max(0, v) / wsum for k, v in w.items()}

# Final score
d["score"] = (
    w["profit"]  * d["profit_norm"]   +
    w["quality"] * d["quality_score"] +
    w["yield"]   * d["yield_norm"]    +
    w["sustain"] * d["sustain_score"]
    - hard_penalty
)

# ========== Rank & Output ==========
top = d.sort_values("score", ascending=False).head(3).reset_index(drop=True)

st.subheader(f"Top recommendations for **{region}** ({priority}, risk tolerance {risk_tolerance:.1f})")

for i, row in top.iterrows():
    st.markdown(f"### Option {i+1}: {row['variety']} — {row['type']} ({row['maturity']})")

    # Spec badge & Grain N
    meets_spec = "✅ Meets malting spec" if (low_ok <= row["protein_pct"] <= high_ok) else "⚠️ Spec risk"
    st.markdown(f"**{meets_spec}**  ·  Grain N: **{row['grain_n_pct']:.2f}%**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- **N program:** {row['program_n']}")
        st.markdown(f"- **CP program:** {row['program_cp']}")
        st.markdown(f"- **Expected yield:** {row['yield_t_ha']:.1f} t/ha")
    with col2:
        st.markdown(f"- **Price assumption:** €{row['price_eur_t']:.0f}/t")
        st.markdown(f"- **Revenue:** €{row['revenue_eur_ha']:.0f}/ha")
        st.markdown(f"- **Cost:** €{row['cost_eur_ha']:.0f}/ha")

    st.markdown(f"**Estimated profit:** €{row['profit_eur_ha']:.0f}/ha")
    st.progress(float(row["score"]))

st.caption(
    "Sustainability score (0–1) is a simple proxy from the N and crop protection intensity of each program "
    "(higher = lower expected footprint). This is illustrative — a real system would link to proper LCA/footprint models."
)
st.caption(
    "Prices are monthly proxies (FRED barley; World Bank urea) used for defaults only. "
    "Actual local/malting spot may differ; adjust premiums/discounts as needed."
)

st.write("---")
st.write("Built by Nikolay Georgiev — demo to showcase how AI-style logic can simplify farmer choices.")
