import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Farmer Choice Assistant (Demo)", page_icon="🌾", layout="centered")

# ========== External price fetchers (cached) ==========
@st.cache_data(ttl=24*3600)
def fetch_barley_price_usd_t() -> float | None:
    try:
        fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PBARLUSDM"
        df = pd.read_csv(fred_url)
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["DATE"]).sort_values("DATE")
        val = df.iloc[-1]["PBARLUSDM"]
        return float(val) if pd.notnull(val) else None
    except Exception:
        return None

@st.cache_data(ttl=24*3600)
def fetch_urea_price_usd_t() -> float | None:
    try:
        wb_url = ("https://thedocs.worldbank.org/en/doc/"
                  "5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx")
        raw = pd.read_excel(wb_url, sheet_name="Monthly Prices", header=4, engine="openpyxl")
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

# ========== Data load ==========
DATA_PATH = Path("data/barley_scenarios.csv")
df = pd.read_csv(DATA_PATH, sep=",", engine="c", encoding="utf-8", skip_blank_lines=True)
df.columns = [c.strip().lower() for c in df.columns]

# Check required columns
required = ["region","crop","variety","type","maturity","program_n","program_cp",
            "yield_t_ha","cost_eur_ha","protein_pct","sustain_score","p_program","late_n"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"CSV missing required columns: {missing}")
    st.stop()

# ========== UI ==========
st.title("🌾 Malting Barley (Demo)")
st.markdown(
    "This demo shows how AI-style decision support can simplify complex choices. "
    "It **does not replace** agronomists or farmer expertise — it provides quick, neutral baseline options."
)

region = st.selectbox("Region", sorted(df["region"].unique().tolist()))

priority = st.radio(
    "Your priority",
    [
        "Balanced",
        "Maximize profit",
        "Maximize yield",
        "Higher sustainability",
        "Lower cost",
        "Maximize extract (starch) within N 1.6–1.75%"
    ],
    index=0
)

risk_tolerance = st.slider("Risk tolerance (0=low, 1=high)", 0.0, 1.0, 0.3, 0.1)

# Market defaults via proxies
barley_proxy = fetch_barley_price_usd_t()
urea_proxy   = fetch_urea_price_usd_t()
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
    st.caption("Barley: FRED global barley price (PBARLUSDM), monthly proxy. Adjust malting premium for your spec/region.")
    st.caption("Urea: World Bank CMO 'Urea (bulk, Middle East)', monthly proxy (context only).")

st.divider()

# ========== Filter ==========
d = df[df["region"] == region].copy()
if d.empty:
    st.warning("No data for this region.")
    st.stop()

# Clean categorical strings
for c in ["region","crop","variety","type","maturity","program_n","program_cp","p_program"]:
    d[c] = d[c].astype(str).str.strip()
d["late_n"] = d["late_n"].astype(int)

# Helpers
def minmax(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)

# Normalize basic metrics
d["yield_norm"] = minmax(d["yield_t_ha"])
d["cost_norm"]  = 1 - minmax(d["cost_eur_ha"])  # lower cost = higher score

# ========== Malting spec + economics ==========
low_ok, high_ok = 10.0, 11.5  # protein %
def quality_band(p):
    if low_ok <= p <= high_ok: return "malting"
    if 9.5 <= p < low_ok or high_ok < p <= 12.0: return "edge"
    return "oos"

d["quality"] = d["protein_pct"].apply(quality_band)
d["grain_n_pct"] = d["protein_pct"] / 6.25

def price_per_t(row):
    if row["quality"] == "malting": return base_price + malting_premium
    if row["quality"] == "oos":     return max(0, base_price - oos_discount)
    return base_price
d["price_eur_t"] = d.apply(price_per_t, axis=1)

d["revenue_eur_ha"] = d["yield_t_ha"] * d["price_eur_t"]
d["profit_eur_ha"]  = d["revenue_eur_ha"] - d["cost_eur_ha"]
d["profit_norm"]    = minmax(d["profit_eur_ha"])

def quality_score(p):
    if low_ok <= p <= high_ok: return 1.0
    if 9.5 <= p < low_ok or high_ok < p <= 12.0: return 0.6
    return 0.15
d["quality_score"] = d["protein_pct"].apply(quality_score)

hard_penalty = np.where((d["protein_pct"] < 9.5) | (d["protein_pct"] > 12.0), 0.25 * (1 - risk_tolerance), 0.0)

# ========== Extract (starch) proxy ==========
# Favor yield while keeping protein near ~10.5% (grain N ~1.68%)
target_protein = 10.5
dev = (d["protein_pct"] - target_protein).abs()
dev_scaled = (dev / 1.5).clip(0, 1)         # within ±1.5% is best
quality_alignment = 1 - dev_scaled           # 1 at 10.5%, down to 0 beyond ~12% or <9%
extract_raw = d["yield_norm"] * quality_alignment
d["extract_norm"] = minmax(extract_raw)

# Small agronomy nudges for extract priority
# Bonus if good P program; penalty if late N (tends to push protein up)
p_bonus_flags = d["p_program"].str.contains("starter p", case=False) | d["p_program"].str.contains("soil p adequate", case=False)
d["extract_bonus"]   = np.where(p_bonus_flags, 0.04, 0.0)
d["extract_penalty"] = np.where(d["late_n"] == 1, 0.08 * (1 - risk_tolerance), 0.0)

# ========== Scoring ==========
# Base weights incl. extract
w = {"profit": 0.32, "quality": 0.24, "yield": 0.20, "sustain": 0.14, "extract": 0.10}

if priority == "Maximize profit":
    w["profit"] += 0.18; w["yield"] -= 0.06; w["sustain"] -= 0.06; w["extract"] -= 0.06
elif priority == "Maximize yield":
    w["yield"]  += 0.18; w["profit"] -= 0.06; w["sustain"] -= 0.06; w["extract"] -= 0.06
elif priority == "Higher sustainability":
    w["sustain"] += 0.18; w["profit"] -= 0.09; w["yield"]  -= 0.05; w["extract"] -= 0.04
elif priority == "Lower cost":
    # cost flows through profit; we gently nudge sustain/quality
    w["sustain"] += 0.08; w["quality"] += 0.04; w["profit"] -= 0.06; w["yield"] -= 0.06
elif priority == "Maximize extract (starch) within N 1.6–1.75%":
    w["extract"] += 0.20; w["yield"] += 0.05; w["profit"] -= 0.10; w["sustain"] -= 0.05

# Normalize weights
wsum = sum(max(0, v) for v in w.values())
w = {k: max(0, v) / wsum for k, v in w.items()}

# Base score
d["score"] = (
    w["profit"]  * d["profit_norm"]   +
    w["quality"] * d["quality_score"] +
    w["yield"]   * d["yield_norm"]    +
    w["sustain"] * d["sustain_score"] +
    w["extract"] * d["extract_norm"]
    - hard_penalty
)

# Apply extract-specific nudges only when that priority is chosen
if priority == "Maximize extract (starch) within N 1.6–1.75%":
    d["score"] = d["score"] + d["extract_bonus"] - d["extract_penalty"]

# ========== Rank & Output ==========
top = d.sort_values("score", ascending=False).head(3).reset_index(drop=True)

st.subheader(f"Top recommendations for **{region}** ({priority})")

for i, row in top.iterrows():
    st.markdown(f"### Option {i+1}: {row['variety']} — {row['type']} ({row['maturity']})")

    meets_spec = "✅ Meets malting spec" if (low_ok <= row["protein_pct"] <= high_ok) else "⚠️ Spec risk"
    st.markdown(f"**{meets_spec}**  ·  Grain N: **{row['grain_n_pct']:.2f}%**  ·  P program: **{row['p_program']}**  ·  Late N: **{'Yes' if row['late_n']==1 else 'No'}**")

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
