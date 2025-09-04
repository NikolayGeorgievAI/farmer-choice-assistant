import streamlit as st
import pandas as pd
import numpy as np
import re
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

# Market defaults via proxies (graceful fallbacks)
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
    st.caption("Barley: FRED global barley price (PBARLUSDM), monthly proxy. Adjust malting premium for region/spec.")
    st.caption("Urea: World Bank CMO 'Urea (bulk, Middle East)', monthly proxy (context only).")

st.divider()

# ========== Filter & clean ==========
d = df[df["region"] == region].copy()
if d.empty:
    st.warning("No data for this region.")
    st.stop()

for c in ["region","crop","variety","type","maturity","program_n","program_cp","p_program"]:
    d[c] = d[c].astype(str).str.strip()
d["late_n"] = d["late_n"].astype(int)

# Helpers
def minmax(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)

# Estimate kg N/ha from program_n text (e.g., "Low (60 kg N, 1x)")
def extract_n_rate(s: str) -> float:
    if not isinstance(s, str): return 0.0
    m = re.search(r"(\d+)\s*kg\s*N", s)
    return float(m.group(1)) if m else 0.0

d["n_rate_kg_ha"] = d["program_n"].map(extract_n_rate)

# Tie cost to urea price (urea ~46% N). Baseline at 400 USD/t.
baseline_urea = 400.0
unit_cost_now = (urea_price / 1000.0) / 0.46          # USD per kg N
unit_cost_base = (baseline_urea / 1000.0) / 0.46
delta_per_kgN = unit_cost_now - unit_cost_base        # USD per kg N difference vs baseline
d["cost_adj_eur_ha"] = d["cost_eur_ha"] + d["n_rate_kg_ha"] * delta_per_kgN  # assume €≈$

# Normalize core metrics (with adjusted cost)
d["yield_norm"] = minmax(d["yield_t_ha"])
d["cost_norm"]  = 1 - minmax(d["cost_adj_eur_ha"])

# Malting spec band
low_ok, high_ok = 10.0, 11.5
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

# Economics with adjusted cost
d["revenue_eur_ha"] = d["yield_t_ha"] * d["price_eur_t"]
d["profit_eur_ha"]  = d["revenue_eur_ha"] - d["cost_adj_eur_ha"]
d["profit_norm"]    = minmax(d["profit_eur_ha"])

# Quality score (baked in for "Balanced" ranking)
def quality_score(p):
    if low_ok <= p <= high_ok: return 1.0
    if 9.5 <= p < low_ok or high_ok < p <= 12.0: return 0.6
    return 0.15
d["quality_score"] = d["protein_pct"].apply(quality_score)

# Hard penalty far out of spec
hard_penalty = np.where((d["protein_pct"] < 9.5) | (d["protein_pct"] > 12.0), 0.25 * (1 - risk_tolerance), 0.0)

# Extract (starch) proxy
target_protein = 10.5
dev = (d["protein_pct"] - target_protein).abs()
dev_scaled = (dev / 1.5).clip(0, 1)     # within ±1.5% best
quality_alignment = 1 - dev_scaled
extract_raw = d["yield_norm"] * quality_alignment
d["extract_norm"] = minmax(extract_raw)

# Small agronomy nudges for extract priority
p_bonus_flags = d["p_program"].str.contains("starter p", case=False) | d["p_program"].str.contains("soil p adequate", case=False)
d["extract_bonus"]   = np.where(p_bonus_flags, 0.05, 0.0)
d["extract_penalty"] = np.where(d["late_n"] == 1, 0.10 * (1 - risk_tolerance), 0.0)

# Composite score (used only for Balanced + tie-breaks)
w_balanced = {"profit": 0.40, "quality": 0.25, "yield": 0.20, "sustain": 0.10, "extract": 0.05}
wsum = sum(w_balanced.values())
w_balanced = {k: v/wsum for k, v in w_balanced.items()}
d["score"] = (
    w_balanced["profit"]  * d["profit_norm"]   +
    w_balanced["quality"] * d["quality_score"] +
    w_balanced["yield"]   * d["yield_norm"]    +
    w_balanced["sustain"] * d["sustain_score"] +
    w_balanced["extract"] * d["extract_norm"]
    - hard_penalty
)

# -------- Priority-driven ranking (CLEAR reordering) --------
if priority == "Maximize profit":
    d["_pri_key"] = d["profit_eur_ha"];     asc = [False]
elif priority == "Maximize yield":
    d["_pri_key"] = d["yield_t_ha"];        asc = [False]
elif priority == "Lower cost":
    d["_pri_key"] = d["cost_adj_eur_ha"];   asc = [True]
elif priority == "Higher sustainability":
    d["_pri_key"] = d["sustain_score"];     asc = [False]
elif priority.startswith("Maximize extract"):
    d["_pri_key"] = d["extract_norm"] + d["extract_bonus"] - d["extract_penalty"]; asc = [False]
else:  # Balanced
    d["_pri_key"] = d["score"];             asc = [False]

# Tie-breakers: then by composite score, then by profit
sort_cols = ["_pri_key", "score", "profit_eur_ha"]
asc += [False, False]
top = d.sort_values(sort_cols, ascending=asc).head(3).reset_index(drop=True)

# ========== Output ==========
st.subheader(f"Top recommendations for **{region}** ({priority})")

def meets_spec_label(pct: float) -> str:
    return "✅ Meets malting spec" if (low_ok <= pct <= high_ok) else "⚠️ Spec risk"

for i, row in top.iterrows():
    st.markdown(f"### Option {i+1}: {row['variety']} — {row['type']} ({row['maturity']})")
    st.markdown(f"**{meets_spec_label(row['protein_pct'])}**  ·  Grain N: **{row['grain_n_pct']:.2f}%**  ·  P program: **{row['p_program']}**  ·  Late N: **{'Yes' if row['late_n']==1 else 'No'}**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- **N program:** {row['program_n']}")
        st.markdown(f"- **CP program:** {row['program_cp']}")
        st.markdown(f"- **Expected yield:** {row['yield_t_ha']:.1f} t/ha")
    with col2:
        st.markdown(f"- **Price assumption:** €{row['price_eur_t']:.0f}/t")
        st.markdown(f"- **Revenue:** €{row['revenue_eur_ha']:.0f}/ha")
        st.markdown(f"- **Cost (adj. for urea):** €{row['cost_adj_eur_ha']:.0f}/ha")

    st.markdown(f"**Estimated profit:** €{row['profit_eur_ha']:.0f}/ha")
    st.progress(float(row["score"]))

# --- Narrative for Option 1 ---
if not top.empty:
    best = top.iloc[0]
    why = ""
    if priority == "Maximize profit":
        why = (f"**Why Option 1?** Highest margin at ~€{best['profit_eur_ha']:.0f}/ha, driven by "
               f"{best['yield_t_ha']:.1f} t/ha yield and malting acceptance (protein {best['protein_pct']:.1f}%). "
               f"Even with urea ~{urea_price} USD/t, costs stay competitive. Option 2 trades a little margin for a different risk profile.")
    elif priority == "Maximize yield":
        why = (f"**Why Option 1?** Top yield at {best['yield_t_ha']:.1f} t/ha while staying within/near malting spec "
               f"(protein {best['protein_pct']:.1f}%). If you want a safer protein buffer, Option 2 is a good trade-off.")
    elif priority == "Higher sustainability":
        why = (f"**Why Option 1?** Best sustainability score with a leaner N program and standard CP. "
               f"Protein {best['protein_pct']:.1f}% remains in the malting band; profit ≈€{best['profit_eur_ha']:.0f}/ha.")
    elif priority == "Lower cost":
        why = (f"**Why Option 1?** Lowest adjusted cost/ha (urea-linked) while keeping malting acceptance "
               f"(protein {best['protein_pct']:.1f}%). Profit stays competitive; inputs and risk are contained.")
    elif priority.startswith("Maximize extract"):
        why = (f"**Why Option 1?** Grain N ~{best['grain_n_pct']:.2f}% (protein {best['protein_pct']:.1f}%) near "
               f"the 1.6–1.75% target for extract, with strong yield ({best['yield_t_ha']:.1f} t/ha). "
               f"{'No late N' if best['late_n']==0 else 'Late N present'}; P program **{best['p_program']}** supports early vigor.")
    else:
        why = (f"**Why Option 1?** Best balance of margin (≈€{best['profit_eur_ha']:.0f}/ha), malting acceptance "
               f"(protein {best['protein_pct']:.1f}%), and operational risk.")
    st.info(why)

# Footers
st.caption(
    "Sustainability score (0–1) is a simple proxy from the N and crop protection intensity of each program "
    "(higher = lower expected footprint). Illustrative only."
)
st.caption(
    "Prices are monthly proxies (FRED barley; World Bank urea) used for defaults. Local/malting spot may differ; "
    "adjust premiums/discounts as needed. Adjusted cost ties fertilizer cost to urea via estimated kg N from the N program."
)
st.write("---")
st.write("Built by Nikolay Georgiev — demo to showcase how AI-style logic can simplify farmer choices.")
