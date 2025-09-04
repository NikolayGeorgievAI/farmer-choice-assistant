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
colp1, colp2, colp3, colp4, colp5 = st.columns(5)
with colp1:
    base_price = st.number_input("Base barley price (€/t)", min_value=50, max_value=500, value=base_price_default, step=5)
with colp2:
    malting_premium = st.number_input("Malting premium (€/t)", min_value=0, max_value=200, value=25, step=5)
with colp3:
    oos_discount = st.number_input("Out-of-spec discount (€/t)", min_value=0, max_value=200, value=20, step=5)
with colp4:
    urea_price = st.number_input("Urea price (USD/t, proxy)", min_value=100, max_value=1200, value=urea_price_default, step=10)
with colp5:
    p_price = st.number_input("P product price (€/t)", min_value=200, max_value=1500, value=900, step=25)

st.subheader("Agronomy levers")
colt1, colt2 = st.columns(2)
with colt1:
    use_n_inhib = st.checkbox("Use N inhibitor (↓20% N, +€20/t urea, −0.1 pp protein)")
with colt2:
    use_p_prot  = st.checkbox("Use P protector (+0.15 t/ha with Starter/adequate P, +€20/t P, +€4/t premium)")

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

# Crude P rate per program for demo (kg/ha of P-product)
def estimate_p_rate(program: str) -> float:
    prog = (program or "").lower()
    if "starter" in prog:        return 25.0
    if "adequate" in prog:       return 10.0
    return 0.0  # no starter

d["n_rate_kg_ha"] = d["program_n"].map(extract_n_rate)
d["p_rate_kg_ha"] = d["p_program"].map(estimate_p_rate)

# --- Apply agronomy levers ---
n_rate_eff = d["n_rate_kg_ha"] * (0.8 if use_n_inhib else 1.0)
urea_price_eff = urea_price + (20 if use_n_inhib else 0)         # USD/t
p_price_eff    = p_price + (20 if use_p_prot else 0)             # €/t

# Cost adjustment for N (assume €≈$ for demo)
baseline_urea = 400.0
unit_cost_now  = (urea_price_eff / 1000.0) / 0.46      # €/kg N
unit_cost_base = (baseline_urea / 1000.0) / 0.46
delta_per_kgN  = unit_cost_now - unit_cost_base
cost_n_adj     = n_rate_eff * delta_per_kgN            # €/ha

# Cost for P product
cost_p = d["p_rate_kg_ha"] * (p_price_eff / 1000.0)    # €/ha

# Base cost + N delta + P cost
d["cost_adj_eur_ha"] = d["cost_eur_ha"] + cost_n_adj + cost_p

# --- Yield & extract effects from P protector ---
yield_bonus_p = np.where(d["p_rate_kg_ha"] > 0, (0.15 if use_p_prot else 0.0), 0.0)  # t/ha
d["yield_eff_t_ha"] = d["yield_t_ha"] + yield_bonus_p

# --- Protein adjustments ---
d["protein_eff"] = d["protein_pct"]
if use_n_inhib:
    d["protein_eff"] = d["protein_eff"] - 0.1  # −0.1 pp protein with inhibitor

# Normalize core metrics (with adjusted cost & yield)
d["yield_norm"] = minmax(d["yield_eff_t_ha"])
d["cost_norm"]  = 1 - minmax(d["cost_adj_eur_ha"])

# Malting spec band (on effective protein)
low_ok, high_ok = 10.0, 11.5
def quality_band(p):
    if low_ok <= p <= high_ok: return "malting"
    if 9.5 <= p < low_ok or high_ok < p <= 12.0: return "edge"
    return "oos"
d["quality"] = d["protein_eff"].apply(quality_band)
d["grain_n_pct"] = d["protein_eff"] / 6.25

# Price per tonne (malting premium boosted by P protector)
extra_premium = 4 if use_p_prot else 0
def price_per_t(row):
    if row["quality"] == "malting": return base_price + malting_premium + extra_premium
    if row["quality"] == "oos":     return max(0, base_price - oos_discount)
    return base_price
d["price_eur_t"] = d.apply(price_per_t, axis=1)

# Economics with adjusted cost & yield
d["revenue_eur_ha"] = d["yield_eff_t_ha"] * d["price_eur_t"]
d["profit_eur_ha"]  = d["revenue_eur_ha"] - d["cost_adj_eur_ha"]
d["profit_norm"]    = minmax(d["profit_eur_ha"])

# Quality score (for Balanced)
def quality_score(p):
    if low_ok <= p <= high_ok: return 1.0
    if 9.5 <= p < low_ok or high_ok < p <= 12.0: return 0.6
    return 0.15
d["quality_score"] = d["protein_eff"].apply(quality_score)

# Hard penalty far out of spec (reduced if inhibitor on)
hard_penalty_base = ((d["protein_eff"] < 9.5) | (d["protein_eff"] > 12.0)).astype(float) * 0.25
hard_penalty = hard_penalty_base * (1 - risk_tolerance) * (0.7 if use_n_inhib else 1.0)

# Extract (starch) proxy (use effective yield & protein)
target_protein = 10.5
dev = (d["protein_eff"] - target_protein).abs()
dev_scaled = (dev / 1.5).clip(0, 1)
quality_alignment = 1 - dev_scaled
extract_raw = d["yield_norm"] * quality_alignment
d["extract_norm"] = minmax(extract_raw) + (0.05 if (use_p_prot & (d["p_rate_kg_ha"]>0)).any() else 0.0)

# Agronomy nudges for extract: P good, late N bad (less bad with inhibitor)
p_bonus_flags = d["p_program"].str.contains("starter p", case=False) | d["p_program"].str.contains("adequate", case=False)
d["extract_bonus"]   = np.where(p_bonus_flags, 0.05, 0.0)
late_penalty_scale   = 0.10 * (0.5 if use_n_inhib else 1.0) * (1 - risk_tolerance)
d["extract_penalty"] = np.where(d["late_n"] == 1, late_penalty_scale, 0.0)

# Composite score (Balanced + tie-breaks)
w_balanced = {"profit": 0.40, "quality": 0.25, "yield": 0.20, "sustain": 0.10, "extract": 0.05}
wsum = sum(w_balanced.values()); w_balanced = {k: v/wsum for k, v in w_balanced.items()}
d["score"] = (
    w_balanced["profit"]  * d["profit_norm"]   +
    w_balanced["quality"] * d["quality_score"] +
    w_balanced["yield"]   * d["yield_norm"]    +
    w_balanced["sustain"] * d["sustain_score"] +
    w_balanced["extract"] * d["extract_norm"]
    - hard_penalty
)

# -------- Priority-driven ranking --------
if priority == "Maximize profit":
    d["_pri_key"] = d["profit_eur_ha"];     asc = [False]
elif priority == "Maximize yield":
    d["_pri_key"] = d["yield_eff_t_ha"];    asc = [False]
elif priority == "Lower cost":
    d["_pri_key"] = d["cost_adj_eur_ha"];   asc = [True]
elif priority == "Higher sustainability":
    # small sustainability bump for inhibitor/protector
    sustain_adj = d["sustain_score"] + (0.02 if use_n_inhib else 0.0) + (0.01 if use_p_prot else 0.0)
    d["_pri_key"] = sustain_adj;            asc = [False]
elif priority.startswith("Maximize extract"):
    d["_pri_key"] = d["extract_norm"] + d["extract_bonus"] - d["extract_penalty"]; asc = [False]
else:  # Balanced
    d["_pri_key"] = d["score"];             asc = [False]

# Tie-breakers
sort_cols = ["_pri_key", "score", "profit_eur_ha"]
asc += [False, False]
top = d.sort_values(sort_cols, ascending=asc).head(3).reset_index(drop=True)

# ========== Output ==========
st.subheader(f"Top recommendations for **{region}** ({priority})")

def meets_spec_label(pct: float) -> str:
    return "✅ Meets malting spec" if (low_ok <= pct <= high_ok) else "⚠️ Spec risk"

for i, row in top.iterrows():
    st.markdown(f"### Option {i+1}: {row['variety']} — {row['type']} ({row['maturity']})")
    lever_text = []
    if use_n_inhib: lever_text.append("N inhibitor")
    if use_p_prot:  lever_text.append("P protector")
    lever_str = (" · Levers: " + ", ".join(lever_text)) if lever_text else ""
    st.markdown(f"**{meets_spec_label(row['protein_eff'])}**  ·  Grain N: **{row['grain_n_pct']:.2f}%**  ·  P program: **{row['p_program']}**  ·  Late N: **{'Yes' if row['late_n']==1 else 'No'}**{lever_str}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- **N program:** {row['program_n']}")
        st.markdown(f"- **CP program:** {row['program_cp']}")
        st.markdown(f"- **Expected yield:** {row['yield_eff_t_ha']:.1f} t/ha")
    with col2:
        st.markdown(f"- **Price assumption:** €{row['price_eur_t']:.0f}/t")
        st.markdown(f"- **Revenue:** €{row['revenue_eur_ha']:.0f}/ha")
        st.markdown(f"- **Cost (adj.):** €{row['cost_adj_eur_ha']:.0f}/ha")

    st.markdown(f"**Estimated profit:** €{row['profit_eur_ha']:.0f}/ha")
    st.progress(float(row["score"]))

# --- Narrative for Option 1 ---
if not top.empty:
    best = top.iloc[0]
    why = []
    if use_n_inhib:
        why.append("N inhibitor on: ~20% lower N use, +€20/t urea; protein slightly reduced (−0.1 pp).")
    if use_p_prot:
        why.append("P protector on: +0.15 t/ha with Starter/adequate P and +€4/t premium; +€20/t P cost.")
    if priority == "Maximize profit":
        why.insert(0, f"Highest margin at ~€{best['profit_eur_ha']:.0f}/ha with malting acceptance (protein {best['protein_eff']:.1f}%).")
    elif priority == "Maximize yield":
        why.insert(0, f"Top yield at {best['yield_eff_t_ha']:.1f} t/ha while keeping protein {best['protein_eff']:.1f}% near malting band.")
    elif priority == "Higher sustainability":
        why.insert(0, f"Best sustainability profile with leaner inputs; profit ≈€{best['profit_eur_ha']:.0f}/ha.")
    elif priority == "Lower cost":
        why.insert(0, f"Lowest adjusted cost/ha while maintaining malting acceptance (protein {best['protein_eff']:.1f}%).")
    elif priority.startswith("Maximize extract"):
        why.insert(0, f"Grain N ~{best['grain_n_pct']:.2f}% (protein {best['protein_eff']:.1f}%), strong yield {best['yield_eff_t_ha']:.1f} t/ha for extract focus.")
    else:
        why.insert(0, f"Best overall balance of profit (≈€{best['profit_eur_ha']:.0f}/ha), quality, and risk.")
    st.info(" ".join(why))

# Footers
st.caption(
    "Sustainability score (0–1) is a simple proxy from the N and crop protection intensity of each program "
    "(higher = lower expected footprint). Illustrative only."
)
st.caption(
    "Prices are monthly proxies (FRED barley; World Bank urea) used for defaults. Local/malting spot may differ; "
    "adjust premiums/discounts as needed. Costs reflect estimated kg N from the N program and simple P-rate assumptions."
)
st.write("---")
st.write("Built by Nikolay Georgiev — demo to showcase how AI-style logic can simplify farmer choices.")
