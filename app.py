import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path

st.set_page_config(page_title="Farmer Choice Assistant (Demo)", page_icon="🌾", layout="centered")

# ================= External price fetchers (cached) =================
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

# ================= Helpers =================
def minmax(series: pd.Series) -> pd.Series:
    if series.size == 0:
        return series
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn + 1e-9)

def extract_n_rate(s: str) -> float:
    """Parse '... (60 kg N, ...)' -> 60."""
    if not isinstance(s, str):
        return 0.0
    m = re.search(r"(\d+)\s*kg\s*N", s)
    return float(m.group(1)) if m else 0.0

def estimate_p_rate(program: str) -> float:
    """Crude P product rate proxy (kg/ha) by program text."""
    prog = (program or "").lower()
    if "starter" in prog:  return 25.0
    if "adequate" in prog: return 10.0
    return 0.0

def quality_band(p, low_ok=10.0, high_ok=11.5):
    if low_ok <= p <= high_ok: return "malting"
    if 9.5 <= p < low_ok or high_ok < p <= 12.0: return "edge"
    return "oos"

def calc_price_per_t(quality, base_price, premium, discount, extra_premium=0):
    if quality == "malting": 
        return base_price + premium + extra_premium
    if quality == "oos":
        return max(0, base_price - discount)
    return base_price

def cp_yield_bump(level: str) -> float:
    return {"minimal": 0.0, "standard": 0.2, "intensive": 0.5}.get(level, 0.0)

def cp_cost_bump(level: str) -> float:
    return {"minimal": 0.0, "standard": 50.0, "intensive": 100.0}.get(level, 0.0)

# ================= Data load =================
DATA_PATH = Path("data/barley_scenarios.csv")
df = pd.read_csv(DATA_PATH, sep=",", engine="c", encoding="utf-8", skip_blank_lines=True)
df.columns = [c.strip().lower() for c in df.columns]

required = ["region","crop","variety","type","maturity","program_n","program_cp",
            "yield_t_ha","cost_eur_ha","protein_pct","sustain_score","p_program","late_n"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"CSV missing required columns: {missing}")
    st.stop()

# ================= UI =================
st.title("🌾 Malting Barley (Demo)")
st.markdown("This demo shows how AI-style decision support can simplify complex choices. "
            "It **does not replace** agronomists or farmer expertise — it provides quick, neutral baseline options.")

mode = st.radio("Mode", ["Advisor (rank given programs)", "Optimizer (tune inputs per priority)"], index=0)

region = st.selectbox("Region", sorted(df["region"].unique().tolist()))
priority = st.radio("Your priority", [
    "Balanced",
    "Maximize profit",
    "Maximize yield",
    "Higher sustainability",
    "Lower cost",
    "Maximize extract (starch) within N 1.6–1.75%"
], index=0)
risk_tolerance = st.slider("Risk tolerance (0=low, 1=high)", 0.0, 1.0, 0.3, 0.1)

# Prices (with proxies)
barley_proxy = fetch_barley_price_usd_t()
urea_proxy   = fetch_urea_price_usd_t()
base_price_default  = int(round(barley_proxy, 0)) if barley_proxy else 180
urea_price_default  = int(round(urea_proxy, 0))   if urea_proxy   else 400

st.subheader("Market assumptions")
c1,c2,c3,c4,c5 = st.columns(5)
with c1: base_price = st.number_input("Base barley price (€/t)", 50, 500, base_price_default, 5)
with c2: malting_premium = st.number_input("Malting premium (€/t)", 0, 200, 25, 5)
with c3: oos_discount = st.number_input("Out-of-spec discount (€/t)", 0, 200, 20, 5)
with c4: urea_price = st.number_input("Urea price (USD/t, proxy)", 100, 1200, urea_price_default, 10)
with c5: p_price = st.number_input("P product price (€/t)", 200, 1500, 900, 25)

if mode.startswith("Advisor"):
    st.subheader("Agronomy levers")
    a1,a2 = st.columns(2)
    with a1:
        use_n_inhib = st.checkbox("Use N inhibitor (↓20% N, +€20/t urea, −0.1 pp protein)", value=False)
    with a2:
        use_p_prot  = st.checkbox("Use P protector (+0.15 t/ha with Starter/adequate P, +€20/t P, +€4/t premium)", value=False)

with st.expander("Price sources (proxies)", expanded=False):
    st.caption("Barley: FRED global barley price (PBARLUSDM), monthly proxy. Adjust malting premium for region/spec.")
    st.caption("Urea: World Bank CMO 'Urea (bulk, Middle East)', monthly proxy (context only).")

st.divider()

# ================= Filter & clean =================
d0 = df[df["region"] == region].copy()
if d0.empty:
    st.warning("No data for this region.")
    st.stop()

for c in ["region","crop","variety","type","maturity","program_n","program_cp","p_program"]:
    d0[c] = d0[c].astype(str).str.strip()
d0["late_n"] = d0["late_n"].astype(int)
d0["n_rate_kg_ha"] = d0["program_n"].map(extract_n_rate)
d0["p_rate_kg_ha"] = d0["p_program"].map(estimate_p_rate)

# ================= Core physics (single-candidate calc) =================
def evaluate_candidate(row, dN=0, cp_level="standard", lateN=None, use_inhib=False, use_pprot=False):
    """
    Evaluate one candidate (variety + lever settings). Returns dict with effective metrics and levers.
    """
    # Base values
    y0 = row["yield_t_ha"]
    prot0 = row["protein_pct"]
    n_base = float(row["n_rate_kg_ha"])
    p_rate = float(row["p_rate_kg_ha"])
    sustain0 = float(row["sustain_score"])
    late_base = int(row["late_n"])
    lateN = late_base if lateN is None else int(lateN)

    # Effective N (base + delta, then inhibitor efficiency)
    n_target = max(0.0, n_base + float(dN))
    n_eff = n_target * (0.8 if use_inhib else 1.0)

    # Fertilizer costs (assume €≈$ for demo)
    urea_eff = urea_price + (20 if use_inhib else 0)  # USD/t
    unit_cost_now  = (urea_eff / 1000.0) / 0.46       # €/kg N
    unit_cost_base = (400.0   / 1000.0) / 0.46
    # Reprice base N at today's price + add/less for ΔN
    costN = n_eff * unit_cost_now - n_base * unit_cost_base

    # CP cost bump
    costCP = cp_cost_bump(cp_level)

    # P product cost
    p_price_eff = p_price + (20 if use_pprot else 0)  # €/t
    costP = p_rate * (p_price_eff / 1000.0)

    # Yield response
    alpha = 0.6  # magnitude of N response (demo)
    dY_N = alpha * (np.sqrt(max(0.1, n_eff) / max(0.1, n_base)) - 1.0) if n_base > 0 else 0.0
    dY_CP = cp_yield_bump(cp_level)
    dY_P  = (0.15 if (use_pprot and p_rate > 0) else 0.0)
    y_eff = max(0.1, y0 + dY_N + dY_CP + dY_P)

    # Protein response
    beta1, beta2, beta3 = 0.30, 0.40, 0.20
    dProt_N    = beta1 * np.log(max(0.1, n_eff) / max(0.1, n_base)) if n_base > 0 else 0.0
    dProt_late = (beta2 * (0.5 if use_inhib else 1.0)) if lateN == 1 else 0.0
    dProt_dil  = -beta3 * (y_eff - y0)
    prot_eff = prot0 + dProt_N + dProt_late + dProt_dil + (-0.1 if use_inhib else 0.0)

    # Quality & price
    q = quality_band(prot_eff)
    extra_prem = 4 if (use_pprot and p_rate > 0) else 0
    price_t = calc_price_per_t(q, base_price, malting_premium, oos_discount, extra_prem)

    # Sustainability (proxy)
    sustain = sustain0 \
        - 0.002*(n_eff - n_base) \
        - (0.00 if cp_level=="minimal" else (0.03 if cp_level=="standard" else 0.07)) \
        + (0.02 if lateN==0 else 0.0) \
        + (0.02 if use_inhib else 0.0) + (0.01 if use_pprot else 0.0)
    sustain = float(np.clip(sustain, 0.0, 1.0))

    # Economics
    cost_adj = float(row["cost_eur_ha"]) + costN + costCP + costP
    revenue = y_eff * price_t
    profit  = revenue - cost_adj

    # Extract proxy
    target_prot = 10.5
    align = 1.0 - np.clip(abs(prot_eff - target_prot)/1.5, 0, 1)
    extract_score = align * (y_eff / (y0 + 1e-9))  # relative to baseline

    return {
        "yield": y_eff,
        "protein": prot_eff,
        "grainN": prot_eff/6.25,
        "quality": q,
        "price_eur_t": price_t,
        "cost_adj": cost_adj,
        "revenue": revenue,
        "profit": profit,
        "sustain": sustain,
        "extract_score": extract_score,
        "n_base": n_base,
        "n_eff": n_eff,
        "levers": {
            "ΔN": int(dN),
            "CP": cp_level,
            "Late N": "Yes" if lateN==1 else "No",
            "N inhibitor": "On" if use_inhib else "Off",
            "P protector": "On" if use_pprot else "Off"
        }
    }

# ================= Build candidates (Advisor vs Optimizer) =================
candidates = []

if mode.startswith("Advisor"):
    # Use user toggles; do NOT change ΔN or CP from base
    for _, row in d0.iterrows():
        cp_level = "standard" if "standard" in row["program_cp"].lower() else \
                   ("minimal" if "minimal" in row["program_cp"].lower() else "intensive")
        res = evaluate_candidate(
            row,
            dN=0,
            cp_level=cp_level,
            lateN=row["late_n"],
            use_inhib=use_n_inhib,
            use_pprot=use_p_prot
        )
        res.update({
            "variety": row["variety"], "type": row["type"], "maturity": row["maturity"],
            "p_program": row["p_program"], "program_n": row["program_n"], "program_cp": row["program_cp"]
        })
        candidates.append(res)

else:
    # Optimizer: small grid search around each row
    dN_grid = [-20, 0, +20, +30]
    cp_grid = ["minimal", "standard", "intensive"]
    late_grid = [0, 1]
    inhib_grid = [False, True]

    for _, row in d0.iterrows():
        allow_pp = (row["p_rate_kg_ha"] > 0)
        pprot_grid = [False, True] if allow_pp else [False]
        for dN in dN_grid:
            for cp in cp_grid:
                for late in late_grid:
                    for inhib in inhib_grid:
                        for pprot in pprot_grid:
                            res = evaluate_candidate(row, dN=dN, cp_level=cp, lateN=late, use_inhib=inhib, use_pprot=pprot)
                            # Risk penalty far outside malting band (dampened by risk tolerance)
                            penalty = 0.0
                            if res["protein"] < 9.5 or res["protein"] > 12.0:
                                penalty = 9999 * (1 - risk_tolerance)  # basically exclude at low risk tolerance
                            # Balanced composite (for balanced ranking & tie-breaks)
                            score_bal = 0.40*(res["profit"]) + \
                                        0.25*(1.0 if res["quality"]=="malting" else 0.6 if res["quality"]=="edge" else 0.15) + \
                                        0.20*(res["yield"]) + 0.10*(res["sustain"]) + 0.05*(res["extract_score"])
                            res.update({"_penalty": penalty, "_score_bal": score_bal})
                            res.update({
                                "variety": row["variety"], "type": row["type"], "maturity": row["maturity"],
                                "p_program": row["p_program"], "program_n": row["program_n"], "program_cp": row["program_cp"]
                            })
                            candidates.append(res)

# ================= Ranking =================
dd = pd.DataFrame(candidates)
if dd.empty:
    st.warning("No candidates found.")
    st.stop()

# Priority-driven primary sort key
if priority == "Maximize profit":
    dd["_pri_key"] = dd["profit"];     asc = [False]
elif priority == "Maximize yield":
    dd["_pri_key"] = dd["yield"];      asc = [False]
elif priority == "Lower cost":
    dd["_pri_key"] = dd["cost_adj"];   asc = [True]
elif priority == "Higher sustainability":
    dd["_pri_key"] = dd["sustain"];    asc = [False]
elif priority.startswith("Maximize extract"):
    dd["_pri_key"] = dd["extract_score"]; asc = [False]
else:  # Balanced
    dd["_pri_key"] = dd["_score_bal"] if "_score_bal" in dd else dd["profit"]; asc = [False]

# Apply risk penalty in optimizer
if "_penalty" in dd:
    dd["_pri_key"] = dd["_pri_key"] - dd["_penalty"]

# Tie-breakers: then by profit, then by yield
sort_cols = ["_pri_key", "profit", "yield"]
asc += [False, False]
top = dd.sort_values(sort_cols, ascending=asc).head(3).reset_index(drop=True)

# ================= Output =================
st.subheader(f"Top recommendations for **{region}** ({priority}) — {mode.split()[0]} mode")

def badge(quality): 
    return "✅ Meets malting spec" if quality=="malting" else ("⚠️ Spec risk" if quality=="edge" else "❌ Out of spec")

for i, row in top.iterrows():
    st.markdown(f"### Option {i+1}: {row['variety']} — {row['type']} ({row['maturity']})")
    chips = " · ".join([f"{k}: {v}" for k,v in row["levers"].items()]) if isinstance(row["levers"], dict) else ""
    st.markdown(f"**{badge(row['quality'])}**  ·  Grain N: **{row['grainN']:.2f}%**  ·  P program: **{row['p_program']}**  ·  {chips}")

    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"- **N program (base):** {row['program_n']}")
        # Show N base → effective (Advisor) or effective with base (Optimizer)
        if mode.startswith("Advisor"):
            # In Advisor, only inhibitor can change N rate
            if "N inhibitor" in row["levers"] and row["levers"]["N inhibitor"] == "On":
                st.markdown(f"- **N rate:** {row['n_base']:.0f} → **{row['n_eff']:.0f} kg N/ha** (inhibitor −20%)")
            else:
                st.markdown(f"- **N rate:** {row['n_base']:.0f} kg N/ha")
        else:
            st.markdown(f"- **N rate (effective):** {row['n_eff']:.0f} kg N/ha (base {row['n_base']:.0f})")

        st.markdown(f"- **CP program (base):** {row['program_cp']}")
        st.markdown(f"- **Expected yield:** {row['yield']:.1f} t/ha")

    with c2:
        st.markdown(f"- **Price assumption:** €{row['price_eur_t']:.0f}/t")
        st.markdown(f"- **Revenue:** €{row['revenue']:.0f}/ha")
        st.markdown(f"- **Cost (adj.):** €{row['cost_adj']:.0f}/ha")

    st.markdown(f"**Estimated profit:** €{row['profit']:.0f}/ha")

    # Display-only progress based on normalized _pri_key within the top-3
    bar_norm = 0.5
    if len(top) > 1:
        keyvals = top["_pri_key"].fillna(0).to_numpy(dtype=float)
        bar_norm = float((keyvals[i]-keyvals.min())/(keyvals.max()-keyvals.min()+1e-9))
    st.progress(bar_norm)

# Narrative for Option 1
if not top.empty:
    b = top.iloc[0]
    why = []
    if mode.startswith("Optimizer"):
        why.append("Optimizer tuned the program levers for your priority.")
    if priority == "Maximize profit":
        why.append(f"Highest margin (~€{b['profit']:.0f}/ha) with malting acceptance (protein {b['protein']:.1f}%).")
    elif priority == "Maximize yield":
        why.append(f"Top yield ({b['yield']:.1f} t/ha) while staying within/near malting spec (protein {b['protein']:.1f}%).")
    elif priority == "Higher sustainability":
        why.append(f"Best sustainability profile; inputs cut where they hurt footprint without dropping out of spec.")
    elif priority == "Lower cost":
        why.append(f"Lowest adjusted cost/ha while maintaining malting acceptance (protein {b['protein']:.1f}%).")
    elif priority.startswith("Maximize extract"):
        why.append(f"Grain N ~{b['grainN']:.2f}% (protein {b['protein']:.1f}%), strong yield for higher extract.")
    else:
        why.append(f"Balanced trade-off among profit, quality and risk.")
    if isinstance(b["levers"], dict):
        why.append("Chosen changes: " + ", ".join([f"{k} {v}" for k,v in b['levers'].items()]))
    st.info(" ".join(why))

# Footers
st.caption("Sustainability score (0–1) is a simple proxy using N & crop protection intensity (illustrative only).")
st.caption("Prices are monthly proxies (FRED barley; World Bank urea) for defaults. Local/malting spot may differ; adjust premiums. "
           "Physics are simplified demo coefficients; tune with trials/farm records for local calibration.")
st.write("---")
st.write("Built by Nikolay Georgiev — demo to showcase how AI-style logic can simplify farmer choices.")
