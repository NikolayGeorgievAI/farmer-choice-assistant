import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Farmer Choice Assistant (Demo)", page_icon="🌾", layout="centered")

# --- Load data ---
DATA_PATH = Path("data/barley_scenarios.csv")
df = pd.read_csv(DATA_PATH)

# --- Sidebar / Inputs ---
st.title("🌾 Farmer Choice Assistant — Malting Barley (Demo)")

st.markdown(
    "This demo shows how AI-style decision support can simplify complex choices. "
    "It **does not replace** agronomists or farmer expertise — it provides quick, neutral baseline options."
)

region = st.selectbox("Region", sorted(df["region"].unique().tolist()))
priority = st.radio(
    "Your priority",
    ["Balanced", "Maximize yield", "Meet malting specs", "Lower cost", "Higher sustainability"],
    index=0
)
risk_tolerance = st.slider("Risk tolerance (0=low, 1=high)", 0.0, 1.0, 0.3, 0.1)

st.divider()

# --- Filter by region ---
d = df[df["region"] == region].copy()
if d.empty:
    st.warning("No data for this region.")
    st.stop()

# --- Scoring logic ---
# Normalize metrics for comparability
def minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

d["yield_norm"] = minmax(d["yield_t_ha"])
d["cost_norm"] = 1 - minmax(d["cost_eur_ha"])          # lower cost = higher score
# Protein target for malting: <= 11.5 (lower better up to threshold; if above, penalize)
protein_target = 11.5
d["protein_score"] = np.where(d["protein_pct"] <= protein_target,
                              1 - (d["protein_pct"] / protein_target)*0.3,  # gentle slope below target
                              np.maximum(0, 1 - (d["protein_pct"] - protein_target)*0.5))  # steep penalty above
# Clip to [0,1]
d["protein_score"] = d["protein_score"].clip(0,1)

# Base weights
w = {
    "yield": 0.33,
    "malting": 0.27,
    "cost": 0.20,
    "sustain": 0.20
}

# Adjust weights by priority
if priority == "Maximize yield":
    w["yield"] += 0.20; w["cost"] -= 0.10; w["sustain"] -= 0.10
elif priority == "Meet malting specs":
    w["malting"] += 0.20; w["yield"] -= 0.10; w["cost"] -= 0.10
elif priority == "Lower cost":
    w["cost"] += 0.20; w["yield"] -= 0.10; w["sustain"] -= 0.10
elif priority == "Higher sustainability":
    w["sustain"] += 0.20; w["yield"] -= 0.10; w["cost"] -= 0.10

# Normalize weights to sum=1
wsum = sum(w.values())
for k in w:
    w[k] = max(0, w[k]) / wsum

# Add risk handling: if low tolerance, penalize high protein & very intensive CP
cp_penalty = d["program_cp"].str.contains("Intensive").astype(int) * (1 - risk_tolerance) * 0.15
protein_penalty = np.where(d["protein_pct"] > protein_target, (1 - risk_tolerance) * 0.20, 0.0)

# Final score
d["score"] = (
    w["yield"] * d["yield_norm"] +
    w["malting"] * d["protein_score"] +
    w["cost"] * d["cost_norm"] +
    w["sustain"] * d["sustain_score"]
    - cp_penalty - protein_penalty
)

# Rank top 3 scenarios
top = d.sort_values("score", ascending=False).head(3).reset_index(drop=True)

# --- Show recommendations ---
st.subheader(f"Top recommendations for **{region}** ({priority}, risk tolerance {risk_tolerance:.1f})")

for i, row in top.iterrows():
    st.markdown(f"### Option {i+1}: {row['variety']} — {row['type']} ({row['maturity']})")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- **N program:** {row['program_n']}")
        st.markdown(f"- **CP program:** {row['program_cp']}")
        st.markdown(f"- **Expected yield:** {row['yield_t_ha']:.1f} t/ha")
    with col2:
        st.markdown(f"- **Cost:** €{row['cost_eur_ha']:.0f}/ha")
        st.markdown(f"- **Protein est.:** {row['protein_pct']:.1f}% (≤ 11.5% targets malting)")
        st.markdown(f"- **Sustainability score:** {row['sustain_score']:.2f}")
    st.progress(float(row["score"]))

st.caption(
    "Demo only: values are simplified and illustrative. Use alongside agronomist advice, farm records, and local regulations."
)

# --- (Optional) Natural-language explanation without external LLM ---
if not top.empty:
    best = top.iloc[0]
    rationale = (
        f"**Why Option 1?** It balances malting quality (protein ~{best['protein_pct']:.1f}%) with solid yield "
        f"({best['yield_t_ha']:.1f} t/ha) and a {best['program_n'].lower()} / {best['program_cp'].lower()} program. "
        f"Total cost ≈ €{best['cost_eur_ha']:.0f}/ha and sustainability score {best['sustain_score']:.2f}. "
        f"If malt acceptance is critical, options with protein above 11.5% are riskier despite higher yields."
    )
    st.info(rationale)

# --- Footer ---
st.write("---")
st.write("Built by Nikolay Georgiev — demo to showcase how AI-like logic can simplify farmer choices.")
