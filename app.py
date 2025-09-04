import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Farmer Choice Assistant (Demo)", page_icon="🌾", layout="centered")

# ---------- Robust CSV loader ----------
def load_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Data file not found at: {path}. Make sure you have data/barley_scenarios.csv in the repo.")
        st.stop()

    # Try common separators first (comma/semicolon/tab/pipe), then fallback to autodetect
    seps = [",", ";", "\t", "|"]
    last_err = None
    for sep in seps:
        try:
            df_try = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            # Require at least several columns to consider it valid
            if df_try.shape[1] >= 5:
                return df_try
        except Exception as e:
            last_err = e
            continue

    # Fallback: python engine autodetect, skip bad lines rather than crash
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", on_bad_lines="skip")
    except Exception as e:
        st.error(f"Could not parse CSV. Last error: {e or last_err}")
        st.stop()

# --- Load data (strict comma CSV) ---
DATA_PATH = Path("data/barley_scenarios.csv")

# Read as plain UTF-8 CSV with comma delimiter
df = pd.read_csv(
    DATA_PATH,
    sep=",",            # force comma
    engine="c",         # fast/strict parser
    encoding="utf-8",   # Excel usually saves fine with this
    skip_blank_lines=True
)

# Normalize headers
df.columns = [c.strip().lower() for c in df.columns]


# Debug (visible in the app; remove later if you want)
with st.expander("Debug: data preview", expanded=False):
    st.write("Columns loaded:", list(df.columns))
    st.dataframe(df.head())

# Validate required columns
required = ["region","crop","variety","type","maturity",
            "program_n","program_cp","yield_t_ha","cost_eur_ha",
            "protein_pct","sustain_score"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"CSV is missing required columns: {missing}\nFound columns: {list(df.columns)}")
    st.stop()

# Clean string fields
for c in ["region","crop","variety","type","maturity","program_n","program_cp"]:
    df[c] = df[c].astype(str).str.strip()

# ---------- UI ----------
st.title("🌾 Malting Barley (Demo)")
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

# ---------- Filter ----------
d = df[df["region"] == region].copy()
if d.empty:
    st.warning("No data for this region.")
    st.stop()

# ---------- Scoring logic ----------
def minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

d["yield_norm"] = minmax(d["yield_t_ha"])
d["cost_norm"] = 1 - minmax(d["cost_eur_ha"])  # lower cost = higher score

protein_target = 11.5
d["protein_score"] = np.where(
    d["protein_pct"] <= protein_target,
    1 - (d["protein_pct"] / protein_target) * 0.3,
    np.maximum(0, 1 - (d["protein_pct"] - protein_target) * 0.5),
).clip(0, 1)

weights = {"yield": 0.33, "malting": 0.27, "cost": 0.20, "sustain": 0.20}
if priority == "Maximize yield":
    weights["yield"] += 0.20; weights["cost"] -= 0.10; weights["sustain"] -= 0.10
elif priority == "Meet malting specs":
    weights["malting"] += 0.20; weights["yield"] -= 0.10; weights["cost"] -= 0.10
elif priority == "Lower cost":
    weights["cost"] += 0.20; weights["yield"] -= 0.10; weights["sustain"] -= 0.10
elif priority == "Higher sustainability":
    weights["sustain"] += 0.20; weights["yield"] -= 0.10; weights["cost"] -= 0.10
wsum = sum(max(0, v) for v in weights.values())
weights = {k: max(0, v) / wsum for k, v in weights.items()}

cp_penalty = d["program_cp"].str.contains("Intensive", case=False).astype(int) * (1 - risk_tolerance) * 0.15
protein_penalty = np.where(d["protein_pct"] > protein_target, (1 - risk_tolerance) * 0.20, 0.0)

d["score"] = (
    weights["yield"] * d["yield_norm"] +
    weights["malting"] * d["protein_score"] +
    weights["cost"] * d["cost_norm"] +
    weights["sustain"] * d["sustain_score"] -
    cp_penalty - protein_penalty
)

top = d.sort_values("score", ascending=False).head(3).reset_index(drop=True)

# ---------- Output ----------
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

st.caption("Demo only: values are simplified and illustrative. Use alongside agronomist advice, farm records, and local regulations.")

if not top.empty:
    best = top.iloc[0]
    st.info(
        f"**Why Option 1?** It balances malting quality (protein ~{best['protein_pct']:.1f}%) "
        f"with solid yield ({best['yield_t_ha']:.1f} t/ha) and a {best['program_n'].lower()} / "
        f"{best['program_cp'].lower()} program. Cost ≈ €{best['cost_eur_ha']:.0f}/ha "
        f"and sustainability score {best['sustain_score']:.2f}. "
        f"If malt acceptance is critical, options with protein above 11.5% are riskier despite higher yields."
    )

st.write("---")
st.write("Built by Nikolay Georgiev — demo to showcase how AI-style logic can simplify farmer choices.")

