import json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hero or Villain?", page_icon="🦸", layout="centered")


@st.cache_resource
def load_artifacts():
    weights = dict(np.load("villain_model.npz"))
    with open("villain_meta.json") as f:
        meta = json.load(f)
    chars = pd.read_csv("characters.csv")
    return weights, meta, chars


def predict_villain_prob(features, w):
    """Same network as the notebook: scale -> Dense(8, ReLU) -> Dense(1, sigmoid)."""
    x = (features - w["mean"]) / w["scale"]
    hidden = np.maximum(0, x @ w["W1"] + w["b1"])
    logit = hidden @ w["W2"] + w["b2"]
    return float(1 / (1 + np.exp(-logit.ravel()[0])))


def build_features(choices, appearances, year, meta):
    """Turn the user's choices into the same 25 columns the model was trained on."""
    row = pd.get_dummies(pd.DataFrame([choices]), dtype=float)
    row["appearances"] = np.log1p(appearances)
    row["year"] = year
    row = row.reindex(columns=meta["feature_columns"], fill_value=0.0)
    return row.values.astype(float)


def show_result(p):
    label = "🦹 Villain" if p >= 0.5 else "🦸 Hero"
    st.metric("Prediction", label, f"{p:.0%} chance of being a villain", delta_color="off")
    st.progress(p, text=f"Villain probability: {p:.0%}")


weights, meta, chars = load_artifacts()
cats = meta["categories"]

st.title("🦸 Hero or Villain?")
st.write("A small neural network trained on 15,000+ DC and Marvel characters guesses "
         "whether a character is a **hero** or a **villain** from facts like sex, identity, "
         "eye and hair colour, appearances and first-appearance year.")

tab_search, tab_create = st.tabs(["🔍 Look up a character", "✏️ Create your own"])

with tab_search:
    query = st.text_input("Type a character name", placeholder="e.g. Batman, Joker, Magneto")
    if query:
        matches = chars[chars["name"].str.contains(query, case=False, na=False, regex=False)]
        matches = matches.sort_values("appearances", ascending=False).head(50)
        if matches.empty:
            st.warning("No character found. Try another name.")
        else:
            name = st.selectbox(f"{len(matches)} match(es) – pick one", matches["name"])
            row = matches[matches["name"] == name].iloc[0]
            show_result(float(row["p_villain"]))
            correct = (row["p_villain"] >= 0.5) == (row["alignment"] == "Villain")
            st.write(f"**Publisher:** {row['publisher']}  |  **Actually:** {row['alignment']}  |  "
                     f"{'✅ model is right' if correct else '❌ model is wrong'}")

with tab_create:
    c1, c2 = st.columns(2)
    with c1:
        choices = {
            "publisher": st.selectbox("Publisher", cats["publisher"]),
            "sex": st.selectbox("Sex", cats["sex"]),
            "identity": st.selectbox("Identity", cats["identity"]),
        }
    with c2:
        choices["alive"] = st.selectbox("Alive?", cats["alive"])
        choices["eye"] = st.selectbox("Eye colour", cats["eye"])
        choices["hair"] = st.selectbox("Hair colour", cats["hair"])
    appearances = st.number_input("Number of comic appearances", min_value=0, max_value=5000, value=10, step=1)
    year = st.slider("Year of first appearance", meta["year_min"], meta["year_max"], int(meta["year_median"]))

    if st.button("Predict", type="primary"):
        p = predict_villain_prob(build_features(choices, appearances, year, meta), weights)
        show_result(p)

st.caption(f"Model: 25 inputs → 8 hidden (ReLU) → 1 output (sigmoid), trained with PyTorch. "
           f"Test accuracy: {meta['test_accuracy']:.0%}. Data: FiveThirtyEight comic characters dataset.")
