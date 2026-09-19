import json

import pandas as pd
import streamlit as st

from engine import KB, extract, merge, questions, recommend, render

st.set_page_config(page_title="AI Biodiversity Scientist", page_icon="🌍", layout="wide")
st.title("🌍 AI Biodiversity Scientist")
st.caption("Retrieval-grounded, multi-metric recommendations with cited evidence. Built for Darukaa.Earth.")

if "profile" not in st.session_state:
    st.session_state.profile = {}
    st.session_state.chat = []
    st.session_state.trace = None

SAMPLE = {"soc": 0.3, "rainfall": "low", "land_use": "monoculture", "crop": "wheat", "region": "semi-arid"}


def respond(user_text: str, structured: dict | None = None):
    new = extract(user_text) if user_text else {}
    if structured:
        new.update(structured)
    st.session_state.profile = merge(st.session_state.profile, new)
    p = st.session_state.profile
    skip = "skip" in user_text.lower() or "just recommend" in user_text.lower()
    missing = questions(p, include_optional=False)
    if missing and not skip:
        known = {k: v for k, v in p.items() if not k.startswith("climate_zone")}
        msg = "To give evidence-based advice I need a bit more information:\n\n" + "\n".join(f"- {q}" for q in missing)
        if known:
            msg += "\n\nSo far I have: `" + json.dumps(known) + "`  (say *skip* to proceed with what I have)"
        return msg
    issues, recs, links = recommend(p)
    st.session_state.trace = recs
    msg = render(issues, recs, links)
    opt = questions(p)
    if opt and issues:
        msg += "\n---\n" + "\n".join(f"- {q}" for q in opt)
    return msg


# ---- Sidebar: structured input + memory view ----
with st.sidebar:
    st.header("Structured input (JSON)")
    raw = st.text_area("Paste JSON", value=json.dumps(SAMPLE, indent=2), height=180)
    if st.button("Analyse JSON"):
        try:
            data = json.loads(raw)
            st.session_state.chat.append(("user", f"[JSON input] {json.dumps(data)}"))
            st.session_state.chat.append(("assistant", respond("", data)))
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
    st.header("Conversation memory")
    st.json(st.session_state.profile)
    if st.button("Reset conversation"):
        st.session_state.clear()
        st.rerun()
    st.caption(f"Knowledge base: {len(KB)} cited interventions, TF-IDF vector index + trigger matching.")

# ---- Chat ----
if not st.session_state.chat:
    st.info('Try: "Biodiversity is declining on my land" (I will ask follow-ups), or: '
            '"Soil organic carbon 0.3%, low rainfall, monoculture wheat, semi-arid"')

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

if prompt := st.chat_input("Describe your land, soil, climate..."):
    st.session_state.chat.append(("user", prompt))
    st.session_state.chat.append(("assistant", respond(prompt)))
    st.rerun()

# ---- Retrieval transparency ----
if st.session_state.trace:
    with st.expander("How knowledge was retrieved (score = 0.4 x trigger matches + TF-IDF similarity)"):
        st.dataframe(pd.DataFrame([
            {"knowledge entry": r["entry"]["title"], "matched issues": ", ".join(r["hit"]),
             "similarity": round(r["sim"], 3), "score": round(r["score"], 3)}
            for r in st.session_state.trace]), use_container_width=True)
