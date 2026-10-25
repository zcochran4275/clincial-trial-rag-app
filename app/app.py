"""Streamlit chat UI for the trial-signal RAG agent, deployed as a Databricks App.

Local:      cd app && streamlit run app.py      (uses .env: DATABRICKS_HOST/TOKEN, DATABRICKS_WAREHOUSE_ID, OPENAI_API_KEY)
Databricks: Compute > Apps > Create app, point it at this folder; see app.yaml.
"""
from __future__ import annotations

import streamlit as st

from trial_signal.rag.agent import answer
from trial_signal.rag.tools import DatabricksBackend

st.set_page_config(page_title="Trial Signal", layout="wide")
st.title("Trial Signal")
st.caption(
    "Ask about completed HER2-therapy trials and post-market FAERS reports. "
    "Research tool only, not medical advice. FAERS counts are reports, not incidence."
)


@st.cache_resource
def backend() -> DatabricksBackend:
    return DatabricksBackend()


EXAMPLES = [
    "Which tucatinib trials allowed patients with untreated brain metastases?",
    "What is the pooled diarrhea rate across trial arms, and how many trials does it come from?",
    "How did neratinib trials define HER2 positivity?",
]

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Try")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask a question") or st.session_state.pop("pending", None)
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching trials..."):
            trace = answer(question, backend())
        st.markdown(trace.answer)
        with st.expander(f"How this was answered ({len(trace.tool_calls)} tool calls, {trace.turns} turns)"):
            for call in trace.tool_calls:
                st.code(f"{call['name']}({call['args']})", language="python")
            st.write("Cited trials:", ", ".join(trace.cited_nct_ids) or "none")
    st.session_state.history.append({"role": "assistant", "content": trace.answer})
