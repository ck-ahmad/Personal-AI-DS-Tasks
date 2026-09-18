
"""Minimal Streamlit demo UI.

Run:
    streamlit run streamlit_app.py
"""

import streamlit as st
from capstone_api import safe_call, normalize_prediction_disclaimer

st.set_page_config(page_title="AFL Assistant", page_icon="🏉", layout="centered")
st.title("🏉 AFL Assistant")
st.caption("AFL-only chat + retrieval + prediction")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

query = st.chat_input("Ask an AFL question...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            result = safe_call(query, "streamlit-demo")
            response = normalize_prediction_disclaimer(
                str(result.get("final_response", "")),
                result.get("intent"),
            )
            st.markdown(response)
            if result.get("intent"):
                st.caption(
                    f"Intent: {result.get('intent')} · "
                    f"Tool: {result.get('tool_name') or 'none'}"
                )
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as exc:
            response = f"Safe error: {exc}"
            st.error(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
