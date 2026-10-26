"""Step 5: a chatbot with memory and streaming. It needs your OpenAI key (see the README).

    streamlit run apps/05_chatbot.py
"""

import os

import streamlit as st
from openai import OpenAI

MODEL = "gpt-5.4-nano"
SYSTEM = "You are a friendly support assistant of a digital bank. Answer in at most three sentences."


def api_key():
    """st.secrets on Streamlit Cloud or with a local .streamlit/secrets.toml; otherwise the .env file."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        return os.environ.get("OPENAI_API_KEY")


@st.cache_resource
def get_client(key):
    return OpenAI(api_key=key)


st.title("Bank support chat")
key = api_key()
if not key:
    st.error("No OpenAI key found. Add it to .streamlit/secrets.toml or .env (see the README).")
    st.stop()
client = get_client(key)

if "messages" not in st.session_state:          # the conversation survives reruns
    st.session_state.messages = []

for message in st.session_state.messages:        # redraw the history at every rerun
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Ask about your card, transfers or the app"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=MODEL, stream=True,
            messages=[{"role": "system", "content": SYSTEM}] + st.session_state.messages)
        answer = st.write_stream(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.write(f"{len(st.session_state.messages)} messages in this conversation")
    if st.button("New conversation"):
        st.session_state.messages = []
        st.rerun()
