"""Step 1: your first app, and how Streamlit reruns it.

    streamlit run apps/01_hello.py
"""

import streamlit as st

st.title("Hello, Streamlit")
st.write("This page is a Python script. Every time you touch a widget, Streamlit runs the whole script again, from top to bottom.")

name = st.text_input("Your name")
if st.button("Greet"):
    st.success(f"Hello {name}!")

st.markdown("---")
st.subheader("Text, numbers and data")
st.markdown("You can write **Markdown**, show `code`, and display numbers:")
col1, col2, col3 = st.columns(3)
col1.metric("Customers", "3,333")
col2.metric("Churn rate", "14.5%", "-1.2 points")
col3.metric("Model ROC AUC", "0.89")
