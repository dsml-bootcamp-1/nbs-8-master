"""Step 3: a data dashboard, with filters, charts and caching.

    streamlit run apps/03_data_and_charts.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn dashboard", layout="wide")
st.title("Churn dashboard")


@st.cache_data
def load_data(path="data/churn.csv"):
    churn = pd.read_csv(path)
    churn["churned"] = churn["Churn?"] == "True."
    return churn


churn = load_data()

with st.sidebar:
    st.header("Filters")
    states = st.multiselect("States", sorted(churn["State"].unique()))
    plans = st.radio("International plan", ["all", "yes", "no"], horizontal=True)
    min_calls = st.slider("At least this many calls to customer service", 0, int(churn["CustServ Calls"].max()), 0)

selected = churn[churn["CustServ Calls"] >= min_calls]
if states:
    selected = selected[selected["State"].isin(states)]
if plans != "all":
    selected = selected[selected["Int'l Plan"] == plans]

col1, col2, col3 = st.columns(3)
col1.metric("Customers", f"{len(selected):,}")
col2.metric("Churn rate", f"{selected['churned'].mean():.1%}" if len(selected) else "-")
col3.metric("Average day minutes", f"{selected['Day Mins'].mean():.0f}" if len(selected) else "-")

if selected.empty:
    st.warning("No customers match these filters.")
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Churn rate by customer-service calls")
    st.bar_chart(selected.groupby("CustServ Calls")["churned"].mean())
with right:
    st.subheader("Day minutes, churned against stayed")
    fig, ax = plt.subplots(figsize=(5, 3))
    for churned, group in selected.groupby("churned"):
        ax.hist(group["Day Mins"], bins=30, alpha=0.6, label="churned" if churned else "stayed")
    ax.set_xlabel("Day minutes")
    ax.legend()
    st.pyplot(fig)

st.subheader("The customers")
st.dataframe(selected.drop(columns=["churned"]), width="stretch", hide_index=True)
st.download_button("Download as CSV", selected.to_csv(index=False), "customers.csv", "text/csv")
