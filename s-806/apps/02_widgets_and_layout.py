"""Step 2: widgets, layout, and keeping values between reruns with session state.

    streamlit run apps/02_widgets_and_layout.py
"""

import streamlit as st

st.title("Widgets and layout")

with st.sidebar:
    st.header("Settings")
    plan = st.selectbox("Plan", ["Basic", "Premium"])
    months = st.slider("Months as a customer", min_value=1, max_value=72, value=12)
    newsletter = st.checkbox("Subscribed to the newsletter")

st.write(f"A **{plan}** customer for **{months} months**, newsletter: **{'yes' if newsletter else 'no'}**.")

tab_form, tab_counter = st.tabs(["A form", "Session state"])

with tab_form:
    st.write("Inside a form, nothing reruns until the user presses the submit button.")
    with st.form("contact"):
        email = st.text_input("Email")
        message = st.text_area("Message")
        sent = st.form_submit_button("Send")
    if sent:
        if "@" not in email:
            st.error("Please enter a valid email.")
        else:
            st.success(f"Thanks! We will answer {email} soon.")

with tab_counter:
    st.write("A normal variable is reset at every rerun. `st.session_state` keeps values for as long as the browser tab is open.")
    if "clicks" not in st.session_state:
        st.session_state.clicks = 0
    if st.button("Click me", key="counter"):
        st.session_state.clicks += 1
    st.write(f"You clicked {st.session_state.clicks} times.")
