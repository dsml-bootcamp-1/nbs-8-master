"""Step 4: put a saved model behind a form. Run `python train_churn_model.py` once first.

    streamlit run apps/04_model_app.py
"""

import json

import joblib
import pandas as pd
import streamlit as st

st.title("Will this customer churn?")


@st.cache_resource
def load_model():
    """Loaded once, then shared by every rerun and every user."""
    model = joblib.load("models/churn_pipeline.joblib")
    with open("models/churn_defaults.json") as file:
        defaults = json.load(file)
    return model, defaults


model, defaults = load_model()

with st.form("customer"):
    left, right = st.columns(2)
    international = left.selectbox("International plan", ["no", "yes"])
    voicemail = left.selectbox("Voicemail plan", ["no", "yes"])
    service_calls = left.number_input("Calls to customer service", min_value=0, max_value=10, value=1)
    day_minutes = right.slider("Day minutes per month", 0.0, 350.0, float(defaults["Day Mins"]))
    international_minutes = right.slider("International minutes per month", 0.0, 20.0, float(defaults["Intl Mins"]))
    months = right.slider("Months as a customer", 1, 243, int(defaults["Account Length"]))
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    customer = dict(defaults)                    # every column the pipeline expects, at a typical value
    customer.update({"Int'l Plan": international, "VMail Plan": voicemail, "CustServ Calls": service_calls,
                     "Day Mins": day_minutes, "Day Charge": day_minutes * 0.17,
                     "Intl Mins": international_minutes, "Intl Charge": international_minutes * 0.27,
                     "Account Length": months})
    probability = model.predict_proba(pd.DataFrame([customer]))[0, 1]
    st.metric("Probability of churn", f"{probability:.0%}")
    st.progress(float(probability))
    if probability >= 0.5:
        st.error("High risk: offer this customer a retention call.")
    else:
        st.success("Low risk.")
    with st.expander("What the model received"):
        st.json(customer)

st.caption("Random forest trained on the telecom churn data (test ROC AUC about 0.89). The other columns are set to typical values.")
