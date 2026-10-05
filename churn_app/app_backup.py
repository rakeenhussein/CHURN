"""
Customer Churn Prediction — Streamlit application
SUZA Data Science Practical Training, Group Four

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

RAW_PREDICTORS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",j
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "tenure", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


# ----------------------------------------------------------------------------
# Load the saved model artifacts (cached so this runs once, not on every click)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("churn_scaler.pkl")
    features = joblib.load("churn_features.pkl")
    return model, scaler, features


try:
    model, scaler, feature_names = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Place **churn_model.pkl**, **churn_scaler.pkl** "
        "and **churn_features.pkl** in the same folder as this app."
    )
    st.stop()


# ----------------------------------------------------------------------------
# Shared preprocessing — must mirror Stage 3 cleaning + Stage 4 encoding exactly
# ----------------------------------------------------------------------------
def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw customer rows into the exact 23 encoded columns the model expects."""
    df = raw_df.copy()

    # Stage 3 cleaning rules, applied again to new data
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    if "SeniorCitizen" in df.columns:
        # Accept either the 0/1 coding or the Yes/No coding
        df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes",
                                                           "0": "No", "1": "Yes"})

    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]:
        if col in df.columns:
            df[col] = df[col].replace("No internet service", "No")

    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    # Stage 4 encoding.
    # NOTE: drop_first is deliberately NOT used here, even though training used
    # it. On a single row, drop_first would drop the customer's own category —
    # e.g. a customer paying by electronic check would end up with every payment
    # dummy set to 0, silently producing the wrong prediction. Encoding all
    # categories and then reindexing to the saved training columns gives the
    # correct result for one row or a million, because the reference categories
    # that drop_first removed are simply absent from feature_names.
    encoded = pd.get_dummies(df[RAW_PREDICTORS])

    # Force the exact training columns, in the exact training order, filling any
    # category this batch did not contain with 0.
    encoded = encoded.reindex(columns=feature_names, fill_value=0)

    # Scale the numerical columns with the scaler fitted during training
    encoded[NUMERIC_COLS] = scaler.transform(encoded[NUMERIC_COLS])

    return encoded


def risk_band(probability: float) -> tuple:
    """Map a churn probability to a label and colour."""
    if probability >= 0.70:
        return "HIGH RISK", "#C0392B"
    if probability >= 0.50:
        return "MEDIUM RISK", "#E67E22"
    if probability >= 0.30:
        return "LOW RISK", "#F1C40F"
    return "VERY LOW RISK", "#27AE60"

def csore_customers(raw_df):
    raw_df.copy()
    x = preprocess(df)
    probs=model.predict_proba(x)[:, 1]
    df["ChurnProbability"] = probs
    df["RiskBand"] = [risk_band(p)[0] for p in probs]
    return df

def get_data():


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("Customer Churn Prediction")
st.caption(
    "Predicting which telecommunications customers are likely to discontinue "
    "their service, using account, service and billing information."
)

with st.expander("About this model"):
    st.markdown(
        """
        **Model:** Logistic Regression with balanced class weights
        **Trained on:** 4,225 customers (60% of the Telco Customer Churn dataset)
        **Held-out test performance:** recall 0.80, precision 0.51, ROC-AUC 0.849

        The model identifies **80% of customers who go on to churn**. Because the
        target is imbalanced (26.5% churn), the model is deliberately tuned to
        catch churners rather than to maximise overall accuracy — a missed churner
        costs more than an unnecessary retention offer.

        **Intended use:** a ranked list of at-risk customers for the retention team
        to work through. Roughly half of flagged customers would have stayed anyway,
        so predictions should support human decisions, not replace them.
        """
    )

tab_single, tab_batch = st.tabs(["Single customer", "Batch upload"])


# ----------------------------------------------------------------------------
# TAB 1 — single customer
# ----------------------------------------------------------------------------
with tab_single:
    st.subheader("Enter customer details")

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Has partner", ["No", "Yes"])
            dependents = st.selectbox("Has dependents", ["No", "Yes"])

            st.markdown("**Account**")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )

        with c2:
            st.markdown("**Services**")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox("Multiple lines", ["No", "Yes"])
            internet = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )
            online_sec = st.selectbox("Online security", ["No", "Yes"])
            online_bak = st.selectbox("Online backup", ["No", "Yes"])
            device = st.selectbox("Device protection", ["No", "Yes"])

        with c3:
            st.markdown("**Services (continued)**")
            tech = st.selectbox("Technical support", ["No", "Yes"])
            tv = st.selectbox("Streaming TV", ["No", "Yes"])
            movies = st.selectbox("Streaming movies", ["No", "Yes"])

            st.markdown("**Billing**")
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly = st.number_input(
                "Monthly charges", min_value=0.0, max_value=200.0,
                value=70.0, step=0.05
            )
            total = st.number_input(
                "Total charges", min_value=0.0, max_value=10000.0,
                value=840.0, step=1.0
            )

        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if submitted:
        customer = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "PhoneService": phone,
            "MultipleLines": multiple, "InternetService": internet,
            "OnlineSecurity": online_sec, "OnlineBackup": online_bak,
            "DeviceProtection": device, "TechSupport": tech,
            "StreamingTV": tv, "StreamingMovies": movies, "tenure": tenure,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": total,
        }])

        X_new = preprocess(customer)
        probability = float(model.predict_proba(X_new)[0, 1])
        label, colour = risk_band(probability)

        st.divider()
        r1, r2 = st.columns([1, 2])

        with r1:
            st.metric("Churn probability", f"{probability:.1%}")
            st.markdown(
                f"<div style='background:{colour};color:white;padding:12px;"
                f"border-radius:6px;text-align:center;font-weight:600;'>"
                f"{label}</div>",
                unsafe_allow_html=True,
            )

        with r2:
            st.progress(probability)
            if probability >= 0.50:
                st.warning(
                    f"This customer is predicted to churn. Estimated risk "
                    f"{probability:.1%} — recommend contacting them with a "
                    f"retention offer."
                )
            else:
                st.success(
                    f"This customer is predicted to stay. Estimated risk "
                    f"{probability:.1%} — no immediate action required."
                )

            st.caption(
                "Threshold: 50%. The model finds about 80% of real churners at "
                "this cut-off, and roughly half of flagged customers would have "
                "stayed anyway."
            )

        # Show which of this customer's characteristics push risk up or down
        contributions = pd.DataFrame({
            "Variable": feature_names,
            "Value": X_new.iloc[0].values,
            "Coefficient": model.coef_[0],
        })
        contributions["Effect"] = contributions["Value"] * contributions["Coefficient"]
        contributions = contributions[contributions["Value"] != 0]
        contributions = contributions.reindex(
            contributions["Effect"].abs().sort_values(ascending=False).index
        ).head(6)
        contributions["Direction"] = contributions["Effect"].apply(
            lambda v: "Increases risk" if v > 0 else "Reduces risk"
        )

        st.divider()
        st.subheader("What is driving this prediction")
        st.dataframe(
            contributions[["Variable", "Direction", "Effect"]]
            .round(3).reset_index(drop=True),
            use_container_width=True,
        )
        st.caption(
            "These show association, not causation. They indicate which "
            "characteristics the model weighted most for this customer."
        )


# ----------------------------------------------------------------------------
# TAB 2 — batch upload
# ----------------------------------------------------------------------------
with tab_batch:
    st.subheader("Score a list of customers")
    st.markdown(
        "Upload a CSV with the same columns as the project dataset. "
        "A `customerID` column is optional but recommended so predictions can "
        "be matched back to customers."
    )

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(raw):,}** rows and **{raw.shape[1]}** columns.")

        missing = [c for c in RAW_PREDICTORS if c not in raw.columns]
        if missing:
            st.error(f"These required columns are missing: {', '.join(missing)}")
            st.stop()

        X_new = preprocess(raw)
        probs = model.predict_proba(X_new)[:, 1]

        results = pd.DataFrame()
        if "customerID" in raw.columns:
            results["customerID"] = raw["customerID"]
        results["tenure"] = raw["tenure"]
        results["Contract"] = raw["Contract"]
        results["MonthlyCharges"] = raw["MonthlyCharges"]
        results["ChurnProbability"] = probs.round(4)
        results["Prediction"] = ["Churn" if p >= 0.5 else "Stay" for p in probs]
        results["RiskBand"] = [risk_band(p)[0] for p in probs]
        results = results.sort_values("ChurnProbability", ascending=False)

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Customers scored", f"{len(results):,}")
        m2.metric("Predicted to churn", f"{(probs >= 0.5).sum():,}")
        m3.metric("High risk (≥70%)", f"{(probs >= 0.70).sum():,}")
        m4.metric("Average risk", f"{probs.mean():.1%}")

        st.subheader("Customers ranked by churn risk")
        top_n = st.slider(
            "Show top N customers", 10, min(500, len(results)),
            min(50, len(results))
        )
        st.dataframe(results.head(top_n), use_container_width=True)

        st.download_button(
            "Download full results as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
            type="primary",
        )

        st.caption(
            "Results are sorted highest risk first, so the retention team can "
            "work down the list until its budget or capacity is reached."
        )


st.divider()
st.caption(
    "SUZA Data Science Practical Training · Group Four · "
    "Predicting Customer Churn in a Telecommunications Company"
)
