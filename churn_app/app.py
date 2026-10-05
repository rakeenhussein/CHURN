"""
Customer Churn Prediction — Streamlit dashboard
SUZA Data Science Practical Training, Group Four

Run with:  streamlit run app.py

Tabs
  1. Overview         — portfolio summary of the uploaded customer file
  2. Customer search  — look up one customer by (partial) customerID
  3. Live monitor     — SIMULATED live feed that replays the uploaded file
  4. Single customer  — score a hand-entered customer
"""

import time
import warnings

import altair as alt
import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

RAW_PREDICTORS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "tenure", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

INTERNET_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

HIGH_RISK = 0.70
CHURN_THRESHOLD = 0.50

BAND_ORDER = ["VERY LOW RISK", "LOW RISK", "MEDIUM RISK", "HIGH RISK"]
BAND_COLOURS = ["#27AE60", "#F1C40F", "#E67E22", "#C0392B"]

DEMO_FILE = "sample_customers.csv"


# ----------------------------------------------------------------------------
# Load the saved model artifacts (cached so this runs once, not on every click)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # The .pkl files were saved with scikit-learn 1.6.1. Loading them in a newer
    # version prints an InconsistentVersionWarning; predictions were verified to
    # reproduce the notebook's test metrics exactly, so silence it here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
# Shared preprocessing — must mirror the encoding the saved model was trained on
# ----------------------------------------------------------------------------
def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw customer rows into the exact encoded columns the model expects."""
    df = raw_df.copy()

    # Stage 3 cleaning rules, applied again to new data
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    if "SeniorCitizen" in df.columns:
        # Accept either the 0/1 coding or the Yes/No coding
        df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes",
                                                           "0": "No", "1": "Yes"})

    # Internet add-on services.
    # NOTE: the saved model was trained on cleaned_project_dataset.csv, which
    # KEPT the "No internet service" category (churn_features.pkl contains
    # e.g. "OnlineSecurity_No internet service"). Collapsing it to "No" here
    # would switch those columns off and roughly double the predicted risk of
    # every customer without internet. So we do the opposite: any customer with
    # InternetService == "No" gets "No internet service" in the six add-on
    # columns — this also makes the single-customer form (which offers only
    # No/Yes) encode exactly like the training data.
    if "InternetService" in df.columns:
        no_internet = df["InternetService"] == "No"
        for col in INTERNET_ADDONS:
            if col in df.columns:
                df.loc[no_internet, col] = "No internet service"

    # Rule 4 WAS applied before training (there is no
    # "MultipleLines_No phone service" column), so keep it.
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
    if probability >= HIGH_RISK:
        return "HIGH RISK", "#C0392B"
    if probability >= CHURN_THRESHOLD:
        return "MEDIUM RISK", "#E67E22"
    if probability >= 0.30:
        return "LOW RISK", "#F1C40F"
    return "VERY LOW RISK", "#27AE60"


def band_badge(probability: float) -> None:
    label, colour = risk_band(probability)
    st.markdown(
        f"<div style='background:{colour};color:white;padding:12px;"
        f"border-radius:6px;text-align:center;font-weight:600;'>"
        f"{label}</div>",
        unsafe_allow_html=True,
    )


def score_customers(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Score every row once. Returns the raw rows plus prediction columns."""
    df = raw_df.copy()
    if "customerID" not in df.columns:
        df.insert(0, "customerID", [f"ROW-{i + 1:05d}" for i in range(len(df))])
    df["customerID"] = df["customerID"].astype(str)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    probs = model.predict_proba(preprocess(df))[:, 1]
    df["ChurnProbability"] = probs
    df["Prediction"] = ["Churn" if p >= CHURN_THRESHOLD else "Stay" for p in probs]
    df["RiskBand"] = [risk_band(p)[0] for p in probs]
    return df


def prediction_drivers(customer_row: pd.DataFrame, top: int = 6) -> pd.DataFrame:
    """Coefficient × value for one customer — what pushes their risk up or down."""
    x = preprocess(customer_row)
    contributions = pd.DataFrame({
        "Variable": feature_names,
        "Value": x.iloc[0].values.astype(float),
        "Coefficient": model.coef_[0],
    })
    contributions["Effect"] = contributions["Value"] * contributions["Coefficient"]
    contributions = contributions[contributions["Value"] != 0]
    contributions = contributions.reindex(
        contributions["Effect"].abs().sort_values(ascending=False).index
    ).head(top)
    contributions["Direction"] = contributions["Effect"].apply(
        lambda v: "Increases risk" if v > 0 else "Reduces risk"
    )
    return contributions[["Variable", "Direction", "Effect"]].round(3).reset_index(drop=True)


def show_drivers(customer_row: pd.DataFrame) -> None:
    st.subheader("What is driving this prediction")
    st.dataframe(prediction_drivers(customer_row), width="stretch", hide_index=True)
    st.caption(
        "Effect = model coefficient × the customer's (scaled) value. These show "
        "association, not causation — they indicate which characteristics the "
        "model weighted most for this customer."
    )


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def reset_monitor() -> None:
    st.session_state.mon_running = False
    st.session_state.mon_pos = 0
    st.session_state.mon_history = []
    st.session_state.mon_alerts = []
    st.session_state.mon_last_batch = None


for key, default in [("data", None), ("data_key", None), ("data_name", None)]:
    st.session_state.setdefault(key, default)
if "mon_running" not in st.session_state:
    reset_monitor()


def load_dataset(raw: pd.DataFrame, key: str, name: str) -> None:
    """Validate and score a new file ONCE, then share it with every tab."""
    missing = [c for c in RAW_PREDICTORS if c not in raw.columns]
    if missing:
        st.sidebar.error(f"Missing required columns: {', '.join(missing)}")
        return
    st.session_state.data = score_customers(raw)
    st.session_state.data_key = key
    st.session_state.data_name = name
    reset_monitor()
    st.rerun()  # redraw sidebar + tabs with the freshly scored data


# ----------------------------------------------------------------------------
# Sidebar — data upload (scored once) + live monitor controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Customer data")
    uploaded = st.file_uploader("Upload a customer CSV", type="csv")

    if uploaded is not None:
        # Only rescore when a genuinely new file arrives. Streamlit reruns the
        # whole script on every click; without this check thousands of rows
        # would be rescored on each interaction.
        key = f"{uploaded.name}:{uploaded.size}"
        if key != st.session_state.data_key:
            with st.spinner("Scoring customers..."):
                load_dataset(pd.read_csv(uploaded), key, uploaded.name)
    elif st.session_state.data is None:
        if st.button("Use demo file (sample_customers.csv)", width="stretch"):
            try:
                load_dataset(pd.read_csv(DEMO_FILE), f"demo:{DEMO_FILE}", DEMO_FILE)
            except FileNotFoundError:
                st.error(f"{DEMO_FILE} not found next to app.py.")

    data = st.session_state.data
    if data is not None:
        st.success(f"**{st.session_state.data_name}** — {len(data):,} customers scored")
    else:
        st.info("Upload a CSV with the project dataset's columns to enable the "
                "Overview, Customer search and Live monitor tabs.")

    st.divider()
    st.header("Live monitor (simulation)")

    def _start():
        if st.session_state.data is not None and \
                st.session_state.mon_pos >= len(st.session_state.data):
            reset_monitor()
        st.session_state.mon_running = True

    def _pause():
        st.session_state.mon_running = False

    b1, b2, b3 = st.columns(3)
    b1.button("Start", on_click=_start, disabled=data is None, width="stretch",
              type="primary")
    b2.button("Pause", on_click=_pause, disabled=data is None, width="stretch")
    b3.button("Reset", on_click=reset_monitor, disabled=data is None, width="stretch")

    batch_size = st.slider("Customers per batch", 1, 50, 10, key="mon_batch")
    interval = st.select_slider(
        "Seconds between batches", options=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
        value=1.0, key="mon_interval",
    )
    if st.session_state.mon_running:
        st.caption("🟢 Feed running")
    elif st.session_state.mon_pos:
        st.caption("⏸ Feed paused")


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

tab_overview, tab_search, tab_live, tab_single = st.tabs(
    ["Overview", "Customer search", "Live monitor", "Single customer"]
)


def need_data() -> bool:
    if st.session_state.data is None:
        st.info("Upload a customer CSV in the sidebar (or use the demo file) to "
                "see this tab.")
        return True
    return False


def band_chart(df: pd.DataFrame, title: str):
    counts = (df["RiskBand"].value_counts()
              .reindex(BAND_ORDER, fill_value=0)
              .rename_axis("Risk band").reset_index(name="Customers"))
    return alt.Chart(counts, title=title).mark_bar().encode(
        x=alt.X("Risk band:N", sort=BAND_ORDER, title=None,
              axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Customers:Q"),
        color=alt.Color("Risk band:N", scale=alt.Scale(domain=BAND_ORDER,
                                                       range=BAND_COLOURS),
                        legend=None),
        tooltip=["Risk band", "Customers"],
    ).properties(height=280)


# ----------------------------------------------------------------------------
# TAB 1 — Overview
# ----------------------------------------------------------------------------
with tab_overview:
    if not need_data():
        df = st.session_state.data
        probs = df["ChurnProbability"]
        n_churn = int((probs >= CHURN_THRESHOLD).sum())

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total customers", f"{len(df):,}")
        m2.metric("Predicted to churn", f"{n_churn:,}")
        m3.metric("Predicted churn %", f"{n_churn / len(df):.1%}")
        m4.metric("High risk (≥70%)", f"{int((probs >= HIGH_RISK).sum()):,}")
        m5.metric("Average risk", f"{probs.mean():.1%}")

        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(band_chart(df, "Customers by risk band"), width="stretch")
        with c2:
            by_contract = (df.groupby("Contract")["ChurnProbability"]
                           .apply(lambda p: (p >= CHURN_THRESHOLD).mean() * 100)
                           .reset_index(name="Predicted churn %"))
            chart = alt.Chart(by_contract,
                              title="Predicted churn rate by contract type").mark_bar(
                color="#C0392B").encode(
                x=alt.X("Contract:N", sort=["Month-to-month", "One year", "Two year"],
                        title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Predicted churn %:Q", scale=alt.Scale(domain=[0, 100])),
                tooltip=["Contract", alt.Tooltip("Predicted churn %:Q", format=".1f")],
            ).properties(height=280)
            st.altair_chart(chart, width="stretch")

        hist = alt.Chart(df[["ChurnProbability"]],
                         title="Distribution of churn probability").mark_bar(
            color="#34495E").encode(
            x=alt.X("ChurnProbability:Q", bin=alt.Bin(step=0.05, extent=[0, 1]),
                    title="Churn probability", axis=alt.Axis(format="%")),
            y=alt.Y("count():Q", title="Customers"),
        ).properties(height=240)
        rule = alt.Chart(pd.DataFrame({"x": [CHURN_THRESHOLD]})).mark_rule(
            color="#C0392B", strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart(hist + rule, width="stretch")
        st.caption("Dashed line = 50% decision threshold.")

        ranked = (df[["customerID", "tenure", "Contract", "MonthlyCharges",
                      "ChurnProbability", "Prediction", "RiskBand"]]
                  .sort_values("ChurnProbability", ascending=False)
                  .reset_index(drop=True))
        ranked["ChurnProbability"] = ranked["ChurnProbability"].round(4)

        st.subheader("20 highest-risk customers")
        st.dataframe(
            ranked.head(20), width="stretch", hide_index=True,
            column_config={"ChurnProbability": st.column_config.ProgressColumn(
                "Churn probability", min_value=0.0, max_value=1.0, format="percent")},
        )
        st.download_button(
            "Download all ranked predictions (CSV)",
            data=ranked.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions_ranked.csv",
            mime="text/csv",
            type="primary",
        )
        st.caption(
            "Sorted highest risk first, so the retention team can work down the "
            "list until its budget or capacity is reached."
        )


# ----------------------------------------------------------------------------
# TAB 2 — Customer search
# ----------------------------------------------------------------------------
with tab_search:
    if not need_data():
        df = st.session_state.data
        query = st.text_input("Search by customer ID (full or partial)",
                              placeholder="e.g. 7590-VHVEG or 7590",
                              key="search_query").strip()

        if query:
            matches = df[df["customerID"].str.contains(query, case=False,
                                                       regex=False, na=False)]
            if matches.empty:
                st.warning(f"No customer ID contains “{query}”.")
            else:
                if len(matches) > 1:
                    st.caption(f"{len(matches):,} matching customers")
                    options = (matches.sort_values("ChurnProbability", ascending=False)
                               ["customerID"].tolist())
                    chosen = st.selectbox(
                        "Select a customer", options,
                        format_func=lambda cid: (
                            f"{cid}  —  "
                            f"{df.loc[df.customerID == cid, 'ChurnProbability'].iloc[0]:.1%}"),
                    )
                else:
                    chosen = matches["customerID"].iloc[0]

                cust = df[df["customerID"] == chosen].head(1)
                row = cust.iloc[0]
                p = float(row["ChurnProbability"])

                st.divider()
                st.subheader(f"Customer {chosen}")
                r1, r2 = st.columns([1, 2])
                with r1:
                    st.metric("Churn probability", f"{p:.1%}")
                    band_badge(p)
                with r2:
                    st.progress(p)
                    if p >= CHURN_THRESHOLD:
                        st.warning("Predicted to churn — recommend contacting this "
                                   "customer with a retention offer.")
                    else:
                        st.success("Predicted to stay — no immediate action required.")

                d1, d2, d3 = st.columns(3)
                with d1:
                    st.markdown("**Account**")
                    st.write({
                        "Gender": row["gender"],
                        "Senior citizen": str(row["SeniorCitizen"]).replace("1", "Yes").replace("0", "No"),
                        "Partner": row["Partner"],
                        "Dependents": row["Dependents"],
                        "Tenure (months)": int(row["tenure"]),
                        "Contract": row["Contract"],
                    })
                with d2:
                    st.markdown("**Services**")
                    st.write({c: row[c] for c in [
                        "PhoneService", "MultipleLines", "InternetService",
                        *INTERNET_ADDONS]})
                with d3:
                    st.markdown("**Billing**")
                    st.write({
                        "Paperless billing": row["PaperlessBilling"],
                        "Payment method": row["PaymentMethod"],
                        "Monthly charges": float(row["MonthlyCharges"]),
                        "Total charges": float(row["TotalCharges"]),
                    })

                st.divider()
                show_drivers(cust)


# ----------------------------------------------------------------------------
# TAB 3 — Live monitor (SIMULATED)
# ----------------------------------------------------------------------------
with tab_live:
    st.warning(
        "**Simulated live feed.** Customers from the uploaded CSV are replayed in "
        "batches on a timer to demonstrate how the model could monitor incoming "
        "customers. This is not connected to a real data source.",
        icon="🧪",
    )
    if not need_data():
        df = st.session_state.data
        pos = st.session_state.mon_pos
        n = len(df)
        arrived = df.iloc[:pos]

        st.progress(pos / n, text=f"{pos:,} of {n:,} customers arrived")

        if pos == 0:
            st.info("Press **Start** in the sidebar to begin the simulated feed.")
        else:
            a_probs = arrived["ChurnProbability"]
            a_churn = int((a_probs >= CHURN_THRESHOLD).sum())
            a_high = int((a_probs >= HIGH_RISK).sum())
            history = st.session_state.mon_history
            prev_rate = history[-2]["Predicted churn %"] if len(history) > 1 else None
            rate = a_churn / pos * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Customers arrived", f"{pos:,}")
            m2.metric("Predicted churn %", f"{rate:.1f}%",
                      None if prev_rate is None else f"{rate - prev_rate:+.1f} pts",
                      delta_color="inverse")
            m3.metric("Predicted to churn", f"{a_churn:,}")
            m4.metric("High-risk alerts (≥70%)", f"{a_high:,}")

            # High-risk alerts from the most recent batch
            last = st.session_state.mon_last_batch
            if last is not None:
                new_alerts = last[last["ChurnProbability"] >= HIGH_RISK]
                if len(new_alerts):
                    for _, a in new_alerts.head(5).iterrows():
                        st.error(
                            f"**HIGH-RISK ALERT** — customer **{a['customerID']}** "
                            f"arrived with churn probability **{a['ChurnProbability']:.1%}** "
                            f"({a['Contract']}, tenure {int(a['tenure'])} months, "
                            f"{a['MonthlyCharges']:.2f}/month)",
                            icon="🚨",
                        )
                    if len(new_alerts) > 5:
                        st.error(f"…and {len(new_alerts) - 5} more high-risk customers "
                                 f"in this batch.", icon="🚨")
                else:
                    st.success("No high-risk customers in the latest batch.", icon="✅")

            hist_df = pd.DataFrame(history)
            line = alt.Chart(hist_df, title="Running predicted churn rate").mark_line(
                point=True, color="#C0392B").encode(
                x=alt.X("Customers arrived:Q"),
                y=alt.Y("Predicted churn %:Q", scale=alt.Scale(domain=[0, 100])),
                tooltip=["Customers arrived", alt.Tooltip("Predicted churn %:Q",
                                                          format=".1f")],
            ).properties(height=280)

            c1, c2 = st.columns([3, 2])
            with c1:
                st.altair_chart(line, width="stretch")
            with c2:
                st.altair_chart(band_chart(arrived, "Arrived customers by risk band"),
                                width="stretch")

            if st.session_state.mon_alerts:
                st.subheader(f"Alert log ({len(st.session_state.mon_alerts):,})")
                log = pd.DataFrame(st.session_state.mon_alerts).iloc[::-1]
                st.dataframe(
                    log, width="stretch", hide_index=True, height=250,
                    column_config={"ChurnProbability": st.column_config.NumberColumn(
                        "Churn probability", format="percent")},
                )

            if pos >= n:
                st.success("Feed complete — every customer in the file has arrived. "
                           "Press **Reset** then **Start** to replay.")


# ----------------------------------------------------------------------------
# TAB 4 — Single customer (hand-entered)
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

        probability = float(model.predict_proba(preprocess(customer))[0, 1])

        st.divider()
        r1, r2 = st.columns([1, 2])

        with r1:
            st.metric("Churn probability", f"{probability:.1%}")
            band_badge(probability)

        with r2:
            st.progress(probability)
            if probability >= CHURN_THRESHOLD:
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

        st.divider()
        show_drivers(customer)


st.divider()
st.caption(
    "SUZA Data Science Practical Training · Group Four · "
    "Predicting Customer Churn in a Telecommunications Company"
)


# ----------------------------------------------------------------------------
# Live monitor timer — MUST stay at the very bottom of the script.
# Everything above has already rendered; now wait, let the next batch
# "arrive", and rerun so every tab redraws with the new numbers.
# ----------------------------------------------------------------------------
if st.session_state.mon_running and st.session_state.data is not None:
    df = st.session_state.data
    if st.session_state.mon_pos >= len(df):
        st.session_state.mon_running = False
    else:
        time.sleep(float(st.session_state.mon_interval))

        start = st.session_state.mon_pos
        end = min(start + int(st.session_state.mon_batch), len(df))
        batch = df.iloc[start:end]
        st.session_state.mon_pos = end
        st.session_state.mon_last_batch = batch

        arrived_probs = df["ChurnProbability"].iloc[:end]
        st.session_state.mon_history.append({
            "Customers arrived": end,
            "Predicted churn %": float(round(
                (arrived_probs >= CHURN_THRESHOLD).mean() * 100, 2)),
        })

        high = batch[batch["ChurnProbability"] >= HIGH_RISK]
        batch_no = len(st.session_state.mon_history)
        for _, r in high.iterrows():
            st.session_state.mon_alerts.append({
                "Batch": batch_no,
                "customerID": r["customerID"],
                "ChurnProbability": round(float(r["ChurnProbability"]), 4),
                "Contract": r["Contract"],
                "tenure": int(r["tenure"]),
                "MonthlyCharges": float(r["MonthlyCharges"]),
            })
        if len(high):
            st.toast(f"{len(high)} high-risk customer(s) just arrived", icon="🚨")

        if end >= len(df):
            st.session_state.mon_running = False
        st.rerun()
