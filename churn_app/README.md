# Customer Churn Prediction — Streamlit Dashboard

SUZA Data Science Practical Training · Group Four

## Files required in this folder

| File | Where it comes from |
|---|---|
| `app.py` | this application |
| `churn_model.pkl` | Stage 4, Task 7 |
| `churn_scaler.pkl` | Stage 4, Task 7 |
| `churn_features.pkl` | Stage 4, Task 7 |
| `sample_customers.csv` | 200 test customers (optional — used by the "Use demo file" button) |

## Setup (once)

```bash
pip install -r requirements.txt
```

Requires Streamlit 1.49 or newer (the app uses `width="stretch"`).

## Run

```bash
streamlit run app.py
```

The app opens automatically at http://localhost:8501

## How it works

Upload a customer CSV in the **sidebar** (or click **Use demo file**). The file is
scored **once** and shared by every tab; it is only rescored when a new file is
uploaded. The CSV may be the raw dataset or the cleaned one — the app applies the
cleaning rules itself. `customerID` is optional but needed for searching.

**Overview** — total customers, predicted churners (count and %), high-risk
count (≥70%), average risk; charts of customers by risk band, predicted churn rate
by contract type and the churn-probability distribution; the 20 highest-risk
customers and a CSV download of all ranked predictions.

**Customer search** — find a customer by full or partial `customerID`; shows
their churn probability, risk band, account/service/billing details and the
characteristics driving the prediction (model coefficient × value).

**Live monitor (simulation)** — replays the uploaded CSV in batches on a timer
to simulate customers arriving. Churn %, counts, a running-rate line chart and a
risk-band chart update as they arrive; any customer at ≥70% raises an alert.
Start / Pause / Reset, batch size and speed are in the sidebar.
*This is a simulation, not a connection to a live data source.*

**Single customer** — enter one customer's details by hand and score them.
