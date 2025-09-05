# loan_app.py
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------- Load model safely ----------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_model.pkl")
if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Please run: python train_model.py (model file not found)")
    st.stop()

model, feature_names = joblib.load(MODEL_PATH)

# ---------- Page config ----------
st.set_page_config(page_title="Home Loan Eligibility Checker", layout="wide")

# ---------- Dark theme CSS ----------
st.markdown("""
<style>
    .main { background-color: #0F1117; color: #E1E1E1; }
    h1, h2, h3 { color: #7CD1F9; }
    .stButton>button {
        background: linear-gradient(90deg,#6C63FF,#00D4FF);
        color: white; border:none; padding:8px 16px; border-radius:10px; font-weight:700;
    }
    .stButton>button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🏦 Home Loan Eligibility Checker</h1>", unsafe_allow_html=True)
st.caption("Fill details on the left. See decision and live charts on the right.")
st.write("---")

# ---------- Layout ----------
left, right = st.columns([1, 1])

# ---------- Left: Inputs ----------
with left:
    st.subheader("📋 Applicant Profile")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("Self Employed", ["Yes", "No"])
    emp_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Government", "Other"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    city_tier = st.selectbox("City Tier", ["1", "2", "3"])

    st.subheader("💰 Financials")
    app_inc = st.number_input("Applicant Monthly Income (₹)", min_value=0, max_value=1000000, value=60000, step=1000)
    co_inc  = st.number_input("Coapplicant Monthly Income (₹)", min_value=0, max_value=1000000, value=10000, step=1000)
    existing_emi = st.number_input("Existing EMI (₹/month)", min_value=0, max_value=300000, value=5000, step=1000)
    savings = st.number_input("Savings / Liquid Funds (₹)", min_value=0, max_value=5000000, value=100000, step=10000)
    other_debts = st.number_input("Other Debts (₹)", min_value=0, max_value=5000000, value=0, step=10000)

    st.subheader("🏠 Loan Request")
    loan_amount = st.number_input("Loan Amount Needed (₹)", min_value=10000, max_value=100000000, value=1500000, step=10000)
    loan_term = st.slider("Loan Term (Months)", 6, 60, 24)
    credit_score = st.slider("Credit Score (300–900)", 300, 900, 720)

    submitted = st.button("Check Eligibility 🚀")

# Helper: monthly EMI (approx) using credit-score-based rate
def emi(p, annual_rate, months):
    r = annual_rate / 12.0
    if months < 1: months = 1
    if r <= 0: return p / months
    return p * (r * (1 + r)**months) / ((1 + r)**months - 1)

def score_to_rate(score):
    base = 0.085
    adj = (700 - min(score, 700)) * 0.00015
    return float(np.clip(base + adj, 0.085, 0.17))

annual_rate = score_to_rate(credit_score)
new_emi = emi(loan_amount, annual_rate, loan_term)
total_income = app_inc + co_inc
dti = (existing_emi + new_emi) / max(total_income, 1)

# ---------- Prepare a single-row DataFrame exactly like training ----------
row = {
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_emp,
    "Employment_Type": emp_type,
    "Property_Area": property_area,
    "City_Tier": city_tier,
    "ApplicantIncome": app_inc,
    "CoapplicantIncome": co_inc,
    "Existing_EMI": existing_emi,
    "Savings": savings,
    "Other_Debts": other_debts,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_Score": credit_score
}
input_df = pd.get_dummies(pd.DataFrame([row]), drop_first=True)
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# ---------- Right: Result + Live Visuals ----------
with right:
    st.subheader("📊 Decision & Insights")

    if submitted:
        proba = model.predict_proba(input_df)[0][1]
        pred = int(proba >= 0.5)

        if pred == 1:
            st.success(f"✅ Eligible  •  Confidence: {proba:.2f}")
        else:
            st.error(f"❌ Not Eligible  •  Confidence: {proba:.2f}")

        st.caption(f"Estimated EMI (based on credit score): ₹{int(new_emi):,} / month  •  DTI: {dti:.2f}")

        st.write("---")

        # 1) Donut: Eligibility probability
        fig_pie = px.pie(
            values=[proba, 1 - proba],
            names=["Eligible", "Not Eligible"],
            hole=0.5,
            title="Eligibility Probability",
            color_discrete_sequence=["#00D4FF", "#FF4D4F"]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # 2) Gauge: Credit score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            title={'text': "Credit Score (300–900)"},
            gauge={'axis': {'range': [300, 900]}}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # 3) Line: EMI vs Term (1–60 months)
        months = np.arange(6, 61)
        emis = [emi(loan_amount, annual_rate, m) for m in months]
        fig_line = px.line(x=months, y=emis, labels={"x": "Months", "y": "EMI (₹)"},
                           title="EMI vs Loan Term")
        st.plotly_chart(fig_line, use_container_width=True)

        # 4) Bars: Income vs Loan vs Existing EMI
        fig_bar = px.bar(
            x=["Applicant Income", "Coapplicant Income", "Existing EMI", "Estimated New EMI", "Loan Amount"],
            y=[app_inc, co_inc, existing_emi, new_emi, loan_amount],
            title="Income & Obligations",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # 5) Donut: Monthly budget split (approx)
        monthly_budget = {
            "New EMI": new_emi,
            "Existing EMI": existing_emi,
            "Other Debts (est. / month)": other_debts/60,  # rough spread
            "Free Income": max(total_income - (new_emi + existing_emi + other_debts/60), 0)
        }
        fig_budget = px.pie(
            names=list(monthly_budget.keys()),
            values=list(monthly_budget.values()),
            hole=0.45,
            title="Monthly Budget Split"
        )
        st.plotly_chart(fig_budget, use_container_width=True)
    else:
        st.info("👈 Enter details on the left and click **Check Eligibility** to see results.")
