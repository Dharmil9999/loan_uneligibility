# generate_dataset.py
import numpy as np
import pandas as pd

np.random.seed(42)
N = 6000

# --- Categorical fields ---
Gender = np.random.choice(["Male", "Female"], N, p=[0.6, 0.4])
Married = np.random.choice(["Yes", "No"], N, p=[0.68, 0.32])
Dependents = np.random.choice(["0", "1", "2", "3+"], N, p=[0.45, 0.25, 0.2, 0.1])
Education = np.random.choice(["Graduate", "Not Graduate"], N, p=[0.7, 0.3])
Self_Employed = np.random.choice(["Yes", "No"], N, p=[0.18, 0.82])
Employment_Type = np.random.choice(["Salaried", "Self-Employed", "Government", "Other"],
                                   N, p=[0.58, 0.22, 0.15, 0.05])
Property_Area = np.random.choice(["Urban", "Semiurban", "Rural"], N, p=[0.45, 0.35, 0.20])
City_Tier = np.random.choice(["1", "2", "3"], N, p=[0.35, 0.4, 0.25])

# --- Numeric fields (₹ per month unless noted) ---
ApplicantIncome = np.random.lognormal(mean=np.log(50000), sigma=0.55, size=N).astype(int)  # ~20k–200k
CoapplicantIncome_flag = (np.random.rand(N) > 0.55)
CoapplicantIncome = (np.random.lognormal(mean=np.log(20000), sigma=0.6, size=N) * CoapplicantIncome_flag).astype(int)

Existing_EMI = (np.random.choice([0, 1], N, p=[0.55, 0.45]) * 
                np.random.randint(0, 30000, size=N)).astype(int)

Savings = np.random.randint(0, 800000, N)  # bank balance / liquid savings
Other_Debts = (np.random.choice([0, 1], N, p=[0.6, 0.4]) * 
               np.random.randint(0, 600000, size=N)).astype(int)

LoanAmount = np.random.lognormal(mean=np.log(900000), sigma=0.7, size=N).clip(100000, 10000000).astype(int)  # ₹1L–₹1Cr
Loan_Amount_Term = np.random.randint(6, 61, size=N)  # months 6–60

Credit_Score = np.clip(np.random.normal(700, 75, size=N), 300, 900).astype(int)

# --- Helper: EMI formula for a given principal, annual interest, months ---
def emi(p, annual_rate, months):
    r = annual_rate / 12.0
    if months < 1: months = 1
    if r <= 0: return p / months
    return p * (r * (1 + r)**months) / ((1 + r)**months - 1)

# Map credit score → annual interest rate (lower score → higher rate)
base = 0.085  # 8.5%
rate_adj = (700 - np.minimum(Credit_Score, 700)) * 0.00015
Annual_Rate = np.clip(base + rate_adj, 0.085, 0.17)

New_EMI = np.array([emi(p, r, n) for p, r, n in zip(LoanAmount, Annual_Rate, Loan_Amount_Term)]).astype(int)

Total_Income = ApplicantIncome + CoapplicantIncome
DTI = (Existing_EMI + New_EMI) / np.maximum(Total_Income, 1)  # debt-to-income ratio

# Approval logic (bank-like rules)
score_ok = (
    ((Employment_Type == "Salaried") & (Credit_Score >= 650)) |
    ((Employment_Type == "Government") & (Credit_Score >= 640)) |
    ((Employment_Type == "Self-Employed") & (Credit_Score >= 690)) |
    ((Employment_Type == "Other") & (Credit_Score >= 680))
)

dti_limit = np.where(City_Tier == "1", 0.45, np.where(City_Tier == "2", 0.42, 0.40))
dti_ok = DTI <= dti_limit

amt_to_income = LoanAmount / np.maximum(Total_Income, 1)
mult_limit = np.where(Education == "Graduate", 36, 32)  # graduates get a bit higher multiple
amt_ok = amt_to_income <= mult_limit

savings_buffer_ok = Savings >= (0.05 * LoanAmount)  # at least 5% in liquid buffer

# Mild property/urban boost
urban_boost = (Property_Area == "Urban") & (Credit_Score >= 700) & (DTI <= (dti_limit + 0.02))

approved = score_ok & dti_ok & amt_ok & savings_buffer_ok
approved = approved | urban_boost

# add 7% label noise
flip = np.random.rand(N) < 0.07
approved = np.where(flip, ~approved, approved)

Loan_Status = np.where(approved, 1, 0)  # 1=Eligible, 0=Not Eligible

df = pd.DataFrame({
    "Gender": Gender,
    "Married": Married,
    "Dependents": Dependents,
    "Education": Education,
    "Self_Employed": Self_Employed,
    "Employment_Type": Employment_Type,
    "Property_Area": Property_Area,
    "City_Tier": City_Tier,
    "ApplicantIncome": ApplicantIncome,
    "CoapplicantIncome": CoapplicantIncome,
    "Existing_EMI": Existing_EMI,
    "Savings": Savings,
    "Other_Debts": Other_Debts,
    "LoanAmount": LoanAmount,
    "Loan_Amount_Term": Loan_Amount_Term,
    "Credit_Score": Credit_Score,
    "Loan_Status": Loan_Status
})

df.to_csv("loan_dataset.csv", index=False)
print("✅ Generated loan_dataset.csv with", len(df), "rows and columns:", list(df.columns))
