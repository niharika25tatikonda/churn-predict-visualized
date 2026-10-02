"""
Generates a realistic synthetic Telecom Customer Churn dataset (5,000 records).

Why synthetic? It lets the project run offline with zero downloads. The data mimics
the structure of the public IBM/Kaggle "Telco Customer Churn" dataset, and churn is
driven by realistic factors (contract type, tenure, price, support calls, etc.).
Deliberate "dirty" data (missing values, duplicates, inconsistent text) is injected
so that the preprocessing / wrangling step has real work to do.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000

def build():
    gender = rng.choice(["Male", "Female"], N)
    senior = rng.choice([0, 1], N, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], N, p=[0.48, 0.52])
    dependents = np.where(partner == "Yes", rng.choice(["Yes", "No"], N, p=[0.45, 0.55]),
                          rng.choice(["Yes", "No"], N, p=[0.12, 0.88]))
    contract = rng.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24])
    tenure = np.where(contract == "Month-to-month", rng.gamma(2.0, 9, N),
              np.where(contract == "One year", rng.gamma(4.0, 9, N), rng.gamma(6.0, 9, N)))
    tenure = np.clip(tenure.round(), 1, 72).astype(int)

    internet = rng.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])
    phone = rng.choice(["Yes", "No"], N, p=[0.9, 0.1])
    multi_lines = np.where(phone == "Yes", rng.choice(["Yes", "No"], N), "No phone service")

    def addon(p):
        return np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], N, p=[p, 1 - p]))
    security, backup, protection = addon(0.30), addon(0.35), addon(0.34)
    tech_support, tv, movies = addon(0.29), addon(0.38), addon(0.39)

    paperless = rng.choice(["Yes", "No"], N, p=[0.6, 0.4])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
                         N, p=[0.34, 0.23, 0.22, 0.21])

    monthly = 18 + 22 * (phone == "Yes") + np.select(
        [internet == "DSL", internet == "Fiber optic"], [18, 42], 0)
    for a in (security, backup, protection, tech_support, tv, movies):
        monthly = monthly + 5.5 * (a == "Yes")
    monthly = monthly + 8 * (multi_lines == "Yes") + rng.normal(0, 3, N)
    monthly = np.clip(monthly, 18.25, 119).round(2)
    total = (monthly * tenure * rng.normal(1.0, 0.03, N)).round(2)

    support_calls = rng.poisson(np.where(internet == "Fiber optic", 1.9, 1.1), N)
    data_gb = np.where(internet == "No", 0,
                       rng.gamma(3, 9, N) * np.where(internet == "Fiber optic", 1.6, 1.0)).round(1)

    # churn probability (logit model)
    z = (-1.9
         + 1.45 * (contract == "Month-to-month") - 1.1 * (contract == "Two year")
         - 0.035 * tenure
         + 0.018 * (monthly - 65)
         + 0.55 * (internet == "Fiber optic")
         + 0.40 * (payment == "Electronic check")
         + 0.30 * senior
         + 0.28 * support_calls
         + 0.25 * (paperless == "Yes")
         - 0.40 * (tech_support == "Yes") - 0.35 * (security == "Yes")
         - 0.25 * (partner == "Yes") - 0.20 * (dependents == "Yes")
         + rng.normal(0, 0.6, N))
    churn = (rng.random(N) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(1, N + 1)],
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multi_lines,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": tech_support, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
        "CustomerSupportCalls": support_calls, "AvgMonthlyDataGB": data_gb,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })

    # ---- inject realistic dirtiness ----
    df["TotalCharges"] = df["TotalCharges"].astype(object)
    idx = rng.choice(N, 40, replace=False)
    df.loc[idx, "TotalCharges"] = " "                       # blank strings (as in the real Telco data)
    df.loc[rng.choice(N, 60, replace=False), "MonthlyCharges"] = np.nan
    df.loc[rng.choice(N, 45, replace=False), "AvgMonthlyDataGB"] = np.nan
    df.loc[rng.choice(N, 30, replace=False), "Contract"] = np.nan
    df.loc[rng.choice(N, 80, replace=False), "gender"] = rng.choice(["male", "FEMALE", " Male", "female "], 80)
    df.loc[rng.choice(N, 12, replace=False), "MonthlyCharges"] = 450.0   # outliers / entry errors
    df = pd.concat([df, df.sample(35, random_state=1)], ignore_index=True)  # duplicates
    return df.sample(frac=1, random_state=7).reset_index(drop=True)

if __name__ == "__main__":
    df = build()
    df.to_csv("data/telecom_churn_raw.csv", index=False)
    print(df.shape, "| churn rate:", (df.Churn == "Yes").mean().round(3))
