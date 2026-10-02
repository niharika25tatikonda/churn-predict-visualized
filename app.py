"""
Telecom Customer Churn Analytics Dashboard
Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models import FEATURES, predict_one, train_and_compare
from src.preprocessing import get_data, quality_report

st.set_page_config(page_title="Telecom Churn Analytics", page_icon="📡", layout="wide")

COLORS = {"No": "#2a9d8f", "Yes": "#e76f51"}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetric"] {background: rgba(128,128,128,0.08); padding: 14px 16px;
                             border-radius: 12px; border: 1px solid rgba(128,128,128,0.2);}
.insight {background: rgba(231,111,81,0.10); border-left: 5px solid #e76f51;
          padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;}
.good {background: rgba(42,157,143,0.12); border-left: 5px solid #2a9d8f;
       padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ cached data / models
@st.cache_data(show_spinner="Loading and cleaning data...")
def load():
    return get_data()


@st.cache_resource(show_spinner="Training models (takes ~20 seconds the first time)...")
def train(df):
    return train_and_compare(df)


raw, df, log = load()
bundle = train(df)
results = bundle["results"]
best_model = results["ROC-AUC"].idxmax()


def pct(x):
    return f"{x * 100:.1f}%"


def churn_rate_by(col, data=None):
    d = df if data is None else data
    g = d.groupby(col)["ChurnFlag"].agg(["mean", "count"]).reset_index()
    g.columns = [col, "Churn Rate", "Customers"]
    return g


# ------------------------------------------------------------------ sidebar
st.sidebar.title("📡 Churn Analytics")
page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard", "🧹 Data & Wrangling", "🔍 Exploratory Analysis",
    "🤖 Model Comparison", "🎯 Predict Churn"])
st.sidebar.markdown("---")
st.sidebar.caption(f"**Dataset:** {len(df):,} customers · {df.shape[1]} columns")
st.sidebar.caption("**Goal:** predict which customers will leave (churn) so the company can retain them.")
st.sidebar.caption("**Data note:** synthetic data modelled on the public Telco churn dataset.")


# ================================================================== 1. DASHBOARD
if page == "🏠 Dashboard":
    st.title("📡 Telecom Customer Churn Dashboard")
    st.write("**Churn** means a customer cancels their service. Keeping an existing customer is much "
             "cheaper than finding a new one, so knowing *who is likely to leave and why* is valuable.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", pct(df.ChurnFlag.mean()))
    c3.metric("Avg monthly bill", f"${df.MonthlyCharges.mean():.2f}")
    c4.metric("Avg tenure", f"{df.tenure.mean():.0f} months")
    lost = df.loc[df.ChurnFlag == 1, "MonthlyCharges"].sum()
    c5.metric("Monthly revenue at risk", f"${lost:,.0f}")

    left, right = st.columns([1, 1.4])
    with left:
        st.subheader("Who stays vs. who leaves")
        counts = df.Churn.value_counts().reset_index()
        counts.columns = ["Churn", "Customers"]
        fig = px.pie(counts, names="Churn", values="Customers", hole=0.55,
                     color="Churn", color_discrete_map=COLORS)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=340)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Churn rate by contract type")
        g = churn_rate_by("Contract").sort_values("Churn Rate", ascending=False)
        fig = px.bar(g, x="Contract", y="Churn Rate", text=g["Churn Rate"].map(pct),
                     color="Churn Rate", color_continuous_scale="OrRd")
        fig.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False,
                          margin=dict(t=10, b=10), height=340)
        st.plotly_chart(fig, width="stretch")

    st.subheader("💡 Key insights from the data")
    mtm = df[df.Contract == "Month-to-month"].ChurnFlag.mean()
    two = df[df.Contract == "Two year"].ChurnFlag.mean()
    new = df[df.tenure <= 12].ChurnFlag.mean()
    old = df[df.tenure > 48].ChurnFlag.mean()
    fiber = df[df.InternetService == "Fiber optic"].ChurnFlag.mean()
    dsl = df[df.InternetService == "DSL"].ChurnFlag.mean()
    ts_y = df[df.TechSupport == "Yes"].ChurnFlag.mean()
    ts_n = df[df.TechSupport == "No"].ChurnFlag.mean()
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown(f'<div class="insight"><b>Contract matters most.</b> Month-to-month customers churn at '
                    f'<b>{pct(mtm)}</b> vs only <b>{pct(two)}</b> on two-year contracts.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight"><b>The first year is risky.</b> Customers in their first 12 months '
                    f'churn at <b>{pct(new)}</b>, compared to <b>{pct(old)}</b> after 4 years.</div>', unsafe_allow_html=True)
    with ic2:
        st.markdown(f'<div class="insight"><b>Fiber optic users leave more</b> ({pct(fiber)} vs {pct(dsl)} for DSL), '
                    f'likely due to higher prices.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="good"><b>Tech support helps retention.</b> Customers with it churn at '
                    f'{pct(ts_y)} vs {pct(ts_n)} without.</div>', unsafe_allow_html=True)

    st.subheader("🏆 Best prediction model")
    b = results.loc[best_model]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best model (by ROC-AUC)", best_model)
    m2.metric("ROC-AUC", f"{b['ROC-AUC']:.3f}")
    m3.metric("Recall (churners found)", pct(b["Recall"]))
    m4.metric("Accuracy", pct(b["Accuracy"]))
    st.caption("Head to **Model Comparison** for the full breakdown or **Predict Churn** to score a customer.")


# ================================================================== 2. DATA & WRANGLING
elif page == "🧹 Data & Wrangling":
    st.title("🧹 Data Preprocessing & Wrangling")
    st.write("Real-world data is messy. Below is the raw data's quality, every cleaning step applied, "
             "and the final dataset used for analysis and modelling.")

    t1, t2, t3, t4 = st.tabs(["1️⃣ Raw data", "2️⃣ Problems found", "3️⃣ Cleaning steps", "4️⃣ Clean data"])

    with t1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw rows", f"{raw.shape[0]:,}")
        c2.metric("Raw columns", raw.shape[1])
        c3.metric("Duplicate rows", int(raw.duplicated().sum()))
        st.dataframe(raw.head(100), width="stretch", height=350)
        st.caption("Showing the first 100 raw rows.")

    with t2:
        st.subheader("Missing values before vs. after cleaning")
        before = raw.isna().sum()
        before["TotalCharges"] += int((raw["TotalCharges"].astype(str).str.strip() == "").sum())
        cmp_df = pd.DataFrame({"Before cleaning": before, "After cleaning": df[raw.columns].isna().sum()})
        cmp_df = cmp_df[cmp_df["Before cleaning"] > 0].reset_index(names="Column")
        fig = px.bar(cmp_df.melt("Column", var_name="Stage", value_name="Missing values"),
                     x="Column", y="Missing values", color="Stage", barmode="group",
                     color_discrete_map={"Before cleaning": "#e76f51", "After cleaning": "#2a9d8f"})
        fig.update_layout(height=380, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")

        st.subheader("Other issues detected")
        gender_bad = (~raw.gender.isin(["Male", "Female"])).sum()
        st.markdown(f"- **{int(raw.duplicated().sum())}** duplicate rows\n"
                    f"- **{int(gender_bad)}** inconsistent entries in `gender` (e.g. `FEMALE `, `male`)\n"
                    f"- **{int((raw.MonthlyCharges > 125).sum())}** impossible `MonthlyCharges` values (450.0)\n"
                    f"- `TotalCharges` stored as text because of blank strings")
        st.subheader("Raw column report")
        st.dataframe(quality_report(raw), width="stretch", hide_index=True)

    with t3:
        st.dataframe(log, width="stretch", hide_index=True)
        st.info("Missing numbers were filled with the **median of similar customers** (not one global value) "
                "so the fill-in is realistic. Outliers were treated as entry errors rather than deleted, "
                "so no customers were lost.")

    with t4:
        c1, c2, c3 = st.columns(3)
        c1.metric("Clean rows", f"{df.shape[0]:,}")
        c2.metric("Clean columns", df.shape[1])
        c3.metric("Missing values left", int(df.isna().sum().sum()))
        st.dataframe(df.head(100), width="stretch", height=350)
        st.subheader("Statistical summary")
        st.dataframe(df.describe().T.round(2), width="stretch")
        st.download_button("⬇️ Download clean dataset (CSV)", df.to_csv(index=False).encode(),
                           "telecom_churn_clean.csv", "text/csv")


# ================================================================== 3. EDA
elif page == "🔍 Exploratory Analysis":
    st.title("🔍 Exploratory Data Analysis")
    st.write("Explore patterns in the data. Use the filters to focus on specific customer groups.")

    with st.expander("🎛️ Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        contracts = f1.multiselect("Contract", sorted(df.Contract.unique()), default=sorted(df.Contract.unique()))
        internet = f2.multiselect("Internet service", sorted(df.InternetService.unique()),
                                  default=sorted(df.InternetService.unique()))
        tmin, tmax = f3.slider("Tenure (months)", 1, 72, (1, 72))
    d = df[df.Contract.isin(contracts) & df.InternetService.isin(internet) & df.tenure.between(tmin, tmax)]
    if d.empty:
        st.warning("No customers match these filters.")
        st.stop()
    st.caption(f"Showing **{len(d):,}** customers · churn rate **{pct(d.ChurnFlag.mean())}**")

    tabs = st.tabs(["📊 Distributions", "🎯 Churn drivers", "🔗 Correlations", "💰 Charges & usage"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            num = st.selectbox("Numeric feature", ["tenure", "MonthlyCharges", "TotalCharges",
                                                   "AvgMonthlyDataGB", "CustomerSupportCalls"])
            fig = px.histogram(d, x=num, color="Churn", nbins=35, barmode="overlay", opacity=0.7,
                               color_discrete_map=COLORS, marginal="box")
            fig.update_layout(height=430, margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Overlapping histograms show how churners (orange) differ from loyal customers (green).")
        with c2:
            cat = st.selectbox("Categorical feature", ["Contract", "InternetService", "PaymentMethod",
                                                       "TechSupport", "OnlineSecurity", "PaperlessBilling",
                                                       "Partner", "Dependents", "gender", "TenureGroup"])
            g = d.groupby([cat, "Churn"]).size().reset_index(name="Customers")
            fig = px.bar(g, x=cat, y="Customers", color="Churn", barmode="group",
                         color_discrete_map=COLORS)
            fig.update_layout(height=430, margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")

    with tabs[1]:
        st.subheader("Churn rate across customer groups")
        dim = st.selectbox("Group customers by", ["Contract", "InternetService", "PaymentMethod", "TenureGroup",
                                                  "TechSupport", "OnlineSecurity", "SeniorCitizen",
                                                  "NumAddOns", "CustomerSupportCalls"], key="drv")
        g = churn_rate_by(dim, d)
        fig = px.bar(g, x=dim, y="Churn Rate", text=g["Churn Rate"].map(pct), color="Churn Rate",
                     color_continuous_scale="OrRd", hover_data=["Customers"])
        fig.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False, height=400, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")

        st.subheader("Heatmap: Contract × Internet service")
        pivot = d.pivot_table(index="Contract", columns="InternetService", values="ChurnFlag", aggfunc="mean")
        fig = px.imshow(pivot, text_auto=".0%", color_continuous_scale="OrRd", aspect="auto")
        fig.update_layout(height=320, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Darker = higher churn. Month-to-month fiber customers are the highest-risk segment.")

    with tabs[2]:
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "CustomerSupportCalls",
                    "AvgMonthlyDataGB", "NumAddOns", "SeniorCitizen", "ChurnFlag"]
        corr = d[num_cols].corr().round(2)
        c1, c2 = st.columns([1.3, 1])
        with c1:
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(height=520, margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown("**Correlation with churn**")
            cc = corr["ChurnFlag"].drop("ChurnFlag").sort_values()
            fig = px.bar(cc, orientation="h", color=cc, color_continuous_scale="RdBu_r", range_color=[-1, 1])
            fig.update_layout(height=420, showlegend=False, coloraxis_showscale=False,
                              xaxis_title="Correlation", yaxis_title="", margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Positive → more likely to churn. Negative → more loyal. "
                       "Tenure has a strong negative link: long-time customers rarely leave.")

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(d, x="Contract", y="MonthlyCharges", color="Churn", color_discrete_map=COLORS)
            fig.update_layout(height=400, margin=dict(t=30), title="Monthly bill by contract & churn")
            st.plotly_chart(fig, width="stretch")
        with c2:
            sample = d.sample(min(1500, len(d)), random_state=1)
            fig = px.scatter(sample, x="tenure", y="MonthlyCharges", color="Churn", opacity=0.6,
                             color_discrete_map=COLORS, hover_data=["Contract", "InternetService"])
            fig.update_layout(height=400, margin=dict(t=30), title="Tenure vs. monthly bill")
            st.plotly_chart(fig, width="stretch")
        st.caption("Churners cluster at **short tenure + high monthly bill**.")


# ================================================================== 4. MODEL COMPARISON
elif page == "🤖 Model Comparison":
    st.title("🤖 Machine Learning Model Comparison")
    st.write(f"Three models were trained on **{bundle['train_size']:,}** customers and tested on "
             f"**{bundle['test_size']:,}** unseen customers (80/20 split, stratified).")

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown(
            "- **Accuracy** – share of all predictions that were right.\n"
            "- **Precision** – of customers we flagged as 'will churn', how many really did.\n"
            "- **Recall** – of customers who really churned, how many we caught. *Most important here: "
            "missing a churner costs money.*\n"
            "- **F1-Score** – balance between precision and recall.\n"
            "- **ROC-AUC** – how well the model ranks churners above non-churners (0.5 = guessing, 1.0 = perfect).")

    st.subheader("📋 Scorecard")
    styled = results.style.format("{:.3f}").highlight_max(axis=0, color="#b7e4c7")
    st.dataframe(styled, width="stretch")
    st.success(f"🏆 **{best_model}** has the highest ROC-AUC ({results.loc[best_model, 'ROC-AUC']:.3f}). "
               "Differences between models are small, so the simplest model is also a strong choice.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Metric comparison")
        long = results.reset_index().melt("Model", var_name="Metric", value_name="Score")
        fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group",
                     color_discrete_sequence=["#264653", "#2a9d8f", "#e9c46a"])
        fig.update_layout(yaxis_range=[0, 1], height=400, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("ROC curves")
        fig = go.Figure()
        for (name, (fpr, tpr)), col in zip(bundle["curves"].items(), ["#264653", "#2a9d8f", "#e9c46a"]):
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {results.loc[name, 'ROC-AUC']:.3f})",
                            line=dict(color=col, width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                        line=dict(color="gray", dash="dash"))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                          height=400, margin=dict(t=10), legend=dict(x=0.35, y=0.05))
        st.plotly_chart(fig, width="stretch")

    st.subheader("🔁 Cross-validation stability (5-fold ROC-AUC)")
    cv_df = pd.DataFrame(bundle["cv"]).melt(var_name="Model", value_name="ROC-AUC")
    fig = px.box(cv_df, x="Model", y="ROC-AUC", points="all", color="Model",
                 color_discrete_sequence=["#264653", "#2a9d8f", "#e9c46a"])
    fig.update_layout(showlegend=False, height=340, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Tight boxes mean the model performs consistently on different slices of the data.")

    st.subheader("🔬 Look inside a model")
    pick = st.selectbox("Choose a model", list(results.index), index=list(results.index).index(best_model))
    c1, c2 = st.columns(2)
    with c1:
        cm = bundle["cms"][pick]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Predicted: Stay", "Predicted: Churn"], y=["Actual: Stay", "Actual: Churn"])
        fig.update_layout(title=f"Confusion matrix – {pick}", height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"✅ {tp} churners caught · ❌ {fn} churners missed · ⚠️ {fp} false alarms · "
                   f"✅ {tn} loyal customers correctly identified.")
    with c2:
        imp = bundle["importances"][pick].head(12)[::-1]
        label = "Coefficient size" if pick == "Logistic Regression" else "Importance"
        fig = px.bar(imp, orientation="h", color_discrete_sequence=["#e76f51"])
        fig.update_layout(title=f"Top 12 drivers of churn – {pick}", showlegend=False, height=380,
                          xaxis_title=label, yaxis_title="")
        st.plotly_chart(fig, width="stretch")

    st.subheader("📝 Conclusion")
    st.markdown(
        "- **Logistic Regression** is simple, explainable and catches the most churners (highest recall).\n"
        "- **Random Forest** and **Gradient Boosting** have fewer false alarms (higher precision/accuracy).\n"
        "- Across all models the top churn drivers are **contract type, tenure, and monthly charges**.\n"
        "- If missing a churner is costly, prefer high recall; if retention offers are expensive, prefer high precision.")


# ================================================================== 5. PREDICT
else:
    st.title("🎯 Predict Customer Churn")
    st.write("Enter a customer's details to estimate how likely they are to leave.")

    model_name = st.selectbox("Model to use", list(results.index), index=list(results.index).index(best_model))
    pipe = bundle["pipes"][model_name]
    thr = bundle["thresholds"][model_name]

    with st.form("predict"):
        st.markdown("**👤 Customer profile**")
        a, b, c, e = st.columns(4)
        gender = a.selectbox("Gender", ["Male", "Female"])
        senior = b.selectbox("Senior citizen", ["No", "Yes"])
        partner = c.selectbox("Has partner", ["No", "Yes"])
        dependents = e.selectbox("Has dependents", ["No", "Yes"])

        st.markdown("**📞 Services**")
        a, b, c = st.columns(3)
        phone = a.selectbox("Phone service", ["Yes", "No"])
        multi = b.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = c.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        a, b, c = st.columns(3)
        security = a.selectbox("Online security", ["No", "Yes"])
        backup = b.selectbox("Online backup", ["No", "Yes"])
        protect = c.selectbox("Device protection", ["No", "Yes"])
        a, b, c = st.columns(3)
        tech = a.selectbox("Tech support", ["No", "Yes"])
        tv = b.selectbox("Streaming TV", ["No", "Yes"])
        movies = c.selectbox("Streaming movies", ["No", "Yes"])

        st.markdown("**💳 Account**")
        a, b, c = st.columns(3)
        contract = a.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = b.selectbox("Paperless billing", ["Yes", "No"])
        payment = c.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        a, b, c, e = st.columns(4)
        tenure = a.slider("Tenure (months)", 1, 72, 6)
        monthly = b.number_input("Monthly charges ($)", 18.0, 120.0, 85.0, step=0.5)
        calls = c.slider("Support calls", 0, 10, 3)
        data_gb = e.number_input("Avg monthly data (GB)", 0.0, 200.0, 40.0, step=1.0)
        go_btn = st.form_submit_button("🔮 Predict", type="primary")

    if go_btn:
        if internet == "No":
            security = backup = protect = tech = tv = movies = "No internet service"
            data_gb = 0.0
        if phone == "No":
            multi = "No phone service"
        addons = [security, backup, protect, tech, tv, movies]
        row = {
            "gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": multi,
            "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protect, "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": round(monthly * tenure, 2),
            "CustomerSupportCalls": calls, "AvgMonthlyDataGB": data_gb,
            "NumAddOns": sum(a == "Yes" for a in addons), "ChargePerTenureMonth": monthly,
        }
        proba, flag = predict_one(pipe, row, thr)
        risk = "HIGH" if proba >= 0.6 else "MEDIUM" if proba >= 0.35 else "LOW"
        color = {"HIGH": "#e63946", "MEDIUM": "#f4a261", "LOW": "#2a9d8f"}[risk]

        c1, c2 = st.columns([1, 1.2])
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=proba * 100, number={"suffix": "%"},
                title={"text": "Churn probability"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                       "steps": [{"range": [0, 35], "color": "#d8f3dc"},
                                 {"range": [35, 60], "color": "#ffe8cc"},
                                 {"range": [60, 100], "color": "#ffd6d6"}]}))
            fig.update_layout(height=320, margin=dict(t=60, b=10))
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown(f"### Risk level: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
            st.write("**Prediction:** " + ("⚠️ likely to **churn**" if flag else "✅ likely to **stay**"))
            st.markdown("**Suggested actions**")
            tips = []
            if contract == "Month-to-month":
                tips.append("Offer a discount to move to a 1- or 2-year contract.")
            if tech == "No" and internet != "No":
                tips.append("Offer a free trial of Tech Support / Online Security.")
            if calls >= 3:
                tips.append("High support calls: proactively reach out to resolve open issues.")
            if monthly > 80:
                tips.append("High bill: review the plan for a cheaper bundle.")
            if payment == "Electronic check":
                tips.append("Encourage auto-pay (credit card / bank transfer).")
            if tenure <= 12:
                tips.append("New customer: send a welcome / onboarding offer.")
            for t in tips or ["Customer looks stable. Keep service quality high."]:
                st.markdown(f"- {t}")

        st.subheader("Compare with all models")
        rows = []
        for name, p in bundle["pipes"].items():
            pr, fl = predict_one(p, row, bundle["thresholds"][name])
            rows.append({"Model": name, "Churn probability": pr, "Verdict": "Churn" if fl else "Stay"})
        comp = pd.DataFrame(rows)
        fig = px.bar(comp, x="Model", y="Churn probability", color="Verdict", text=comp["Churn probability"].map(pct),
                     color_discrete_map={"Churn": "#e76f51", "Stay": "#2a9d8f"})
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], height=320, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
