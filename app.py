import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bank Segments & Propensity", page_icon="📈")

# --- Load saved models (from Step 1 / notebook) ---
@st.cache_resource
def load_models():
    cluster_preproc = joblib.load("cluster_preprocessor.pkl")
    kmeans = joblib.load("kmeans.pkl")
    prop_pipe = joblib.load("propensity_pipeline.pkl")
    return cluster_preproc, kmeans, prop_pipe

cluster_preproc, kmeans, prop_pipe = load_models()

# =========================================================
# Cluster name mapping + rich descriptions (edit if needed)
# NOTE: Keys 0..4 are KMeans labels. Verify once after training.
# =========================================================
SEGMENT_INFO = {
    0: {
        "name": "Engaged Professionals",
        "md": """
## Segment 1: **Engaged Professionals**
- **Avg age:** ~41 years  
- **Avg balance:** €1,574  
- Personal loans: 13.2%  
- Housing loans: 62.1%  
- Default history: 0%  
- Previously contacted: 100%  
- Avg previous contacts: 3.18  
- Avg campaign calls: 2.05  
- Top job: *management* (22%)  
- Top marital: *married* (58%)  
- Top contact: *cellular* (91%)

**👉 Interpretation:** Financially stable and responsive mid-aged customers with housing loans.  
Likely engaged with the bank already through previous campaigns, good balances, and low default risk.

**💡 Cross-sell ideas:**  
- Premium Credit Cards (travel perks, cashback)  
- Investment Products (mutual funds, retirement plans)  
- Wealth Management Services (since they already trust the bank)
"""
    },
    1: {
        "name": "eAt-Risk Customrs",
        "md": """
## Segment 2: **At-Risk Customers**
- **Avg age:** ~39 years  
- **Avg balance:** -€138 (negative)  
- Personal loans: 36.9%  
- Housing loans: 53.4%  
- Default history: 100%  
- Previously contacted: 7%  
- Avg previous contacts: 0.27  
- Avg campaign calls: 3.15  
- Top job: *blue-collar* (25%)  
- Top marital: *married* (55%)  
- Top contact: *cellular* (61%)

**👉 Interpretation:** High-risk customers with low/negative balances, high loan rates, and default history. Very low engagement.

**💡 Cross-sell ideas:**  
- Debt Restructuring / Consolidation Loans  
- Micro-insurance products (low premium)  
- Financial literacy or credit counseling programs  
- ❌ Do **not** aggressively push new loans — focus on stabilization
"""
    },
    2: {
        "name": "High Loan Dependents",
        "md": """
## Segment 3: **High Loan Dependents**
- **Avg age:** ~41 years  
- **Avg balance:** €790  
- Personal loans: 100%  
- Housing loans: 58.3%  
- Default history: 0%  
- Previously contacted: 0.4%  
- Avg previous contacts: 0.01  
- Avg campaign calls: 2.95  
- Top job: *blue-collar* (23%)  
- Top marital: *married* (65%)  
- Top contact: *cellular* (61%)

**👉 Interpretation:** Heavy loan-dependent customers with low to moderate balances. They have personal loans across the board, limited prior contact, and are mostly working class.

**💡 Cross-sell ideas:**  
- Credit Counseling / Balance Transfer Offers  
- Secured Credit Cards (controlled exposure)  
- Health/Accident Insurance (to protect loan-dependent households)
"""
    },
    3: {
        "name": "Wealthy Independents",
        "md": """
## Segment 4: **Wealthy Independents**
- **Avg age:** ~43 years  
- **Avg balance:** €1,690  
- Personal loans: 0%  
- Housing loans: 0%  
- Default history: 0%  
- Previously contacted: 0%  
- Avg previous contacts: 0.00  
- Avg campaign calls: 2.99  
- Top job: *management* (24%)  
- Top marital: *married* (60%)  
- Top contact: *cellular* (69%)

**👉 Interpretation:** High-income, debt-free, financially independent individuals with no loan exposure and strong balances.

**💡 Cross-sell ideas:**  
- High-value Investments (stocks, bonds, portfolio mgmt)  
- Health & Retirement Insurance  
- Premium / Luxury Credit Cards (air miles, concierge)
"""
    },
    4: {
        "name": "Loan Seekers with Housing Loans",
        "md": """
## Segment 5: **Loan Seekers with Housing Loans**
- **Avg age:** ~39 years  
- **Avg balance:** €1,252  
- Personal loans: 0%  
- Housing loans: 100%  
- Default history: 0%  
- Previously contacted: 0%  
- Avg previous contacts: 0.00  
- Avg campaign calls: 2.83  
- Top job: *blue-collar* (29%)  
- Top marital: *married* (60%)  
- Top contact: *cellular* (48%)

**👉 Interpretation:** Middle-aged working-class families with stable housing loans and no personal loans. Moderate balances and low engagement.

**💡 Cross-sell ideas:**  
- Credit Cards (for daily purchases, EMI flexibility)  
- Top-up Loans (home improvement, education)  
- Life or Home Insurance (typical family protection products)
"""
    },
}


# --- UI ---
st.title("📊 Segment & Propensity Predictor")
st.caption("Enter customer details → get cluster name + rich description + purchase probability")

# --- Inputs ---
with st.form("form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 18, 100, 39)
        job = st.selectbox("Job", [
            "admin.","unknown","unemployed","management","housemaid","entrepreneur",
            "student","blue-collar","self-employed","retired","technician","services"
        ], index=7)
        marital = st.selectbox("Marital", ["married","single","divorced"], index=0)
        education = st.selectbox("Education", ["unknown","secondary","primary","tertiary"], index=1)
        default = st.selectbox("Default (credit in default?)", ["no","yes"], index=0)
        balance = st.number_input("Balance (€)", value=1200.0, step=50.0)

    with c2:
        housing = st.selectbox("Housing loan?", ["no","yes"], index=1)
        loan = st.selectbox("Personal loan?", ["no","yes"], index=0)
        contact = st.selectbox("Contact type", ["unknown","telephone","cellular"], index=2)
        day = st.number_input("Last contact day", 1, 31, 15)
        month = st.selectbox("Last contact month", [
            "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"
        ], index=4)
        campaign = st.number_input("Contacts in this campaign", 1, 60, 1)
        pdays = st.number_input("Days since last contact (-1 if never)", -1, 999, -1)
        previous = st.number_input("Previous contacts (before this campaign)", 0, 60, 0)
        poutcome = st.selectbox("Previous outcome", ["unknown","failure","other","success"], index=0)

    submitted = st.form_submit_button("Predict")

# --- Helpers to mirror notebook engineering for clustering ---
def make_clustering_row():
    binary = {'yes':1, 'no':0}
    balance_log = np.sign(balance) * np.log1p(abs(balance))
    prev_contacted = int(pdays != -1)
    if age <= 25: age_group = "<=25"
    elif age <= 35: age_group = "26-35"
    elif age <= 50: age_group = "36-50"
    elif age <= 65: age_group = "51-65"
    else: age_group = "65+"

    row = pd.DataFrame([{
        "age": age,
        "balance_log": balance_log,
        "campaign": campaign,
        "previous": previous,
        "day": day,
        "default": binary[default],
        "housing": binary[housing],
        "loan": binary[loan],
        "prev_contacted": prev_contacted,
        "job": job,
        "marital": marital,
        "education": education,
        "contact": contact,
        "month": month,
        "poutcome": poutcome,
        "age_group": age_group
    }])
    num_cols = ['age','balance_log','campaign','previous','day','default','housing','loan','prev_contacted']
    cat_cols = ['job','marital','education','contact','month','poutcome','age_group']
    return row[num_cols + cat_cols]

def make_propensity_row():
    # must match the raw columns used in the notebook (no 'duration')
    return pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,     # keep as yes/no; pipeline encodes
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

# --- Predict & display ---
if submitted:
    # 1) Cluster
    Xc = make_clustering_row()
    Xc_tr = cluster_preproc.transform(Xc)
    seg = int(kmeans.predict(Xc_tr)[0])

    seg_info = SEGMENT_INFO.get(seg, {"name": f"Segment {seg}", "md": f"## Segment {seg}\nDescription unavailable."})
    st.success(f"Predicted Segment: **{seg_info['name']}** (ID: {seg})")
    with st.expander("View segment description"):
        st.markdown(seg_info["md"])

    st.divider()

    # 2) Propensity
    Xp = make_propensity_row()
    proba = float(prop_pipe.predict_proba(Xp)[:,1][0])
    st.subheader("Propensity to Purchase")
    st.metric("Probability", f"{proba*100:.1f}%")
    st.progress(min(max(proba,0.0),1.0))

    if proba >= 0.6:
        st.success("High likelihood – reach out ASAP with a strong offer.")
    elif proba >= 0.35:
        st.info("Medium likelihood – personalize the pitch and follow-up.")
    else:
        st.warning("Low likelihood – nurture with educational content.")

st.write("---")
st.caption("Tip: If labels don’t match your intended segment names, edit SEGMENT_INFO to remap IDs 0–4.")
