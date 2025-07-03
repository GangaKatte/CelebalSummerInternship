import streamlit as st
import joblib
import numpy as np

# Load model
import os
model_path = os.path.join(os.path.dirname(__file__), "creditworthiness_rf_model.pkl")
model = joblib.load(model_path)

# StandardScaler values from training 
mean_duration = 20
std_duration = 10

mean_credit_amount = 3000
std_credit_amount = 2000

mean_age = 35
std_age = 11

st.set_page_config(page_title="Creditworthiness Predictor", layout="wide")
st.title("💰 Creditworthiness Predictor")
st.markdown("📘 **Note:** 1 DM (Deutsche Mark) = 0.511 Euro (€) ≈ ₹46 INR (as of 2025)")

with st.expander("ℹ️ About this app"):
    st.markdown("""
    - Predict whether a person is **creditworthy** using financial attributes.
    - Dataset from Germany (1990s), where money values are in **DM**.
    
    """)

# Mappings
guarantees_map = {"None": 0, "Guarantor": 1, "Co-applicant": 2}
checking_map = {"< 0 DM": 0, "0 ≤ ... < 200 DM": 1, "≥ 200 DM": 2, "No checking account": 3}
credit_history_map = {"No credits taken": 0, "All paid back duly": 1, "Existing paid back": 2, "Delay": 3, "Critical": 4}
purpose_map = {"Car (new)": 0, "Car (used)": 1, "Furniture": 2, "Radio / TV": 3, "Education": 4, "Business": 5, "Repairs": 6, "Vacation": 7, "Other": 8}
savings_map = {"Unknown / No savings": 0, "< 100 DM": 1, "100 ≤ ... < 500 DM": 2, "500 ≤ ... < 1000 DM": 3, "≥ 1000 DM": 4}
employment_map = {"Unemployed": 0, "< 1 year": 1, "1 ≤ ... < 4 years": 2, "4 ≤ ... < 7 years": 3, "≥ 7 years": 4}
personal_status_map = {"Male - Single": 0, "Male - Divorced/Married": 1, "Female - Divorced/Married": 2,"Female - Single": 3}
property_map = {"Real estate": 0, "Savings": 1, "Car or other": 2, "Unknown": 3}
installment_plan_map = {"Bank": 0, "Stores": 1, "None": 2}
housing_map = {"Own": 0, "For free": 1, "Rent": 2}
job_map = {"Unemployed": 0, "Unskilled": 1, "Skilled": 2, "Management": 3}

col1, col2 = st.columns(2)
features = []

with col1:
    checking = st.selectbox("Checking account", checking_map)
    credit_history = st.selectbox("Credit history", credit_history_map)
    purpose = st.selectbox("Purpose", purpose_map)
    savings = st.selectbox("Savings", savings_map)
    employment = st.selectbox("Employment duration", employment_map)
    personal_status = st.selectbox("Personal status", personal_status_map)
    guarantor = st.selectbox("Guarantor", guarantees_map)
    property_val = st.selectbox("Property", property_map)
    installment_plan = st.selectbox("Installment plan", installment_plan_map)
    housing = st.selectbox("Housing", housing_map)

with col2:
    duration_input = st.number_input("Loan Duration (in months)", min_value=6, max_value=72, step=6, value=12)
    duration = (duration_input - mean_duration) / std_duration

    credit_amount_input = st.number_input("Credit Amount (in DM)", min_value=100, max_value=10000, step=100, value=1000)
    credit_amount = (credit_amount_input - mean_credit_amount) / std_credit_amount

    installment_rate = st.number_input("Installment rate (% of income)", value=2.0, step=1.0)
    residence_since = st.number_input("Years at residence", value=2, step=1)

    age_input = st.number_input("Age (in years)", min_value=18, max_value=75, value=30)
    age = (age_input - mean_age) / std_age

    existing_credits = st.number_input("# of existing credits", value=1, step=1)
    job = st.selectbox("Job type", job_map)
    dependents = st.number_input("# of dependents", value=1, step=1)
    telephone = st.selectbox("Telephone", ["Yes", "No"])
    foreign_worker = st.selectbox("Foreign worker", ["Yes", "No"])

# Final features list
features.extend([
    checking_map[checking],
    duration,
    credit_history_map[credit_history],
    purpose_map[purpose],
    credit_amount,
    savings_map[savings],
    employment_map[employment],
    installment_rate,
    personal_status_map[personal_status],
    guarantees_map[guarantor],
    residence_since,
    property_map[property_val],
    age,
    installment_plan_map[installment_plan],
    housing_map[housing],
    existing_credits,
    job_map[job],
    dependents,
    1 if telephone == "Yes" else 0,
    1 if foreign_worker == "Yes" else 0
])
def get_reasons(features):
    reasons = []
    
    if features[1] > 1:  # Duration
        reasons.append("⏱️ Long loan duration may be risky.")
        
    if features[4] > 1:  # Credit amount
        reasons.append("💰 High credit amount requested.")
        
    if features[6] == 0:  # Unemployed
        reasons.append("🚫 Employment duration is too short or unemployed.")
        
    if features[12] < -1:  # Age
        reasons.append("🧒 Age is too young, limited credit history.")
        
    if features[10] < 1:  # Short residence
        reasons.append("🏠 Short residence duration may affect stability.")

    if features[8] > 2:  # Personal status
        reasons.append("👤 Personal status may indicate higher risk.")

    return reasons
st.markdown("---")
if st.button("🔮 Predict Creditworthiness"):
    prediction = model.predict([features])[0]
    result = "✅ Creditworthy" if prediction == 1 else "❌ Not Creditworthy"
    st.success(f"Prediction: {result}")

    if prediction == 0:
        st.markdown("### ⚠️ Possible Reasons for Bad Credit:")
        for reason in get_reasons(features):
            st.markdown(f"- {reason}")


