import streamlit as st
import mlflow.pyfunc
import pandas as pd
import mlflow

# -------------------------------------------------
# 🌟 App Title
# -------------------------------------------------
st.title("💰 EMI Prediction App")
st.write("This app predicts your **Maximum EMI Amount** using your financial profile and loan details.")

# -------------------------------------------------
# 🧾 User Inputs
# -------------------------------------------------
st.header("Enter your details:")

employment_type = st.selectbox("Employment Type", ["Private", "Self-employed"])
company_type = st.selectbox("Company Type", ["MNC", "Mid-size", "Small", "Startup"])
house_type = st.selectbox("House Type", ["Own", "Rented"])
existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
emi_scenario = st.selectbox(
    "EMI Scenario",
    ["Education EMI", "Home Appliances EMI", "Personal Loan EMI", "Vehicle EMI"]
)

monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0, value=50000, step=1000)
years_of_employment = st.number_input("Years of Employment", min_value=0, value=3)
monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0, value=5000, step=1000)
family_size = st.number_input("Family Size", min_value=1, value=2)
dependents = st.number_input("Dependents", min_value=0, value=1)
school_fees = st.number_input("School Fees (₹)", min_value=0, value=0)
college_fees = st.number_input("College Fees (₹)", min_value=0, value=0)
travel_expenses = st.number_input("Travel Expenses (₹)", min_value=0, value=3000)
groceries_utilities = st.number_input("Groceries & Utilities (₹)", min_value=0, value=8000)
other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", min_value=0, value=2000)
current_emi_amount = st.number_input("Current EMI Amount (₹)", min_value=0, value=0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
bank_balance = st.number_input("Bank Balance (₹)", min_value=0, value=50000)
emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0, value=10000)
requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=0, value=200000)
requested_tenure = st.number_input("Requested Tenure (Months)", min_value=6, max_value=120, value=24)

# -------------------------------------------------
# 🧮 Create Input DataFrame
# -------------------------------------------------
input_dict = {
    'monthly_salary': monthly_salary,
    'years_of_employment': years_of_employment,
    'monthly_rent': monthly_rent,
    'family_size': family_size,
    'dependents': dependents,
    'school_fees': school_fees,
    'college_fees': college_fees,
    'travel_expenses': travel_expenses,
    'groceries_utilities': groceries_utilities,
    'other_monthly_expenses': other_monthly_expenses,
    'current_emi_amount': current_emi_amount,
    'credit_score': credit_score,
    'bank_balance': bank_balance,
    'emergency_fund': emergency_fund,
    'requested_amount': requested_amount,
    'requested_tenure': requested_tenure
}

input_df = pd.DataFrame([input_dict])

# -------------------------------------------------
# 🧩 Derived Features (same as training)
# -------------------------------------------------
input_df['debt_to_income'] = (
    input_df['current_emi_amount'] / input_df['monthly_salary']
).replace([float("inf"), -float("inf")], 0).fillna(0)

input_df['expense_to_income'] = (
    input_df['groceries_utilities'] / input_df['monthly_salary']
).replace([float("inf"), -float("inf")], 0).fillna(0)

input_df['affordability_ratio'] = (
    input_df['requested_amount'] / (input_df['monthly_salary'] * input_df['requested_tenure'])
).replace([float("inf"), -float("inf")], 0).fillna(0)

input_df['emi_burden'] = (
    input_df['current_emi_amount'] / (input_df['monthly_salary'] + 1)
).replace([float("inf"), -float("inf")], 0).fillna(0)

input_df['no_credit_history'] = (input_df['credit_score'] <= 0).astype(int)

# -------------------------------------------------
# 🧱 One-Hot Encoding (match all expected columns)
# -------------------------------------------------
for col in [
    'employment_type_Private', 'employment_type_Self-employed',
    'company_type_MNC', 'company_type_Mid-size', 'company_type_Small', 'company_type_Startup',
    'house_type_Own', 'house_type_Rented',
    'existing_loans_Yes',
    'emi_scenario_Education EMI', 'emi_scenario_Home Appliances EMI',
    'emi_scenario_Personal Loan EMI', 'emi_scenario_Vehicle EMI'
]:
    input_df[col] = 0

# Set selected categories to 1
input_df[f'employment_type_{employment_type}'] = 1
input_df[f'company_type_{company_type}'] = 1
input_df[f'house_type_{house_type}'] = 1
if existing_loans == "Yes":
    input_df['existing_loans_Yes'] = 1
input_df[f'emi_scenario_{emi_scenario}'] = 1

# -------------------------------------------------
# 🚀 Load Model from MLflow Registry
# -------------------------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")

try:
    model = mlflow.pyfunc.load_model("models:/EMI_Regression_Model@production")
    st.success("✅ Model loaded successfully from MLflow registry.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# -------------------------------------------------
# 📊 Make Prediction
# -------------------------------------------------
if st.button("Predict EMI"):
    try:

        try:
            expected_columns = model._model_impl.sklearn_model.get_booster().feature_names
            input_df = input_df[expected_columns]
        except Exception:
            st.warning("Could not fetch model schema automatically. Using DataFrame as-is.")
        
        prediction = model.predict(input_df)
        st.success(f"💰 Predicted Maximum EMI Amount: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & MLflow | EMI Prediction System")
