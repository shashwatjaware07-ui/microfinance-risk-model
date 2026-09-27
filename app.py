import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration for a cleaner UI
st.set_page_config(page_title="Credit Risk Assessor", page_icon="🏦", layout="centered")

# --- 1. Load the Saved Model and Scaler ---
# We load the objects we saved in main.ipynb.
# This saves compute since we don't need to retrain the Logistic Regression model.
@st.cache_resource
def load_assets():
    model = joblib.load('credit_risk_model.pkl')
    scaler = joblib.load('credit_risk_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# --- 2. Build the User Interface (Sidebar) ---
st.title("🏦 Credit Risk Assessment App")
st.markdown("""
Welcome to the Credit Risk Assessor! 
Enter the applicant's financial details in the sidebar and click **Calculate Default Risk** to see if they should be approved for a loan.
""")

st.sidebar.header("Applicant Financial Details")
st.sidebar.write("Please enter the financial information:")

# Input fields matching the core financial features in the dataset
# We provide reasonable default values to start with.
income = st.sidebar.number_input("Total Income ($)", min_value=0.0, value=50000.0, step=1000.0)
credit = st.sidebar.number_input("Credit Amount ($)", min_value=0.0, value=150000.0, step=1000.0)
annuity = st.sidebar.number_input("Annuity ($)", min_value=0.0, value=7500.0, step=500.0)
goods_price = st.sidebar.number_input("Goods Price ($)", min_value=0.0, value=150000.0, step=1000.0)

# --- 3. Process Inputs and Calculate Risk ---
# A large prominent button as requested
if st.button("Calculate Default Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing risk..."):
        # The model expects exactly 228 features (due to one-hot encoding).
        # To handle this cleanly, we initialize a 'base applicant' DataFrame with 0s.
        input_data = pd.DataFrame(0, index=[0], columns=model.feature_names_in_)
        
        # For numeric columns, we set them to the mean of the training data 
        # so that missing values don't heavily skew the Logistic Regression.
        numeric_cols = scaler.feature_names_in_
        input_data[numeric_cols] = scaler.mean_
        
        # Now, we update the base applicant with the user's specific inputs!
        # IMPORTANT: In main.ipynb, we applied a log1p transformation to these financial columns.
        # We MUST apply the exact same transformation here before scaling.
        input_data.loc[0, 'AMT_INCOME_TOTAL'] = np.log1p(income)
        input_data.loc[0, 'AMT_CREDIT'] = np.log1p(credit)
        input_data.loc[0, 'AMT_ANNUITY'] = np.log1p(annuity)
        input_data.loc[0, 'AMT_GOODS_PRICE'] = np.log1p(goods_price)
        
        # Scale the numeric features using our loaded StandardScaler
        input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
        
        # Predict the probability of default
        # predict_proba returns an array: [[Probability of Class 0, Probability of Class 1]]
        # Class 1 is 'Default', so we want index 1.
        probabilities = model.predict_proba(input_data)[0]
        prob_default = probabilities[1]
        
        # --- 4. Display the Results ---
        st.subheader("Risk Assessment Result")
        st.write(f"**Probability of Default:** {prob_default:.2%}")
        
        # Show success or warning messages based on the 35% threshold
        if prob_default < 0.35:
            st.success("✅ **[APPROVED]** The applicant's default risk is below the 35% threshold.")
        else:
            st.error("🚨 **[DENIED]** The applicant's default risk exceeds the 35% threshold.")
