import streamlit as st
import pandas as pd
import joblib

# ==============================================================
# PAGE CONFIG
# ==============================================================

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📱",
    layout="wide"
)



# SIDEBAR START
st.sidebar.title("📱 Telecom Churn Predictor")

st.sidebar.markdown("---")

st.sidebar.header("📊 Model Overview")

st.sidebar.markdown("""
**Model Type:**  
Logistic Regression

**🎯 Accuracy:**  
79.91%

**📦 Dataset Size:**  
7,043 Customers

**🧠 Features Used:**  
19 Input Features
""")

st.sidebar.markdown("---")

st.sidebar.header("👩‍💻 Developer")

st.sidebar.markdown("""
**Ashi Saini**  
ML & Data Science Enthusiast
""")

st.sidebar.info("🚀 Built using Streamlit + Scikit-learn")
# SIDEBAR END


# ==============================================================
# LOAD MODEL
# ==============================================================

model = joblib.load("models/best_model.pkl")

scaler = joblib.load("models/scaler.pkl")



# ==============================================================
# TITLE
# ==============================================================

st.title("📱 Telecom Customer Churn Prediction")

st.markdown("""
Predict whether a telecom customer is likely to churn using
**Machine Learning**.
""")

# Dashboard Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🎯 Accuracy",
        value="79.91%"
    )

with col2:
    st.metric(
        label="👥 Customers",
        value="7,043"
    )

with col3:
    st.metric(
        label="📊 Features",
        value="19"
    )

with col4:
    st.metric(
        label="🧠 Model",
        value="Logistic Regression"
    )

st.markdown("---")

# ==============================================================
# CUSTOMER INPUT FORM
# ==============================================================

st.header("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )

with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

st.markdown("---")




        
# ==============================================================
# INTERNET SERVICES
# ==============================================================

st.header("🌐 Internet Services")

col3, col4 = st.columns(2)

with col3:

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

with col4:

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

st.markdown("---")

# ==============================================================
# BILLING INFORMATION
# ==============================================================

st.header("💳 Billing Information")

col5, col6 = st.columns(2)

with col5:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col6:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )

st.markdown("---")

predict = st.button(
    "🔍 Predict Churn",
    use_container_width=True
)
# ==============================================================
# PREDICTION LOGIC
# ==============================================================

if predict:

    # ----------------------------
    # Manual Encoding
    # ----------------------------

    gender = 0 if gender == "Female" else 1

    senior = 0 if senior == "No" else 1

    partner = 0 if partner == "No" else 1

    dependents = 0 if dependents == "No" else 1

    phone_service = 0 if phone_service == "No" else 1

    multiple_lines_map = {
        "No": 0,
        "No phone service": 1,
        "Yes": 2
    }

    internet_service_map = {
        "DSL": 0,
        "Fiber optic": 1,
        "No": 2
    }

    online_service_map = {
        "No": 0,
        "No internet service": 1,
        "Yes": 2
    }

    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }

    paperless = 0 if paperless == "No" else 1

    payment_method_map = {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }

    # ----------------------------
    # Create Input DataFrame
    # ----------------------------

    input_data = pd.DataFrame({

        "gender":[gender],
        "SeniorCitizen":[senior],
        "Partner":[partner],
        "Dependents":[dependents],
        "tenure":[tenure],
        "PhoneService":[phone_service],
        "MultipleLines":[multiple_lines_map[multiple_lines]],
        "InternetService":[internet_service_map[internet_service]],
        "OnlineSecurity":[online_service_map[online_security]],
        "OnlineBackup":[online_service_map[online_backup]],
        "DeviceProtection":[online_service_map[device_protection]],
        "TechSupport":[online_service_map[tech_support]],
        "StreamingTV":[online_service_map[streaming_tv]],
        "StreamingMovies":[online_service_map[streaming_movies]],
        "Contract":[contract_map[contract]],
        "PaperlessBilling":[paperless],
        "PaymentMethod":[payment_method_map[payment_method]],
        "MonthlyCharges":[monthly_charges],
        "TotalCharges":[total_charges]

    })

    # ----------------------------
    # Feature Scaling
    # ----------------------------

    input_scaled = scaler.transform(input_data)

    # ==============================================================
# MAKE PREDICTION
# ==============================================================

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    churn_probability = probability[0][1] * 100
    stay_probability = probability[0][0] * 100

    st.markdown("---")

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.error(f"""
🚨 **HIGH CHURN RISK**

**Churn Probability:** {churn_probability:.2f}%

### 💡 Recommendation
- 📞 Contact customer immediately
- 🎁 Offer a loyalty discount
- 📅 Suggest a long-term contract
- 💬 Provide premium customer support
""")
    else:
        st.success(f"""
🎉 **LOW CHURN RISK**

**Stay Probability:** {stay_probability:.2f}%

### 💡 Recommendation
- ✅ Continue current service
- 🌟 Reward customer loyalty
- 📦 Recommend premium plans
- 😊 Maintain customer engagement
""")

    # Show probabilities
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="✅ Stay Probability",
            value=f"{stay_probability:.2f}%"
        )

    with col2:
        st.metric(
            label="❌ Churn Probability",
            value=f"{churn_probability:.2f}%"
        )

    st.subheader("📈 Probability Comparison")
    chart_data = pd.DataFrame({
    "Probability": [stay_probability, churn_probability]
}, index=["Stay", "Churn"])

    st.bar_chart(chart_data)

    # Progress Bar
    st.progress(max(stay_probability, churn_probability) / 100)

    st.markdown("---")

    # Customer Summary
    st.subheader("📋 Customer Summary")
    st.dataframe(input_data)

    st.markdown("---")

    # Feature Importance
    st.subheader("⭐ Feature Importance (Model Explainability)")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
### 📊 Key Insight

- Shows most powerful features affecting churn
- Helps understand customer behavior
- Improves model transparency
- Useful for business decisions
""")

        st.info(
            "💡 Example: Contract type, tenure, and monthly charges are usually top factors"
        )

    with col2:
        st.image(
            "images/feature_importance.png",
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("💼 Business Insights")

    if prediction[0] == 1:
        st.warning("""
### 🚨 Recommended Retention Strategy

- 📞 Contact the customer within 48 hours
- 💰 Offer a personalized discount
- 📅 Recommend an annual contract
- 🎁 Provide loyalty rewards
- 💬 Assign premium customer support
""")
    else:
        st.success("""
### 🌟 Customer Retention Status

- 😊 Customer is likely to stay
- 🎯 Recommend premium plan upgrades
- 🎁 Continue loyalty rewards
- 📧 Send personalized offers
- ⭐ Maintain excellent customer service
""")
        
    st.markdown("---")
    st.subheader("📥 Download Prediction Report")
    
    result_df = pd.DataFrame({
    "Prediction": ["Stay" if prediction[0] == 0 else "Churn"],
    "Stay Probability (%)": [round(stay_probability, 2)],
    "Churn Probability (%)": [round(churn_probability, 2)],
    "Contract": [contract],
    "Internet Service": [internet_service],
    "Monthly Charges": [monthly_charges],
    "Total Charges": [total_charges],
    "Tenure (Months)": [tenure]
    })

    st.download_button(
    label="📥 Download Prediction Report (CSV)",
    data=result_df.to_csv(index=False),
    file_name="customer_churn_prediction.csv",
    mime="text/csv"
    )

    
