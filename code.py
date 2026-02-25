import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS (clean & modern)
# -------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #F8F9FA;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}
.result-high {
    background-color: #FDEDEC;
    border-left: 8px solid #E74C3C;
}
.result-low {
    background-color: #E9F7EF;
    border-left: 8px solid #27AE60;
}
h1, h2, h3 {
    color: #1F4E79;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🩺 Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:gray;'>Predict diabetes risk using patient health metrics</p>",
    unsafe_allow_html=True
)

# -------------------------------------------------
# Load model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

model = load_model()

# -------------------------------------------------
# Layout
# -------------------------------------------------
left, right = st.columns([1.2, 1])

# -------------------------------------------------
# Input Section
# -------------------------------------------------
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧾 Patient Details")

    col1, col2 = st.columns(2)
    pregnancies = col1.number_input("Pregnancies", 0, 20, 1)
    age = col2.number_input("Age", 1, 120, 30)

    col1, col2 = st.columns(2)
    glucose = col1.number_input("Glucose Level", 0, 300, 120)
    blood_pressure = col2.number_input("Blood Pressure", 0, 200, 70)

    col1, col2 = st.columns(2)
    skin_thickness = col1.number_input("Skin Thickness", 0, 100, 20)
    insulin = col2.number_input("Insulin", 0, 900, 80)

    col1, col2 = st.columns(2)
    bmi = col1.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = col2.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

    predict_btn = st.button("🔍 Predict Risk")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Prediction Section
# -------------------------------------------------
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if predict_btn:
        input_df = pd.DataFrame([[
            pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, dpf, age
        ]], columns=[
            "Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"
        ])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class='card result-high'>
                <h2>⚠️ HIGH RISK</h2>
                <h3>{probability:.2f}% probability</h3>
                <p>The patient is likely to have diabetes.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='card result-low'>
                <h2>✅ LOW RISK</h2>
                <h3>{probability:.2f}% probability</h3>
                <p>The patient is unlikely to have diabetes.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Enter patient details and click **Predict Risk**")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    "<p style='text-align:center; color:gray; font-size:12px;'>"
    "⚠️ For educational purposes only. Not a medical diagnosis."
    "</p>",
    unsafe_allow_html=True
)
