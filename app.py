from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


FEATURES = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]


st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 35%, #1f2937 100%);
        color: #f8fafc;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 18px;
        padding: 1.2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    .stNumberInput, .stSelectbox {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 12px;
    }
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stForm"] label,
    .stSubheader,
    .stMarkdown p,
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3 {
        color: #f8fafc !important;
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div[role="combobox"],
    div[data-testid="stNumberInput"] div,
    div[data-testid="stSelectbox"] div,
    div[data-testid="stNumberInput"] .st-bb,
    div[data-testid="stSelectbox"] .st-bb {
        color: #f8fafc !important;
        background-color: rgba(15, 23, 42, 0.9) !important;
    }
    div[data-testid="stNumberInput"] input::placeholder,
    div[data-testid="stSelectbox"] input::placeholder {
        color: rgba(248, 250, 252, 0.7) !important;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ef4444, #f97316);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.8rem 1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #f97316, #ef4444);
    }
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
    .css-1d391kg, .css-1cpxqw2 {
        color: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    for filename in ("model.pkl", "model (2).pkl"):
        model_path = Path(__file__).parent / filename
        if model_path.exists():
            return joblib.load(model_path)
    raise FileNotFoundError("model.pkl or model (2).pkl was not found")


st.title("❤️ Heart Disease Prediction")
st.caption("Medical risk assessment dashboard")

try:
    loaded_model = load_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


col1, col2 = st.columns([1.2, 0.8])

with col1:
    with st.form("prediction_form"):
        st.subheader("Patient Details")

        age = st.number_input("Age", min_value=1, max_value=120, value=42)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
        chest_pain_type = st.number_input("ChestPainType", min_value=0, max_value=3, value=2)
        resting_bp = st.number_input("RestingBP", min_value=0, max_value=300, value=120)
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=700, value=240)
        fasting_bs = st.selectbox("FastingBS", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
        resting_ecg = st.number_input("RestingECG", min_value=0, max_value=2, value=1)
        max_hr = st.number_input("MaxHR", min_value=0, max_value=250, value=194)
        exercise_angina = st.selectbox("ExerciseAngina", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=0.8, step=0.1)
        st_slope = st.number_input("ST_Slope", min_value=0, max_value=2, value=0)

        predict_button = st.form_submit_button("Predict Risk")

with col2:
    st.subheader("Prediction Result")
    st.markdown("###")

    if predict_button:
        inp = pd.DataFrame(
            [[
                age,
                sex,
                chest_pain_type,
                resting_bp,
                cholesterol,
                fasting_bs,
                resting_ecg,
                max_hr,
                exercise_angina,
                oldpeak,
                st_slope,
            ]],
            columns=FEATURES,
        )

        prediction = loaded_model.predict(inp)
        result = int(prediction[0])

        if result == 1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease Detected")

        st.metric("Prediction Value", result)

        if hasattr(loaded_model, "predict_proba"):
            confidence = loaded_model.predict_proba(inp)[0][result] * 100
            st.metric("Confidence", f"{confidence:.2f}%")

        st.dataframe(inp, use_container_width=True)
    else:
        st.info("Fill the form and press Predict Risk to see the result.")

