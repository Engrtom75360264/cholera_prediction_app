import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from PIL import Image

# --------------------------- Page Config ---------------------------
st.set_page_config(
    page_title="🦠 Nigeria Cholera Outbreak Cases Prediction",
    page_icon="🦠",
    layout="wide"
)

# --------------------------- CSS ---------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        color: #000000;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        height: 3em;
        width: 10em;
        border-radius:10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------- Cached Model Loader ---------------------------
@st.cache_resource
def load_model(path):
    if not Path(path).exists():
        st.error("❌ Model not found! Run train_save.py first.")
        return None
    return joblib.load(path)

model = load_model("models/best_model.joblib")

# --------------------------- Sidebar Inputs ---------------------------
st.sidebar.header("🦠 Input Features")

state_list = [
    'Abia', 'Adamawa', 'Akwa Ibom', 'Anambra', 'Bauchi', 'Bayelsa', 'Benue', 
    'Borno', 'Cross River', 'Delta', 'Ebonyi', 'Edo', 'Ekiti', 'Enugu', 
    'FCT', 'Gombe', 'Imo', 'Jigawa', 'Kaduna', 'Kano', 'Katsina', 'Kebbi', 
    'Kogi', 'Kwara', 'Lagos', 'Nasarawa', 'Niger', 'Ogun', 'Ondo', 'Osun', 
    'Oyo', 'Plateau', 'Rivers', 'Sokoto', 'Taraba', 'Yobe', 'Zamfara'
]

year = st.sidebar.number_input("Year", min_value=2000, max_value=2030, value=2023, step=1)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=1, step=1)
state = st.sidebar.selectbox("State", state_list)
st.sidebar.markdown("**Note:** WHO baseline CFR for cholera is 1%. Adjust if necessary.")
cfr = st.sidebar.number_input("Case Fatality Rate (CFR %)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
population = st.sidebar.number_input("Population", min_value=0, value=1000, step=100)

# --------------------------- Main Title ---------------------------
st.title("🦠 NIGERIA CHOLERA CASES PREDICTION")
st.markdown("""
Enter the relevant details in the sidebar and click **Predict** to see the estimated cholera cases per 100,000 people.
""")

# --------------------------- Prediction ---------------------------
def predict_cases(model, year, month, state, cfr, population):
    input_df = pd.DataFrame({
        'Year': [year],
        'Month': [month],
        'State': [state],
        'CFR (%)': [cfr],
        'Population': [population]
    })
    predicted_rate = model.predict(input_df)[0]
    predicted_cases = (predicted_rate / 100000) * population
    return predicted_rate, predicted_cases

if st.sidebar.button("Predict Cases per 100,000"):
    if model:
        try:
            predicted_rate, predicted_cases = predict_cases(model, year, month, state, cfr, population)

            st.subheader("📊 Prediction Result")
            st.metric("Predicted Cases per 100,000", f"{predicted_rate:.2f}")
            st.metric("Estimated Total Cases", f"{predicted_cases:.0f}")

            st.write("### 🧭 Risk Level Interpretation")
            if predicted_rate < 5:
                st.success("✅ **Low Risk:** Controlled outbreak level (below 5 per 100,000).")
            elif 5 <= predicted_rate < 20:
                st.warning("⚠️ **Moderate Risk:** Monitor closely and strengthen surveillance.")
            else:
                st.error("🚨 **High Risk:** Severe outbreak risk. Immediate intervention needed!")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# --------------------------- Display Images Lazily ---------------------------
st.markdown("### 🔬 Cholera Bacteria")
col1, col2 = st.columns(2)

def load_image(path):
    if Path(path).exists():
        return Image.open(path)
    return None

img1 = load_image("Statics/cholera_bacteria_image.png")
img2 = load_image("Statics/cholera_on_petrishbox.png")

if img1:
    col1.image(img1, use_container_width=True)
if img2:
    col2.image(img2, use_container_width=True)
