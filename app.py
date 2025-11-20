import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------
# File paths for both environments
# -----------------------------
MODEL_PATHS = [
    "/mnt/data/Student_model.pkl",  
    "Student_model.pkl"
]

DATA_PATHS = [
    "/mnt/data/Employee_clean_Data.csv",
    "Employee_clean_Data.csv"
]

def resolve_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

MODEL_PATH = resolve_path(MODEL_PATHS)
DATA_PATH = resolve_path(DATA_PATHS)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    if MODEL_PATH is None:
        st.error("❌ Model file missing. Upload Student_model.pkl.")
        return None
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    if DATA_PATH is None:
        st.error("❌ Data file missing. Upload Employee_clean_Data.csv.")
        return None
    return pd.read_csv(DATA_PATH)

model = load_model()
data = load_data()

if model is None or data is None:
    st.stop()

# -----------------------------
# UI
# -----------------------------
st.title("Student Prediction App")
st.write("This app uses your Student_model.pkl to predict output.")

st.subheader("Dataset Preview")
st.dataframe(data.head())

# -----------------------------
# Inputs
# -----------------------------
st.subheader("Enter Input Values")

numeric_columns = [c for c in data.columns if data[c].dtype != 'object']
user_inputs = {}

for col in numeric_columns:
    default_value = float(data[col].mean())
    user_inputs[col] = st.number_input(f"Enter {col}", value=default_value)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([user_inputs])

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🎯 Prediction: **{prediction}**")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
