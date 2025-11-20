import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Configuration ---
MODEL_PATH = 'student_model.pkl'

# --- Custom Styling ---
st.markdown("""
<style>
.main-header {
    font-size: 32px;
    font-weight: 700;
    color: #4CAF50;
    text-align: center;
    margin-bottom: 20px;
}
.sub-header {
    font-size: 20px;
    font-weight: 600;
    color: #1E88E5;
    margin-top: 15px;
}
.stSlider > div > div > div:nth-child(2) {
    background-color: #2196F3;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 24px;
    border: none;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
}
.stButton>button:hover {
    background-color: #388E3C;
}
</style>
""", unsafe_allow_html=True)


# --- Functions ---

@st.cache_resource
def load_model(path):
    """Loads the pickled model object."""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{path}'. Please ensure the model file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_score(model, data):
    """Performs prediction using the loaded model."""
    try:
        # Prediction expects a 2D array-like input
        prediction = model.predict(data)
        # Ensure the output is a single scalar value for display
        return np.round(prediction[0], 2)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# --- Main Application ---

st.markdown('<p class="main-header">Student Performance Predictor</p>', unsafe_allow_html=True)
st.write("Predict the final score of a student based on their study habits and attendance.")

# Load the model
model = load_model(MODEL_PATH)

if model is not None:
    st.markdown('<p class="sub-header">Student Input Features</p>', unsafe_allow_html=True)

    # 1. Hours Studied (Float/Int Input)
    hours_studied = st.slider(
        'Hours Studied per Week',
        min_value=1.0, 
        max_value=20.0, 
        value=5.0, 
        step=0.5,
        help='Average number of hours the student dedicated to studying.'
    )

    # 2. Attendance (Percentage Input)
    attendance = st.slider(
        'Attendance Percentage (%)',
        min_value=50, 
        max_value=100, 
        value=85, 
        step=1,
        help='Overall attendance rate in percentage.'
    )

    # 3. Assignments Submitted (Int Input)
    # Assuming maximum assignments is 10 based on the provided sample data
    assignments_submitted = st.slider(
        'Assignments Submitted',
        min_value=0, 
        max_value=10, 
        value=8, 
        step=1,
        help='Number of assignments completed out of the total.'
    )

    # Prepare input data for the model
    # The order must match the order the model was trained on!
    input_data = pd.DataFrame([[hours_studied, attendance, assignments_submitted]],
                              columns=['Hours_Studied', 'Attendance', 'Assignments_Submitted'])

    st.markdown("---")

    # Prediction Button
    if st.button('Predict Final Score'):
        # Get the prediction
        predicted_score = predict_score(model, input_data)

        if predicted_score is not None:
            # Display the result
            st.success(f"### Predicted Score: {predicted_score}")
            st.balloons()

    # Optional: Display the first few rows of the training data
    st.markdown('<p class="sub-header">Original Training Data Snippet</p>', unsafe_allow_html=True)
    # Note: In a real app, you might not include the CSV file itself, 
    # but here we show it for context based on your file upload.
    df_data = pd.read_csv("student_scores (1) - Copy.csv")
    st.dataframe(df_data.head(), use_container_width=True)


st.info("Note: This prediction is based on the trained linear model provided.")
