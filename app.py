import streamlit as st
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ----------------------------
# MODEL DEFINITION
# ----------------------------
class CustomMLPModel(nn.Module):
    def __init__(self):
        super(CustomMLPModel, self).__init__()
        self.fc1 = nn.Linear(12, 24)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(24, 12)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(12, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# ----------------------------
# LOAD MODEL & SCALER
# ----------------------------
@st.cache_resource
def load_model():
    model = CustomMLPModel()
    model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_data
def load_scaler():
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.save')
    return joblib.load(scaler_path)

# ----------------------------
# APP STYLE
# ----------------------------
st.set_page_config(page_title="Fitbit Calories Predictor", layout="wide")
st.markdown("""
<style>
body {background-color: #f0f2f6;}
h1 {text-align: center; color: #1e3d59; font-size: 2.2rem; font-weight: bold;}
h2 {color: #1e3d59;}
.section {background-color: #ffffff; padding: 1rem 2rem; border-radius: 12px; box-shadow: 0px 4px 12px rgba(0,0,0,0.08); margin-bottom: 1rem;}
.metric-label {font-size: 1.1rem; color: #555;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# MAIN APP
# ----------------------------
def main():
    st.title("🔥 Fitbit Calories Burned Predictor 🔥")
    st.write("Enter your daily Fitbit activity data below to predict calories burned.")

    model = load_model()
    scaler = load_scaler()

    feature_descriptions = {
        "TotalSteps": "Total steps taken in the day",
        "TotalDistance": "Total distance in kilometers",
        "TrackerDistance": "Distance recorded by tracker in kilometers",
        "LoggedActivitiesDistance": "Distance from manually logged activities in kilometers",
        "VeryActiveDistance": "Very active movement distance in kilometers",
        "ModeratelyActiveDistance": "Moderately active movement distance in kilometers",
        "LightActiveDistance": "Light activity distance in kilometers",
        "SedentaryActiveDistance": "Distance during sedentary time in kilometers",
        "VeryActiveMinutes": "Minutes spent being very active",
        "FairlyActiveMinutes": "Minutes spent being fairly active",
        "LightlyActiveMinutes": "Minutes spent being lightly active",
        "SedentaryMinutes": "Minutes spent sedentary"
    }

    # Initialize session state for inputs
    if "inputs" not in st.session_state:
        st.session_state.inputs = {feature: 0.0 for feature in feature_descriptions.keys()}

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📥 Enter Your Activity Data")

    # Layout inputs in two columns
    col1, col2 = st.columns(2)
    user_input = {}
    missing_inputs = False

    for i, (feature, desc) in enumerate(feature_descriptions.items()):
        col = col1 if i % 2 == 0 else col2
        user_input[feature] = col.number_input(
            feature,
            help=desc,
            value=st.session_state.inputs[feature],
            key=feature,
            format="%.2f",
            min_value=0.0
        )
        st.session_state.inputs[feature] = user_input[feature]
        if user_input[feature] == 0.0:
            missing_inputs = True

    # Clear button
    if st.button("🧹 Clear Inputs"):
        for key in st.session_state.inputs:
            st.session_state.inputs[key] = 0.0
        st.experimental_rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Generate Prediction
    if st.button("🚀 Generate Prediction"):
        if missing_inputs:
            st.warning("⚠️ Please fill in all fields with your actual data.")
        else:
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler.transform(input_df)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

            with torch.no_grad():
                prediction = model(input_tensor).item()

            # Display prediction as a metric card
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.subheader("📊 Predicted Calories Burned")
            st.metric(label="🔥 Calories Burned", value=f"{prediction:.2f} kcal")

            # Visual feedback based on calories
            if prediction < 1500:
                st.info("🟢 Low calorie burn, try adding more activity!")
            elif prediction < 2500:
                st.success("✅ Moderate calorie burn, good job!")
            else:
                st.warning("⚠️ High calorie burn, make sure to stay hydrated!")

            # Expander for input data
            with st.expander("🔍 Show Input Data Used for Prediction"):
                st.dataframe(input_df)

            st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    main()
