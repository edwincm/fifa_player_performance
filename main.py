import streamlit as st
import numpy as np
from joblib import load

# Load the saved model
model = load('static/best_rf_model.joblib')

# Title and description
st.title("Football Player Performance Predictor")
st.write("Predict the overall performance rating of a football player based on various attributes.")

# Input fields for the features
age = st.slider("Age", min_value=15, max_value=45, value=25)
potential = st.slider("Potential", min_value=40, max_value=99, value=80)
value = st.number_input("Value (in millions)", min_value=0.0, max_value=300.0, value=50.0, step=0.1)
international_reputation = st.number_input("International Reputation", min_value=1, max_value=5, value=3)

# Physical attributes
st.subheader("Physical Attributes")
stamina = st.number_input("Stamina", min_value=0, max_value=100, value=60)
strength = st.number_input("Strength", min_value=0, max_value=100, value=70)
jumping = st.number_input("Jumping", min_value=0, max_value=100, value=65)
agility = st.number_input("Agility", min_value=0, max_value=100, value=70)
physical_score = np.mean([stamina, strength, jumping, agility])

# Passing attributes
st.subheader("Passing Attributes")
crossing = st.number_input("Crossing", min_value=0, max_value=100, value=75)
short_passing = st.number_input("Short Passing", min_value=0, max_value=100, value=80)
vision = st.number_input("Vision", min_value=0, max_value=100, value=70)
passing_score = np.mean([crossing, short_passing, vision])

# Mental attributes
st.subheader("Mental Attributes")
composure = st.number_input("Composure", min_value=0, max_value=100, value=65)
interceptions = st.number_input("Interceptions", min_value=0, max_value=100, value=70)
reactions = st.number_input("Reactions", min_value=0, max_value=100, value=75)
mental_score = np.mean([composure, interceptions, reactions])

# Shooting attributes
st.subheader("Shooting Attributes")
shot_power = st.number_input("Shot Power", min_value=0, max_value=100, value=80)
finishing = st.number_input("Finishing", min_value=0, max_value=100, value=75)
heading_accuracy = st.number_input("Heading Accuracy", min_value=0, max_value=100, value=70)
shooting_score = np.mean([shot_power, finishing, heading_accuracy])

# Other attributes
preferred_foot = st.selectbox("Preferred Foot", options=["Left", "Right"])
attacking_wr = st.selectbox("Attacking Work Rate", options=["Low", "Medium", "High"])
defensive_wr = st.selectbox("Defensive Work Rate", options=["Low", "Medium", "High"])
height_range = st.selectbox("Height Range", options=["170-", "170-185", "185+", "190+"])

# One-hot encoded attributes
st.subheader("Position Attributes")
position = st.selectbox("Position", options=["Attacker", "Defender", "Goalkeeper", "Midfielder", "Substitute"])
position_group_defender = 1 if position == "Defender" else 0
position_group_goalkeeper = 1 if position == "Goalkeeper" else 0
position_group_midfielder = 1 if position == "Midfielder" else 0
position_group_substitute = 1 if position == "Substitute" else 0

st.subheader("Build Type")
build_type = st.selectbox("Build Type", options=["Lean", "Normal", "Stocky"])
build_type_normal = 1 if build_type == "Normal" else 0
build_type_stocky = 1 if build_type == "Stocky" else 0

# Convert categorical features to numerical
preferred_foot = 0 if preferred_foot == "Left" else 1
attacking_wr = {"Low": 1, "Medium": 2, "High": 3}[attacking_wr]
defensive_wr = {"Low": 1, "Medium": 2, "High": 3}[defensive_wr]
height_range = {"170-": 1, "170-185": 2, "185+": 3, "190+": 4}[height_range]

# Combine all features
features = [
    age,
    potential,
    value,
    international_reputation,
    physical_score,
    passing_score,
    mental_score,
    shooting_score,
    preferred_foot,
    attacking_wr,
    defensive_wr,
    height_range,
    position_group_defender,
    position_group_goalkeeper,
    position_group_midfielder,
    position_group_substitute,
    build_type_normal,
    build_type_stocky,
]

# Prediction
if st.button("Predict"):
    prediction = model.predict([features])[0]
    st.success(f"Predicted Overall Rating: {prediction:.2f}")
