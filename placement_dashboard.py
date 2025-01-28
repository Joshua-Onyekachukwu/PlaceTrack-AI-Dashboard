import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load your pre-trained model and scaler (if saved)
model = joblib.load('placement_model.pkl')  # Ensure you have the model saved as 'placement_model.pkl'


# scaler = joblib.load('scaler.pkl')  # Load the scaler if you saved it

# Function to predict placement status based on user input
def predict_placement(input_data):
    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform([input_data])  # Use transform instead of fit_transform
    prediction = model.predict(input_scaled)

    return prediction[0]


# Streamlit UI for input form
st.title("Placement Prediction Dashboard")

st.subheader("Enter student details to predict placement status:")

# Input fields for user data
cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)  # CGPA range from 0 to 10
internships = st.number_input("Number of Internships", 0, 10, 0)
projects = st.number_input("Number of Projects", 0, 10, 0)
certifications = st.number_input("Number of Certifications/Workshops", 0, 10, 0)
aptitude_score = st.slider("Aptitude Test Score", 0, 100, 70)  # Assuming score is out of 100
soft_skill_rating = st.slider("Soft Skill Rating", 1, 5, 3)  # Rating from 1 to 5
extra_curricular = st.slider("Extra Curricular Rating", 1, 5, 3)  # Rating from 1 to 5
placement_training = st.slider("Placement Training", 0, 1, 0)  # 0 = No, 1 = Yes
ssc_marks = st.slider("Senior Secondary Marks", 0, 100, 80)  # Marks out of 100
hsc_marks = st.slider("Higher Secondary Marks", 0, 100, 80)  # Marks out of 100

# Collecting user input into a list (to be passed to the prediction function)
input_data = [cgpa, internships, projects, certifications, aptitude_score, soft_skill_rating,
              extra_curricular, placement_training, ssc_marks, hsc_marks]

# Prediction when user submits the form
if st.button("Predict Placement Status"):
    placement_status = predict_placement(input_data)

    # Display the result
    if placement_status == 1:
        st.success("The student is likely to be Placed!")
    else:
        st.error("The student is likely to be Not Placed.")

