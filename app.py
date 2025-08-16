import streamlit as st
import pandas as pd
import joblib
import os
import random

def get_motivational_quote():
    quotes = [
        "Choose a job you love, and you’ll never have to work a day in your life.",
        "Hard work beats talent when talent doesn’t work hard.",
        "Opportunities don’t happen, you create them.",
        "Don’t watch the clock; do what it does. Keep going.",
        "Success usually comes to those who are too busy to be looking for it.",
        "The future depends on what you do today."
    ]
    return random.choice(quotes)

def generate_career_tip(age, gender, experience, education, job_title):
    tips = []

    # Rule 1: Based on experience
    if experience < 3:
        tips.append("Focus on building strong foundations through online courses (Coursera, Udemy, etc.).")
    elif 3 <= experience < 7:
        tips.append("Consider getting certifications relevant to your field to stand out (e.g., PMP, AWS, Data Science).")
    else:
        tips.append("Work on leadership and mentoring skills. Consider management or advanced specialization.")

    # Rule 2: Based on education
    if "Bachelor" in education:
        tips.append("Pursuing a Master's or specialized certification can accelerate your career.")
    elif "Master" in education:
        tips.append("Look for niche certifications or cross-functional skills (like product management).")

    # Rule 3: Based on job title
    if "Engineer" in job_title:
        tips.append("Stay updated with the latest tools and frameworks in your domain.")
    elif "Manager" in job_title:
        tips.append("Focus on people management and communication skills to climb further.")

    return (tips)

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("salary_prediction_model.pkl")

model = load_model()

# Load dataset (only to grab unique values for dropdowns)
df = pd.read_csv("Salary Data.csv").dropna()

# ----------------------------
# Streamlit App UI
# ----------------------------
st.set_page_config(page_title="Salary Prediction App", page_icon="💰", layout="centered")
st.title("💼 Salary Prediction App")
st.write("Predict salary based on Age, Gender, Years of Experience, Education Level, and Job Title.")

# Sidebar inputs
st.sidebar.header("Input Features")

age = st.sidebar.slider("Age", int(df['Age'].min()), int(df['Age'].max()), 30)
experience = st.sidebar.slider("Years of Experience", 0, int(df['Years of Experience'].max()), 5)
gender = st.sidebar.selectbox("Gender", df['Gender'].unique())
education = st.sidebar.selectbox("Education Level", df['Education Level'].unique())
job_title = st.sidebar.selectbox("Job Title", sorted(df['Job Title'].unique()))

# Prepare input
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Years of Experience': [experience],
    'Education Level': [education],
    'Job Title': [job_title]
})

# Predict
if st.sidebar.button("Predict Salary"):
    predicted_salary = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ${predicted_salary:,.2f}")
    
    tips = generate_career_tip(age, gender, experience, education, job_title)
    for tip in tips:
      st.info(tip)
    
    # Motivational Quote
    st.subheader("")
    st.info(get_motivational_quote())