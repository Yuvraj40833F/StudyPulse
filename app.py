import streamlit as st
import pandas as pd
import numpy as np

from model import train_model

st.set_page_config(
    page_title="StudyPulse",
    page_icon="📚",
    layout="centered"
)

@st.cache_resource
def get_model():
    return train_model()

model, mae, r2 = get_model()

st.title("📚 StudyPulse")
st.subheader("Smart Academic Performance Analyzer")

st.write("Analyse your study habits and predict your academic performance.")

st.header("Enter Your Details")

study = st.slider("Daily Study Hours", 0.0, 15.0, 6.0)
attendance = st.slider("Attendance (%)", 0.0, 100.0, 85.0)
sleep = st.slider("Daily Sleep Hours", 0.0, 15.0, 7.0)
practice = st.slider("Practice Test Score", 0.0, 100.0, 75.0)

if study + sleep >20 and study+sleep <24 :
    st.warning("Very little time remains for other daily activities.")
elif study + sleep > 24:
    st.error("Invalid input! Study and sleep hours cannot exceed 24 combined.")

if st.button("Predict My Marks"):
    if study + sleep > 24:
        st.error("Please correct your study and sleep hours first.")
    else:
        new_student = pd.DataFrame({
            "Study_Hours": [study],
            "Attendance": [attendance],
            "Sleep_Hours": [sleep],
            "Practice_Scores": [practice]
        })

        raw_prediction = model.predict(new_student)[0]
        predicted_marks = np.clip(raw_prediction, 0, 100)

        st.success(f"Predicted Marks: {predicted_marks:.2f}/100")

        if raw_prediction > 100 or raw_prediction < 0:
            st.warning("The original prediction was outside the 0-100 range.")

        st.header("Your Study Insights")

        recommendations = []

        if attendance < 75:
            recommendations.append("Improve your attendance.")

        if practice < 50:
            recommendations.append("Attempt more practice tests.")

        if study < 3:
            recommendations.append("Try increasing your daily study time.")

        if sleep < 6:
            recommendations.append("Try maintaining a better sleep schedule.")

        if recommendations:
            for recommendation in recommendations:
                st.info(recommendation)
        else:
            st.success("Great! Keep maintaining your current habits.")

        st.header("Model Performance")
        col1, col2 = st.columns(2)

        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("R² Score", f"{r2:.3f}")
