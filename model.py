
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def train_model():

    np.random.seed(42)

    students = 50

    study_hours = np.random.uniform(1, 10, students)
    attendance = np.random.uniform(50, 100, students)
    sleep_hours = np.random.uniform(4, 9, students)
    practice_score = np.random.uniform(20, 100, students)

    marks = (
        study_hours * 4
        + attendance * 0.2
        + sleep_hours * 1.5
        + practice_score * 0.3
        + np.random.normal(0, 5, students)
    )

    marks = np.clip(marks, 0, 100)

    data = pd.DataFrame({
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Sleep_Hours": sleep_hours,
        "Practice_Scores": practice_score,
        "Marks": marks
    })

    X = data[[
        "Study_Hours",
        "Attendance",
        "Sleep_Hours",
        "Practice_Scores"
    ]]

    y = data["Marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mae, r2
