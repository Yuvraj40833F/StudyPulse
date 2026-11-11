import numpy as np
import pandas as pd
    
print("Welcome to StudyPulse!")
print("Libraries imported successfully!")

np.random.seed(42)

students=50
study_hours=np.random.uniform(1, 10 , students)
attendance=np.random.uniform(50, 100 , students)
sleep_hours=np.random.uniform(4, 9 , students)
practice_scores=np.random.uniform(20, 100 , students)

marks=(
    study_hours*4 +
    attendance*0.2 +
    sleep_hours*1.5 +
    practice_scores*0.3+
    np.random.normal(0,5, students)
)

marks=np.clip(marks, 0 , 100)

data= pd.DataFrame({
    "Study_Hours" : study_hours,
    "Attendance" : attendance,
    "Sleep_Hours" : sleep_hours, 
    "Practice_Scores" : practice_scores,
    "Marks" : marks
})

print("\nSTUDYPULSE - STUDENT DATA")
print(data.head(10))

print("\nDataset Information")
print(data.shape)

#-----------------------------------------------------------

from sklearn.model_selection import train_test_split

x= data[[
    "Study_Hours",
    "Attendance", 
    "Sleep_Hours",
    "Practice_Scores"
]]

y=data["Marks"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print("Training_Records", len(x_train))
print("Testing_Records", len(x_test))

#-----------------------------------------------------------

from sklearn.linear_model import LinearRegression

model= LinearRegression()
model.fit(x_train, y_train)

predictions=model.predict(x_test)

print("\nActual Marks:")
print(y_test.values)

print("\nPredicted Marks:")
print(predictions)

#------------------------------------------------------------

from sklearn.metrics import mean_absolute_error, r2_score

mae= mean_absolute_error(y_test, predictions)
r2= r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 3))

# student_study = 6
# student_attendance = 85
# student_sleep = 7
# student_practice = 75

student_study = float(input("Enter daily study hours (0-24): "))
student_attendance = float(input("Enter attendance percentage (0-100): "))
student_sleep = float(input("Enter daily sleep hours (0-24): "))
student_practice = float(input("Enter practice test score (0-100): "))

if not (0 <= student_study <= 24):
    print("Invalid study hours! Enter a value between 0 and 24.")
    exit()
if not (0 <= student_attendance <= 100):
    print("Invalid attendance! Enter a value between 0 and 100.")
    exit()
if not (0 <= student_sleep <= 24):
    print("Invalid sleep hours! Enter a value between 0 and 24.")
    exit()
if not (0 <= student_practice <= 100):
    print("Invalid practice score! Enter a value between 0 and 100.")
    exit()

if student_study + student_sleep > 24:
    print("Invalid input! Study and sleep hours cannot exceed 24 combined.")
    exit()

if student_study + student_sleep > 20:
    print("Warning: Very little time remains for other daily activities.")

new_student = pd.DataFrame({
    "Study_Hours": [student_study],
    "Attendance": [student_attendance],
    "Sleep_Hours": [student_sleep],
    "Practice_Scores": [student_practice]
})

raw_prediction = model.predict(new_student)[0]
predicted_marks = np.clip(raw_prediction, 0, 100)
print("\nNEW STUDENT ANALYSIS")
print("Predicted Marks:", round(predicted_marks, 2))

if raw_prediction > 100 or raw_prediction < 0:
    print("Note: Original prediction was outside the 0-100 marks range.")


recommendations = []
if student_attendance < 75:
    recommendations.append("Improve your attendance.")
if student_practice < 50:
    recommendations.append("Attempt more practice tests.")
if student_study < 3:
    recommendations.append("Increase your daily study time.")
if student_sleep < 6:
    recommendations.append("Improve your sleep schedule.")
if len(recommendations) == 0:
    recommendations.append("Great! Keep maintaining your current habits.")
print("\nPERSONALIZED RECOMMENDATIONS")
for recommendation in recommendations:
    print("-", recommendation)

#--------------------------------------------------------------------

