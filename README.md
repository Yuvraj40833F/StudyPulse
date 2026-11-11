# StudyPulse – Smart Academic Performance Analyzer

StudyPulse is a beginner-friendly Machine Learning project that analyses student study habits and predicts their academic performance.

## Features
- Predicts marks using Linear Regression.
- Analyses study hours, attendance, sleep and practice scores.
- Provides personalised study recommendations.
- Interactive dashboard built using Streamlit.
- Evaluates model performance using MAE and R² score.

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Streamlit

## Installation

1. Clone this repository.
2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## Dataset
The project uses synthetically generated sample data for demonstration purposes. Predictions are illustrative and are not validated against real student records.

## Project Structure

- `app.py` – Streamlit dashboard
- `model.py` – Model training and evaluation
- `main.py` – Terminal-based prototype
- `requirements.txt` – Required Python libraries

## Future Improvements
- Use real student performance data.
- Improve model accuracy and validation.
- Add visual analytics and progress tracking.
- Develop more personalised recommendations.