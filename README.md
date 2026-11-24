# Student Performance Analysis & Prediction

## Overview
A beginner-friendly Data Science capstone that explores factors associated with student results and demonstrates a regression model for estimating final scores.

## Dataset
`student_performance.csv` contains 300 **synthetic demonstration records**. It is generated for learning and software testing, not collected from real students. Do not use its results to make real academic decisions.

## Features
- Data inspection, missing-value checks and duplicate removal
- Descriptive statistics and correlation analysis
- Exploratory charts
- Random Forest regression
- Evaluation using MAE, RMSE and R-squared
- Feature-importance visualization

## Setup
```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
pip install -r requirements.txt
python main.py
```

## Project structure
- `student_performance.csv` — sample dataset
- `main.py` — analysis, model training and chart generation
- `requirements.txt` — required Python packages
- `report.docx` — internship capstone report

## Important limitation
The data is synthetic and the model demonstrates a workflow only. For real research, replace it with an authorized, representative dataset, document its source and consent/privacy safeguards, and retrain and validate the model.
