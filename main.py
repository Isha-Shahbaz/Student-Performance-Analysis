"""
Student Performance Analysis & Prediction
Run: pip install -r requirements.txt
     python main.py
Dataset: student_performance.csv (synthetic demonstration data)
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("student_performance.csv")

# 1. Inspect and prepare data
print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
df = df.drop_duplicates().dropna()

# 2. Exploratory data analysis
print("\nDescriptive statistics:\n", df.describe().round(2))
print("\nCorrelation with final score:\n",
      df.corr(numeric_only=True)["final_score"].sort_values(ascending=False))

plt.figure(figsize=(7, 5))
plt.scatter(df["study_hours_per_day"], df["final_score"], alpha=0.6)
plt.xlabel("Study hours per day")
plt.ylabel("Final score")
plt.title("Study Time vs Final Score")
plt.tight_layout()
plt.savefig("study_hours_vs_score.png")
plt.close()

plt.figure(figsize=(7, 5))
df["final_score"].hist(bins=15)
plt.xlabel("Final score")
plt.ylabel("Number of students")
plt.title("Distribution of Final Scores")
plt.tight_layout()
plt.savefig("score_distribution.png")
plt.close()

# 3. Train and evaluate a regression model
X = df.drop(columns=["final_score"])
y = df["final_score"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)
print(f"\nMAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.3f}")

importance = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)
print("\nFeature importance:\n", importance.round(3))

plt.figure(figsize=(8, 5))
importance.sort_values().plot(kind="barh")
plt.xlabel("Feature importance")
plt.title("Factors Used by the Prediction Model")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

print("\nCompleted. Charts saved in the project folder.")
