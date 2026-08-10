# Student Performance Prediction Using Machine Learning
# Module 6: Final AI & ML Project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("student_performance.csv")

print("=" * 60)
print("STUDENT PERFORMANCE PREDICTION")
print("=" * 60)

print("\nDataset loaded successfully!")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))


# --------------------------------------------------
# 2. Explore Dataset
# --------------------------------------------------

print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())


# --------------------------------------------------
# 3. Check and Clean Data
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows containing missing values
df = df.dropna()

print("\nDataset after cleaning:")
print("Rows:", len(df))


# --------------------------------------------------
# 4. Data Visualization
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Study_Hours"],
    df["Final_Score"]
)

plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

plt.scatter(
    df["Attendance"],
    df["Final_Score"]
)

plt.title("Attendance vs Final Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Score")
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 5. Feature Selection
# --------------------------------------------------

features = [
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Assignment_Score",
    "Sleep_Hours",
    "Participation"
]

target = "Final_Score"

X = df[features]
y = df[target]


# --------------------------------------------------
# 6. Split Data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 7. Train Machine Learning Model
# --------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# --------------------------------------------------
# 8. Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

print("\nPredicted values:")
print(np.round(y_pred, 2))

print("\nActual values:")
print(y_test.to_numpy())


# --------------------------------------------------
# 9. Evaluate Model
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


# --------------------------------------------------
# 10. Actual vs Predicted Visualization
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.title("Actual vs Predicted Final Scores")
plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 11. Predict a New Student's Score
# --------------------------------------------------

new_student = pd.DataFrame({
    "Study_Hours": [7],
    "Attendance": [85],
    "Previous_Score": [75],
    "Assignment_Score": [80],
    "Sleep_Hours": [8],
    "Participation": [70]
})

prediction = model.predict(new_student)

print("\n" + "=" * 60)
print("NEW STUDENT PREDICTION")
print("=" * 60)

print("Student information:")
print(new_student.to_string(index=False))

print(f"\nPredicted Final Score: {prediction[0]:.2f}")


print("\nProject completed successfully!")
