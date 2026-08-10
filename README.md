# Student Performance Prediction Using Machine Learning

## Module 6: Final AI & ML Project

This project is an end-to-end Machine Learning application developed as the final project of my AI and Machine Learning internship.

The project analyzes student-related factors such as study hours, attendance, previous performance, assignment scores, and sleep hours to predict a student's expected final score.

## Project Objectives

* Build a complete Machine Learning project.
* Work with a structured dataset.
* Perform data exploration and cleaning.
* Visualize relationships within the dataset.
* Select relevant features for prediction.
* Train a Machine Learning regression model.
* Evaluate model performance.
* Make predictions for new student data.
* Document and publish the project on GitHub.

## Machine Learning Workflow

The project follows a complete Machine Learning workflow:

```text
Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Data Visualization
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Prediction
```

## Dataset

The dataset contains information about student performance and learning-related factors.

### Features

* Study Hours
* Attendance
* Previous Score
* Assignment Score
* Sleep Hours
* Participation

### Target

* Final Score

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* GitHub

## Machine Learning Model

A **Linear Regression** model is used to predict the final student score.

Linear Regression is a supervised Machine Learning algorithm used to predict a continuous numerical value based on one or more input features.

## Project Structure

```text
module-6-student-performance-ml/
│
├── README.md
├── student_performance_prediction.py
├── student_performance.csv
└── requirements.txt
```

## Key Steps

### 1. Data Loading

The dataset is loaded using Pandas.

### 2. Data Exploration

The dataset is examined using:

* `head()`
* `info()`
* `describe()`
* Missing-value checks

### 3. Data Cleaning

Missing or invalid values are identified and handled before training the model.

### 4. Data Visualization

Matplotlib is used to visualize relationships between study-related factors and student performance.

### 5. Feature Selection

Relevant numerical features are selected as inputs for the Machine Learning model.

### 6. Train/Test Split

The dataset is divided into training and testing data.

### 7. Model Training

A Linear Regression model is trained using the training dataset.

### 8. Model Evaluation

The model is evaluated using metrics such as:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

### 9. Prediction

The trained model is used to predict the final score for new student data.

## Expected Learning Outcomes

Through this project, I applied the concepts learned throughout the internship, including:

* Python programming
* Data manipulation with Pandas
* Numerical analysis with NumPy
* Data visualization with Matplotlib
* Supervised Machine Learning
* Regression
* Model evaluation
* GitHub project documentation

## Conclusion

This project demonstrates an end-to-end Machine Learning workflow, from preparing and analyzing data to training a model and generating predictions.

It provided practical experience in applying Python, data analysis, visualization, and Machine Learning concepts to a real-world-inspired problem.

The project also helped strengthen my understanding of how Machine Learning models can be developed, evaluated, and used to make predictions from structured data.
