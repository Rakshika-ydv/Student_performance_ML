import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("student_performance_dataset.csv")
x = data[['Study_Hours_per_Week']]
y = data['Final_Exam_Score']

model = LinearRegression()
model.fit(x, y)
predicted_scores = model.predict(x)
mae = mean_absolute_error(y, predicted_scores)
mse = mean_squared_error(y, predicted_scores)
rmse = np.sqrt(mse)
r2 = r2_score(y, predicted_scores)

print("Mean Absolute Error (MAE): ", round(mae, 2))
print("Mean Squared Error (MSE): ", round(mse, 2))
print("Root Mean Squared Error (MAE): ", round(rmse, 2))
print("R^2  Score (Model Accuracy): ", round(r2, 4))

#histogram
plt.figure(figsize=(10,6))
plt.hist(data["Final_Exam_Score"], bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution Of Final Exam Scores")
plt.xlabel("Final Exam Score")
plt.ylabel("Number Of Students")
plt.grid(True)
plt.show()

#scatter + Regression line
plt.figure(figsize=(10,6))
plt.scatter(x, y, color='blue', label='Actual Scores')
plt.plot(x, predicted_scores, color="red", label="Predicted Scores (Regression Line)")
plt.title("Model Prediction VS Actual Score")
plt.xlabel("Studey Hours Per Week")
plt.ylabel("Final Output")
plt.grid(True)
plt.show()

new_hours = 9
predicted_new_score = model.predict([[new_hours]])
print(f"Predicted Final Score for {new_hours} Hours is {predicted_new_score} Score")