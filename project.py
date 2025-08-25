import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
data = pd.read_csv("Student.csv")
x = data[['Hours']]
y = data['Score']
model = LinearRegression()
model.fit(x, y)
predicted_score = model.predict(x)
mae = mean_absolute_error(y, predicted_score)
mse = mean_squared_error(y, predicted_score)
rmse = np.sqrt(mse)
print("Mean Absolute Error (MAE): ", mae)
print("Mean Squared Error (MSE): ", mse)
print("Root Mean Squared Error (MAE): ", rmse)

new_prediction = model.predict([[7]])
print(f"Predicted Score for 7 hour", new_prediction)