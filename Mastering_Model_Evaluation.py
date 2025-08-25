#Sklearn Matrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#True Answers(what actually happen)
y_true = [1, 0, 1, 1, 0, 1, 0]
#model prediction(what it guessed)
y_pred = [1, 0, 1, 0, 0, 1, 1]
#evaluation
print("Accuarcy", accuracy_score(y_true, y_pred))
print("Prediction", precision_score(y_true, y_pred))
print("Recall", recall_score(y_true, y_pred))
print("F1 Score", f1_score(y_true, y_pred))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_true = [1,0,1,1,0,1,0,0,1,0]
y_pred = [1,0,1,0,0,1,1,0,1,0]
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix")
print(cm)

"""
MEAN ABSOLUTE ERROR(MAE)
1-take the mistake difference
2-remove the minus sign
3-add
4-divide

MEAN SQUARED ERROR(MSE)
1-mistakes square them
2-add
3-divide total

ROOT MEAN SQUARED ERROR(RMSE)
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
#real scores
real_scores = [90, 80, 60, 100]
#model_guess
predicted_scores = [85, 70, 70, 95]
mae = mean_absolute_error(real_scores, predicted_scores)
mse = mean_squared_error(real_scores, predicted_scores)
rmse = np.sqrt(mse)
print("MAE: On average off by: ", mae)
print("MSE: Squared Mistake Value: ", mse)
print("RMSE: Final Realistic Error: ", rmse)