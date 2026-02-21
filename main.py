import numpy as np
import pandas as pd

from utils.preprocessing import load_and_preprocess
from models.linear_model import run_linear_regression
from models.knn_model import run_knn



# Load & Preprocess


X_train, X_test, y_train, y_test, feature_columns = load_and_preprocess(
    "data/student-mat.csv"
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)



# Train Models


linear_model = run_linear_regression(X_train, X_test, y_train, y_test)
knn_model = run_knn(X_train, X_test, y_train, y_test)



#  User Input Prediction


print("\n=== Student Grade Prediction ===")

studytime = int(input("Enter study time (1-4): "))
failures = int(input("Enter number of past failures (0-3): "))
absences = int(input("Enter number of absences: "))
G1 = int(input("Enter first period grade (0-20): "))
G2 = int(input("Enter second period grade (0-20): "))

new_student = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

new_student["studytime"] = studytime
new_student["failures"] = failures
new_student["absences"] = absences
new_student["G1"] = G1
new_student["G2"] = G2

prediction = linear_model.predict(new_student)

print("\nPredicted Final Grade (G3):", round(prediction[0], 2))
