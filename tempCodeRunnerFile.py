import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import numpy as np

def minkowski_distance(x, y, p):
    if len(x) != len(y):
        raise ValueError("Vectors must be of the same length")

    
    distance = 0
    for i in range(len(x)):
        distance += abs(x[i] - y[i]) ** p

    return distance ** (1 / p)



x = [2, 3, 5, 7, 5, 6, 3, 1, 19, 11, 12, 16, 15, 5, 2, 1, 7, 2, 4, 5, 6, 12, 11, 2, 3]
y = [1, 1, 2, 3, 2, 3, 5, 7, 5, 6, 3, 1, 19, 11, 10, 16, 14, 5, 2, 1, 8, 2, 4, 5, 2]


l1 = minkowski_distance(x, y, 1) 
l2 = minkowski_distance(x, y, 2)  
l3 = minkowski_distance(x, y, 3)

# Print results
print("L1 Distance (Manhattan):", l1)
print("L2 Distance (Euclidean):", l2)
print("L3 Distance:", l3)

# 1. Load dataset
diabetes = load_diabetes()
X = diabetes.data[:, np.newaxis, 2]  # BMI column (index 2)
y = diabetes.target                  # Disease progression (target)

# 2. Fit Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Get slope (coefficient) and intercept
slope = model.coef_[0]
intercept = model.intercept_
print("Slope (Coefficient):", slope)
print("Intercept:", intercept)

# 3. Predict for BMI = 0.05
bmi_value = 0.05
prediction = model.predict([[bmi_value]])
print(f"Predicted disease progression for BMI={bmi_value}:", prediction[0])

# 4. Plot BMI vs Target with regression line
plt.figure(figsize=(8,6))
plt.scatter(X, y, color="blue", alpha=0.5, label="Data points")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression Line")
plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Linear Regression: BMI vs Disease Progression")
plt.legend()
plt.grid(True)
plt.show()

