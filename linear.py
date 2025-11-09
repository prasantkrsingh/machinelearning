import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import numpy as np

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
