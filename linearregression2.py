#Predict Student exam scores based on the number of hours they student.Use Linear Regression to predict the score of a Student who studies for 9.5 hours.
#print the slope and intercept of the regression line

# Predict Student Exam Scores using Linear Regression (without libraries)

# Dataset
# Linear Regression from Scratch (No Libraries)

import matplotlib.pyplot as plt

# Dataset
hours = [2.5, 5.1, 3.2, 8.5, 3.5,
         1.5, 9.2, 5.5, 8.3, 2.7,
         7.7, 5.9, 4.5, 3.3, 1.1,
         8.9, 2.5, 1.9, 6.1, 7.4]

scores = [21, 47, 27, 75, 30,
          20, 88, 60, 81, 25,
          85, 62, 41, 42, 17,
          95, 30, 24, 67, 69]

# Step 1: Calculate means
n = len(hours)
mean_x = sum(hours) / n
mean_y = sum(scores) / n

# Step 2: Calculate slope (m) and intercept (c)
numerator = 0
denominator = 0
for i in range(n):
    numerator += (hours[i] - mean_x) * (scores[i] - mean_y)
    denominator += (hours[i] - mean_x) ** 2

m = numerator / denominator   # slope
c = mean_y - m * mean_x       # intercept

print("Slope (m):", m)
print("Intercept (c):", c)

# Step 3: Predict score for 9.5 hours
x_new = 9.5
y_pred = m * x_new + c
print(f"Predicted score for {x_new} hours of study: {y_pred}")

# Step 4: Plotting dataset and regression line
plt.scatter(hours, scores, color="blue", label="Data Points")
y_line = [m * x + c for x in hours]
plt.plot(hours, y_line, color="red", label="Regression Line")

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Student Scores Prediction using Linear Regression")
plt.legend()
plt.show()
