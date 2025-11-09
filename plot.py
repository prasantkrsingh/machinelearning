#plot the regression line with the dataset
import matplotlib.pyplot as plt


hours = [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7,
         7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4]
scores = [21, 47, 27, 75, 30, 20, 88, 60, 81, 25,
          85, 62, 41, 42, 17, 95, 30, 24, 67, 69]


n = len(hours)


sum_x = sum(hours)
sum_y = sum(scores)
sum_xy = sum(x * y for x, y in zip(hours, scores))
sum_x2 = sum(x * x for x in hours)


m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - (sum_x ** 2))
c = (sum_y - m * sum_x) / n

print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")

y_pred_line = [m * x + c for x in hours]


x_new = 9.5
y_new = m * x_new + c
print(f"Predicted Score for {x_new} hours = {y_new}")


plt.scatter(hours, scores, color="blue", label="Actual Data")


plt.plot(hours, y_pred_line, color="red", label="Regression Line")


plt.scatter(x_new, y_new, color="green", s=100, marker="x", label=f"Prediction (9.5 hrs, {y_new:.2f})")

plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.title("Hours vs Scores (Linear Regression)")
plt.legend()
plt.grid(True)
plt.show()
