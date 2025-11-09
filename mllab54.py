# multiple_regression_corrected.py
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Part A (manual OLS) --------------------
hours_study = np.array([1,2,3,4,5,6,7,8], dtype=float)
hours_sleep = np.array([8,7,7,6,6,5,5,4], dtype=float)
scores      = np.array([35.2,37.9,44.3,46.8,53.1,55.9,62.2,64.0], dtype=float)

# Design matrix with intercept
X_A = np.column_stack((np.ones(len(hours_study)), hours_study, hours_sleep))
y_A = scores.reshape(-1, 1)

# Solve normal equation: beta = (X'X)^{-1} X'y
beta_A = np.linalg.inv(X_A.T @ X_A) @ (X_A.T @ y_A)
beta_A = beta_A.flatten()

# Predictions and R^2
yhat_A = X_A @ beta_A.reshape(-1,1)
ss_res_A = float(np.sum((y_A - yhat_A)**2))
ss_tot_A = float(np.sum((y_A - y_A.mean())**2))
r2_A = 1 - ss_res_A / ss_tot_A

print("---- Part A (Study/Sleep -> Score) ----")
print(f"beta0 (intercept) = {beta_A[0]:.4f}")
print(f"beta1 (Hours_Study) = {beta_A[1]:.4f}")
print(f"beta2 (Hours_Sleep) = {beta_A[2]:.4f}")
print(f"R^2 = {r2_A:.6f}")
print()

# Plot predicted vs actual
plt.figure(figsize=(6,5))
plt.scatter(y_A, yhat_A, s=60, label="Predicted vs Actual")
mn = min(float(y_A.min()), float(yhat_A.min()))
mx = max(float(y_A.max()), float(yhat_A.max()))
plt.plot([mn, mx], [mn, mx], linestyle="--", color="red", label="Ideal y = x")
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Part A: Predicted vs Actual Scores")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# -------------------- Part B (Diabetes: bmi & bp -> target) --------------------
# Uses sklearn's diabetes dataset. If sklearn is not installed, this will raise an ImportError.
try:
    from sklearn.datasets import load_diabetes
except ImportError:
    raise ImportError("sklearn is required for Part B. Install with: pip install scikit-learn")

diabetes = load_diabetes()
# Select features: bmi (col 2) and bp (col 3)
X_raw = diabetes.data[:, [2, 3]].astype(float)
y_B = diabetes.target.astype(float)

# Add intercept column
X_B = np.column_stack((np.ones(X_raw.shape[0]), X_raw))

# Solve OLS via least squares (numerically stable)
beta_B, *_ = np.linalg.lstsq(X_B, y_B, rcond=None)
# beta_B is [intercept, coef_bmi, coef_bp]

yhat_B = X_B @ beta_B
ss_res_B = np.sum((y_B - yhat_B)**2)
ss_tot_B = np.sum((y_B - y_B.mean())**2)
r2_B = 1 - ss_res_B / ss_tot_B

print("---- Part B (Diabetes: bmi & bp -> target) ----")
print(f"beta0 (intercept) = {beta_B[0]:.6f}")
print(f"beta1 (bmi)       = {beta_B[1]:.6f}")
print(f"beta2 (bp)        = {beta_B[2]:.6f}")
print(f"R^2 = {r2_B:.6f}")
print()

# Plot predicted vs actual for Part B
plt.figure(figsize=(6,5))
plt.scatter(y_B, yhat_B, s=30, alpha=0.7)
mn = min(y_B.min(), yhat_B.min())
mx = max(y_B.max(), yhat_B.max())
plt.plot([mn, mx], [mn, mx], linestyle="--", color="red")
plt.xlabel("Actual target")
plt.ylabel("Predicted target")
plt.title("Part B: Predicted vs Actual (Diabetes)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
