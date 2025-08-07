import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Step 1: Generate synthetic data for simple linear regression
np.random.seed(42)
X = 2.5 * np.random.randn(100, 1) + 1.5  # Feature
y = 0.8 * X.flatten() + np.random.randn(100) * 0.5 + 1.0  # Target

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Evaluate performance
print("🔹 Simple Linear Regression")
print("R-squared:", round(r2_score(y_test, y_pred), 3))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 3))

# Step 6: Plot results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# ----------------------
# Multiple Linear Regression (with more features)
# ----------------------
# Generate synthetic data
X_multi = np.random.rand(100, 3)  # 3 features
coefs = np.array([1.5, -2.0, 3.0])
y_multi = X_multi @ coefs + np.random.randn(100) * 0.3

# Train-test split
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Train model
multi_model = LinearRegression()
multi_model.fit(X_train_multi, y_train_multi)
y_pred_multi = multi_model.predict(X_test_multi)

# Evaluation
print("\n🔸 Multiple Linear Regression")
print("R-squared:", round(r2_score(y_test_multi, y_pred_multi), 3))
print("Mean Squared Error:", round(mean_squared_error(y_test_multi, y_pred_multi), 3))
print("Model Coefficients:", multi_model.coef_)
