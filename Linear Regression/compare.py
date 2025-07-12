import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Your from-scratch Linear Regression class
class LinearRegression:
    def __init__(self,lr=0.001, n_iter=1000):
        self.lr=lr
        self.n_iter=n_iter
        self.weights=None
        self.bias=None
    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        for _ in range(self.n_iter):
            y_pred = np.dot(X,self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db=(1/n_samples) * np.sum(y_pred-y)
            self.weights=self.weights - self.lr*dw
            self.bias=self.bias - self.lr*db
    def predict(self, X):
        y_pred = np.dot(X,self.weights) + self.bias
        return y_pred

# Step 1: Load and prepare the data
housing = fetch_california_housing(as_frame=False)
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Step 2: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train both models and get predictions
# Train and predict with your from-scratch model
reg_scratch = LinearRegression(lr=0.01, n_iter=2000)
reg_scratch.fit(X_train_scaled, y_train)
predictions_scratch = reg_scratch.predict(X_test_scaled)
mse_scratch = mean_squared_error(y_test, predictions_scratch)
print(f"MSE (from scratch): {mse_scratch:.4f}")

# Train and predict with the scikit-learn model
reg_sklearn = SklearnLinearRegression()
reg_sklearn.fit(X_train_scaled, y_train)
predictions_sklearn = reg_sklearn.predict(X_test_scaled)
mse_sklearn = mean_squared_error(y_test, predictions_sklearn)
print(f"MSE (scikit-learn): {mse_sklearn:.4f}")


#
# =========================================================================================
# Step 4: Plot the Comparison Graph
# =========================================================================================
#

# Create a figure and axis object for the plot.
fig, ax = plt.subplots(figsize=(8, 8))

# Create a scatter plot for the scikit-learn model's predictions.
# The 'alpha' parameter makes the points semi-transparent to see overlaps.
ax.scatter(y_test, predictions_sklearn, alpha=0.7, c='blue', label='Scikit-learn Predictions')

# Create a scatter plot for your from-scratch model's predictions on the SAME axes.
# We use a different color and marker to distinguish them, though they will overlap heavily.
ax.scatter(y_test, predictions_scratch, alpha=0.5, c='orange', marker='x', label='From-Scratch Predictions')

# Add a title and labels to the plot.
ax.set_title('Scratch vs. Scikit-learn: Predicted vs. Actual')
ax.set_xlabel('Actual Values (y_test)')
ax.set_ylabel('Predicted Values')

# Add the diagonal line representing a perfect prediction.
lims = [min(y_test), max(y_test)]
ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Prediction')

# Add a legend to identify which points belong to which model.
ax.legend()
# Ensure the plot has a square aspect ratio.
ax.axis('equal')
# Display the final plot.
plt.show()