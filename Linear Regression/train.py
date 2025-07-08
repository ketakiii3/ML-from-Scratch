import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets # To generate a synthetic regression dataset.
import matplotlib.pyplot as plt
from linear import LinearRegression

# Create a synthetic dataset for regression.
# n_samples=100: generates 100 data points.
# n_features=1: each data point has one feature.
# noise=20: adds random noise to the data to make it more realistic.
# random_state=4: ensures that the same "random" data is generated every time the script is run.
X,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
# Split the data into training and testing sets.
# X_train, y_train: data used to train the model.
# X_test, y_test: data used to evaluate the model's performance on unseen data.
# test_size=0.2: 20% of the data is reserved for testing, 80% for training.
# random_state=1234: ensures the split is reproducible.
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=1234)

# Create an instance of your LinearRegression model with a learning rate of 0.01.
reg = LinearRegression(lr=0.01)
# Train the model using the training data. The fit method will update the model's weights and bias.
reg.fit(X_train, y_train)
# Make predictions on the test set using the trained model.
predictions=reg.predict(X_test)

# Define a function to calculate the Mean Squared Error (MSE).
# MSE measures the average of the squares of the errors.
# (i.e The average squared difference between the estimated values and the actual value.)
def mse(y_test, predictions):
    # The formula is the mean of (actual_value - predicted_value)^2
    return np.mean((y_test-predictions)**2)

# Calculate the MSE for our model's predictions on the test set
mse=mse(y_test, predictions)
# Print the calculated MSE. A lower value indicates a better fit.
print(mse)

# To draw the regression line, make predictions on the entire range of X values.
y_pred_line=reg.predict(X)
# Get a colormap from matplotlib to use for the plot points.
cmap=plt.get_cmap('viridis')
# Create a figure object with a specific size (8 inches wide, 6 inches tall) for our plot.
fig=plt.figure(figsize=(8,6))
# Create a scatter plot of the training data points.
m1=plt.scatter(X_train,y_train,color=cmap(0.9),s=10)
# Create a scatter plot of the testing data points, using a different color.
m2=plt.scatter(X_test,y_test,color=cmap(0.5),s=10)
# Plot the regression line calculated by the model and display the final plot.
plt.plot(X,y_pred_line,color='black', linewidth=2, label='Prediction')
plt.show()
