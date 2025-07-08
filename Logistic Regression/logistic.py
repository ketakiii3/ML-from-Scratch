import numpy as np

# Define the sigmoid function, which squashes values between 0 and 1
def sigmoid(x):
    # Clip the input values to avoid overflow errors in np.exp for very large or small numbers
    x=np.clip(x, -500, 500)
    # Apply the sigmoid formula: 1 / (1 + e^(-x))
    return 1/(1+np.exp(-x))

class LogisticRegression:
    # Initialize the classifier with hyperparameters
    def __init__(self, lr=0.001, n_iter=1000):
        # Set the learning rate for gradient descent
        self.lr=lr
        self.n_iter=n_iter
        # Initialize weights and bias to None; they will be set during fitting
        self.weights=None
        self.bias=None
        
    def fit(self, X, y):
        # Get the number of samples and features from the input data X
        n_samples, n_features=X.shape
        # Initialize the weights as a zero vector with a length equal to the number of features
        self.weights=np.zeros(n_features)
        self.bias=0
        
        for _ in range(self.n_iter):
            # Calculate the linear combination of inputs and weights (the log-odds)
            linear_pred=np.dot(X, self.weights)+self.bias
            # Apply the sigmoid function to get the predicted probabilities
            predictions=sigmoid(linear_pred)
            
            # Calculate the gradient of the cost function with respect to weights and bias
            dw=(1/n_samples) * np.dot(X.T, (predictions-y))
            db=(1/n_samples) * np.sum(predictions-y)
            
            # Update the weights and bias using the gradient descent rule
            self.weights=self.weights-self.lr*dw
            self.bias=self.bias-self.lr*db
        
    def predict(self, X):
        # Calculate the linear combination for the test data
        linear_pred=np.dot(X, self.weights)+self.bias
        # Get the predicted probabilities using the sigmoid function
        y_pred=sigmoid(linear_pred)
        
        # Convert probabilities to class labels (0 or 1) based on a 0.5 threshold
        class_pred=[0 if y<=0.5 else 1 for y in y_pred]
        return class_pred



