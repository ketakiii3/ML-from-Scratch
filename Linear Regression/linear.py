import numpy as np

#Defining Linear Regression Class
class LinearRegression:
    # This is the constructor for the class, which is called when a new object is created.
    # It initializes the model's hyperparameters.
    def __init__(self,lr=0.001, n_iter=1000):
        # Set the learning rate (lr), which controls the step size during gradient descent.
        self.lr=lr
        # Set the number of iterations (n_iter), which is the number of times the model will go through the training data.
        self.n_iter=n_iter
        # Initialize weights to None. They will be set during the training process.
        self.weights=None
        # Initialize bias to None. It will also be set during training.
        self.bias=None
        
    # This method trains the model using the provided training data (X, y).
    def fit(self,X,y):
        # Get the number of data points (samples) and features from the input data X.
        n_samples, n_features = X.shape
        # Initialize the model's weights as a NumPy array of zeros. The size is equal to the number of features.
        self.weights=np.zeros(n_features)
        # Initialize the bias to 0.
        self.bias=0
        
        # Start the training loop, which will run for 'n_iter' iterations
        for _ in range(self.n_iter):
            # Calculate the predicted values (y_pred) using the current weights and bias.
            # This is the linear equation: y = w*X + b
            y_pred = np.dot(X,self.weights) + self.bias
            
            # Calculate the gradient for the weights (dw). This is the partial derivative of the cost function (MSE) with respect to the weights.
            # It tells us how to change the weights to reduce the error.
            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            # Calculate the gradient for the bias (db). This is the partial derivative of the cost function with respect to the bias.
            db=(1/n_samples) * np.sum(y_pred-y)
            
            # Update the weights and bias by taking a step in the opposite direction of the gradient.
            # The size of the step is determined by the learning rate (lr)
            self.weights=self.weights - self.lr*dw
            self.bias=self.bias - self.lr*db
        
    # This method makes predictions on new, unseen data (X).    
    def predict(self, X):
        # Calculate the predicted values using the final learned weights and bias.
        y_pred = np.dot(X,self.weights) + self.bias
        return y_pred