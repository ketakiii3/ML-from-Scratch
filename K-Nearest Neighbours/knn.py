import numpy as np
from collections import Counter

def euclidean_distance(x1,x2):
    # Formula: square root of the sum of squared differences between each element.
    # i.e., distance = sqrt((x1[0] - x2[0])² + (x1[1] - x2[1])² + ... + (x1[n] - x2[n])²)
    # It represents the straight-line distance between two points in Euclidean space.
    distance = np.sqrt(np.sum((x1-x2)**2))
    # alternative L2 norm
    # np.linalg.norm(x1 - x2)
    return distance

class KNN:
    # The constructor method, initialized with a value for k (default is 3)
    def __init__(self, k=3):
        self.k=k
        
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
        
        
    def predict(self,X):
        # Use a list comprehension to call the helper method for each data point in X
        predictions=[self.predict_helper(x) for x in X]
        return predictions
        
    # A private helper method to predict the class for a single data point (x)
    def predict_helper(self,x):
        # Use a list comprehension to compute the Euclidean distance from the input 'x' to every point in the training set
        distances=[euclidean_distance(x,x_train) for x_train in self.X_train]
        
        # Get the indices of the 'k' smallest distances using np.argsort(), which returns the indices that would sort an array
        k_indices=np.argsort(distances)[:self.k]
        # Get the labels of the 'k' nearest neighbors using the indices found above
        k_nearest_labels=[self.y_train[i] for i in k_indices]
        
        # Use Counter to count the occurrences of each class label among the nearest neighbors and find the most common one(s)
        most_common = Counter(k_nearest_labels).most_common()
        # Return the most common label as the prediction. most_common() returns a list of (element, count) tuples.
        return int(most_common[0][0])