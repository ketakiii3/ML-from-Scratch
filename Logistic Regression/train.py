import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from logistic import LogisticRegression

bc=datasets.load_breast_cancer()
# Separate the features (X) and the target labels (y)
X,y=bc.data, bc.target
# Split the dataset into 80% training and 20% testing data
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=1234)

# Create an instance of the LogisticRegression classifier with a learning rate of 0.01
clf=LogisticRegression(lr=0.01)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

def accuracy(y_pred, y_test):
    # Compare predicted labels to true labels and calculate the mean of correct predictions
    return np.sum(y_pred==y_test)/len(y_test)


acc=accuracy(y_pred,y_test)
print(acc)