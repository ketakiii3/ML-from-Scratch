import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
from knn import KNN

iris=datasets.load_iris()
X,y=iris.data, iris.target

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=1234)

clf=KNN(k=5)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)

print(predictions)

acc=np.sum(predictions==y_test)/len(y_test)
print(acc)

plt.figure()
# Create a scatter plot of the 3rd and 4th features (petal length and petal width) of the entire dataset
# 'c=y' colors the points according to their true class label
# 'cmap=cmap' uses the custom colormap defined earlier
# 'edgecolors='k'' gives each point a black border for better visibility
plt.scatter(X[:,2], X[:,3], c=y, cmap=cmap, edgecolors='k', s=20)
plt.show()