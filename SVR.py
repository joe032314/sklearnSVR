
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.svm import SVR
import numpy as np
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

svr = SVR(kernel = 'rbf', C = 1.0, gamma = 0.1, epsilon=0.1)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

model_color = ['m', 'c', 'g']
plt.figure(2, figsize=(8, 6))
# Plot the training points
for i in enumerate(model_color):
    plt.plot(X,svr.fit(X,Y).predict(X), color = model_color[i])
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

plt.xlabel('Sepal length') #標記X軸名稱
plt.ylabel('Sepal width')  #標記Y軸名稱
plt.xlim(x_min, x_max) #設定X軸的範圍
plt.ylim(y_min, y_max) #設定Y軸的範圍
plt.xticks()
plt.yticks()
plt.show()