#Reference : https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py

import numpy as np
import matplotlib.pyplot as plt

import timeit

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

start = timeit.default_timer()

# #############################################################################
# Load data
boston = datasets.load_boston()
#print(boston.data.shape)
#print(boston.target)
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
#print(offset)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
#print("Train : ",X_train,"Test : ",X_test)
# #############################################################################
# Fit regression model

'''
params = {'n_estimators': 1500, 'max_depth': 8, 'min_samples_split': 10,
          'learning_rate': 0.001, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)

#print("Loss Function : ",lossCnt,lrCnt)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)
'''




cnt = 0

lossbuf = ['ls','lad','huber','quantile']
#learning_rateBuf = [0.1,0.5,0.6,0.7,0.8,0.9,0.01,0.001]
learning_rateBuf = [0.1,0.01,0.001]

for lossCnt in lossbuf:
    for lrCnt in learning_rateBuf:
        clf = ensemble.GradientBoostingRegressor(n_estimators=1500,max_depth=8,min_samples_split=10,
                                                 learning_rate=lrCnt,loss='ls')
    
        clf.fit(X_train, y_train)
        
        mse = mean_squared_error(y_test, clf.predict(X_test))
        print("MSE: %.4f" % mse)
               
        stop = timeit.default_timer()
        print('time cost: {0} s'.format(stop - start))
        
#predection=clf.predict(X_test)
#print(predection)
# #############################################################################
# Plot training deviance
'''
# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
'''
'''
# #############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, boston.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
'''