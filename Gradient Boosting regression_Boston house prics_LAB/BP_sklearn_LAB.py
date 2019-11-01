#Reference : https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import timeit
import time

from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV #GridSearchCV網格搜尋;RandomizedSearch隨機搜尋
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

'''Function'''
def std_s(data):
    data2 = np.square( data - np.average(data) )
    A = np.sqrt(sum(data2)/(len(data)-1))
    return A

def avedev(data):
    return (sum(abs(data - np.average(data)))/len(data))

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='b', linestyle='-',label="Mean Diff")
    plt.axhline(md + 1.96*sd, color='r', linestyle='--',label="Diff + 1.96 x sd")
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--',label="Diff - 1.96 x sd")
    plt.legend(loc='best',fontsize=10)

'''Main()'''
start = timeit.default_timer()
path = os.getcwd()
#print(path)
# =============================================================================
'''Deta Information'''
# BPdata            = 2543
# Input(Feature)    = (AGE、BMI、HR、PTT)
# Predection        = (SBP) (DBP)
# offset            = 0.9 ; #Train & Test = 9:1 
# 
# X           = Train Data (AGE、BMI、HR、PTT) 
# Y           = Test  Data (SBP) (DBP)
# 
# Train Data  = 2288
# Test  Data  = 255
# 
# X_train     = Train Set Feature
# y_train     =
# X_train     = Test Set Feature
# y_test      = Test Set Answer
# =============================================================================
'''Load BP Data'''
Bpdata        = pd.read_csv('train_pd.csv')    
#print(Bpdata.shape)
#print(Bpdata.columns)
traindata     = Bpdata.iloc[:,0:4].values
#print(traindata.size)
SBP_target    = Bpdata.iloc[:,4].values
DBP_target    = Bpdata.iloc[:,5].values
#print(DBP_target)
X, y   = shuffle(traindata,SBP_target,random_state=8) #Random
#X, y   = shuffle(traindata,DBP_target,random_state=8) #Random
offset = int(X.shape[0] * 0.9) #Train & Test = 9:1
#print(X.shape[0])
#print(offset)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test   = X[offset:], y[offset:]

'''Store TEST Data for Validation'''
TestData = pd.DataFrame(X_test,columns=['AGE','BMI','HR','PTT'])
TestAns  = pd.DataFrame(y_test,columns=['SBP'])
combinedata   = pd.concat([TestData,TestAns],axis=1)
combinedata.to_csv('Testdata.csv')

#print(X_test)
#print(X_test.size)     #1020  = 255*4  ;4=Input Feature
#print(y_test)
#print(y_test.size)    #255   = 2543-2288
#print(X_train)
#print(X_train.size)   #9152  = 2288*4 ; 4=Input Feature
# =============================================================================
# n_estimators      : 弱機器學習個數
# max_depth         : 決策樹最大深度
# min_samples_split : 內部節點再劃分所需最小樣本數
# learning_rate     : 收縮率;建議lr<=0.1
# loss              : ls(最小平方法)、lad(最小絕對偏差)、Huber、quantile(分位數)
#                   
#'lad (最小絕對偏差): 透過目標的中值來表示
# =============================================================================

# =============================================================================
'''LAB Result '''
#
'''Change n_estimators (樹)'''
# n_estimators:1000、max_depth:8、min_samples_split:10、learning_rate:0.1:loss:ls
# ---> MSE : 3.0893 、R2 : 0.993
# n_estimators:100、max_depth:8、min_samples_split:10、learning_rate:0.1:loss:ls
# ---> MSE : 5.2726 、R2 : 0.988
# n_estimators:50、max_depth:8、min_samples_split:10、learning_rate:0.1:loss:ls
# ---> MSE : 12.5228 、R2 : 0.972
# n_estimators:25、max_depth:8、min_samples_split:10、learning_rate:0.1:loss:ls
# ---> MSE : 39.2349 、R2 : 0.913
'''Change depth (深度)'''
# n_estimators:1000、max_depth:4、min_samples_split:10、learning_rate:0.1:loss:ls
# ---> MSE : 3.4064 、R2 : 0.992
# n_estimators:100、max_depth:4、min_samples_split:10、learning_rate:0.1:loss:ls
# ---> MSE : 108.4962 、R2 : 0.761
# n_estimators:100、max_depth:16、min_samples_split:10、learning_rate:0.1:loss:ls
# ---> MSE : 3.7267 、R2 : 0.991
# n_estimators:10、max_depth:16、min_samples_split:10、learning_rate:0.1:loss:ls
# ---> MSE : 60.561 、R2 : 0.866
# n_estimators:10、max_depth:32、min_samples_split:10、learning_rate:0.1:loss:ls
# ---> MSE : 58.69 、R2 : 0.871
# =============================================================================
'''Fit regression model'''
#
'''Set Parameter '''
tree    = 100
depth   = 8
split   = 10
lrCnt   = 0.1
#
''' Define Save Result Picture File Name'''
file_name = 'Tree_'+str(tree)+'_Depth_'+str(depth)+'_Split_'+\
            str(split)+'_lr_'+str(lrCnt)
#
params = {'n_estimators': tree, 'max_depth': depth, 'min_samples_split': split,
          'learning_rate': lrCnt, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train,y_train)

y_predict = clf.predict(X_test)

'''Store Prediction Data fot Validation'''
Prediction = pd.DataFrame(y_predict,columns=['SBP'])
Prediction.to_csv('Prediction.csv')

loss    = y_predict - y_test
std     = std_s(loss)
avd     = avedev(loss)

print('LAB Paramter:'+file_name)

print("AVD : %.4f" %avd)
print("STD : %.4f" %std)

mse = mean_squared_error(y_test,y_predict)
print("MSE: %.4f" % mse)

mae = mean_absolute_error(y_test,y_predict)
print("MAE: %.4f" % mae)

score = clf.score(X_test,y_test)
print("Score : ",score)

r2  = r2_score(y_test,y_predict)
print("R2 : ",r2)

# =============================================================================
# trainscore = clf.score(X_train,y_train)
# print("train Score : ",trainscore)
# 
# #print(clf.get_params())
# print(clf.train_score_)    
# print(clf.train_score_.size)
# =============================================================================
# =============================================================================
# #For Loop Test Parameter
# 
# lossbuf = ['ls','lad','huber','quantile']
# learning_rateBuf = [0.1,0.01,0.001]
# for lossCnt in lossbuf:
#     for lrCnt in learning_rateBuf:
#         clf = ensemble.GradientBoostingRegressor(n_estimators=1500,max_depth=8,min_samples_split=10,
#                                                  learning_rate=lrCnt,loss='ls')
#         clf.fit(X_train, y_train)
#         
#         #print("Loss Function : ",lossCnt,lrCnt)
#         mse = mean_squared_error(y_test, clf.predict(X_test))
#         print("MSE: %.4f" % mse)
#         
#         r2  = r2_score(y_test,clf.predict(X_test))
#         print("R2 : ",r2)
# =============================================================================
'''Cross Validation lab'''
# =============================================================================
# cv = cross_validate(clf,X_train,y_train,cv=5,scoring=('r2','neg_mean_squared_error'),\
#                     return_train_score=True)
# #print(cv)
# #print(cv['score_time'])
# #print(cv['train_r2'])
# =============================================================================
'''cross_val_predict'''
#Reference:https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_predict.html#sphx-glr-auto-examples-model-selection-plot-cv-predict-py
# =============================================================================
# predicted = cross_val_predict(clf, X_train, y_train, cv=5)
# 
# plt.figure(figsize=(18, 6))
# plt.subplot(1, 2, 1)
# plt.title('Predection Result')
# plt.scatter(y_train, predicted, edgecolors=(0, 0, 0),label='prediction')
# plt.plot([y_train.min(), y_train.max()],[y_train.min(), y_train.max()], 'k--',label='SBP',lw=4)
# 
# plt.legend(loc='upper right')
# plt.xlabel('SBP')
# plt.ylabel('Predicted')
# plt.show()
# # =============================================================================
# # fig, ax = plt.subplots()
# # ax.scatter(y_train, predicted, edgecolors=(0, 0, 0))
# # ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
# # ax.set_xlabel('ANS')
# # ax.set_ylabel('Predicted')
# # plt.legend(loc='upper right')
# # plt.show()  
# # =============================================================================
# =============================================================================
stop = timeit.default_timer()
print('time cost: {0} s'.format(stop - start))

''' Display Now Time'''
# =============================================================================
# localtime = time.asctime(time.localtime(time.time()))
# print(localtime)
# =============================================================================
''' Plot training deviance'''
#
''' compute test set deviance'''
# =============================================================================
# test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
# 
# for i, y_pred in enumerate(clf.staged_predict(X_test)):
#     test_score[i] = clf.loss_(y_test, y_pred)
# 
# plt.figure(figsize=(10, 10))
# #plt.subplot(1, 2, 1)
# #plt.scatter(np.arange(params['n_estimators']) + 1,clf.train_score_,s=40)
# #plt.title('Deviance',fontsize=20)
# plt.title(file_name+'_Deviance',fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
#          label='Training Set Deviance',linewidth=2)
# plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
#          label='Test Set Deviance',linewidth=2)
# plt.legend(loc='upper right',fontsize=20)
# plt.xlabel('Boosting Iterations',fontsize=20)
# plt.ylabel('Deviance',fontsize=20)
# plt.savefig(file_name+"_Deviance.png")
# plt.show()
# =============================================================================
'''Bland-Altman Plot '''
# =============================================================================
# bland_altman_plot(y_test,y_predict)
# plt.title('Bland-Altman Plot')
# plt.xlabel('Averages',fontsize=10)
# plt.ylabel('Differences',fontsize=10)
# plt.savefig(file_name+"_Bland-Altman.png")
# plt.show()
# =============================================================================
''' Plot feature importance'''
# =============================================================================
# feature_importance = clf.feature_importances_
# print(feature_importance)
# # make importances relative to max importance
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
# print(feature_importance)
# sorted_idx = np.argsort(feature_importance)
# print(sorted_idx)
# pos = np.arange(sorted_idx.shape[0]) + .5
# plt.subplot(1, 2, 2)
# plt.barh(pos, feature_importance[sorted_idx], align='center')
# plt.yticks(pos, Bpdata.columns)
# print(pos , Bpdata.columns)
# plt.xlabel('Relative Importance')
# plt.title('Variable Importance')
# plt.show()
# =============================================================================
