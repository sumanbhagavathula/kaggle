import copy
import numpy as np
import pandas as pd
import math
from IPython.core.display import display
from datetime import datetime

import sklearn
from sklearn import linear_model as lm
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

import load_and_clean_data as lcd
import utilityfunctions as util

seed=0
image_features_dir = r'https://s3.amazonaws.com/data558filessuman/DataCompetitionfiles/data'

train_features, train_labels, test_features, test_labels = lcd.load_image_data(image_features_dir, standardize=1)

#display(labelnames)

#scikitlearn example from kernix page:
# https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11

lambda_start = -10
lambda_end = 2
lambdas = np.logspace(lambda_start, lambda_end, 50, base=10)

C=2/(train_features.shape[0] * lambdas)

print('trying with the following hyperparameters for tuning...')
display(C)

logregcv = lm.LogisticRegressionCV(penalty='l1',multi_class='ovr',solver='liblinear', Cs=C, cv=10, max_iter=1000,n_jobs=-1, refit=True, random_state=seed)
logregcv.fit(train_features, train_labels)


bestlambdacvarray = 2/(train_features.shape[0] * logregcv.C_)
display(bestlambdacvarray)
display(pd.DataFrame(logregcv.coef_))

bestlambdacvidx = np.argmin(bestlambdacvarray)
print('best lambda index from cv = ' + str(bestlambdacvidx))

bestlambdacv = bestlambdacvarray[bestlambdacvidx]
print('best lambda from cv = ' + str(bestlambdacv))

print('coeff shape')
display(logregcv.coef_.shape)

nonzerobetaindxs = np.argwhere(logregcv.coef_[bestlambdacvidx,:])
print('non-zero beta indexes')
display(nonzerobetaindxs.shape)
display(nonzerobetaindxs.T[0])

nonzerobetaindxs = nonzerobetaindxs.T[0]


train_features = np.asarray(pd.DataFrame(train_features).iloc[:,nonzerobetaindxs])
test_features = np.asarray(pd.DataFrame(test_features).iloc[:,nonzerobetaindxs])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_features, train_labels, test_size=0.2, random_state=42)


logregl2=lm.LogisticRegression(penalty='l2',multi_class='ovr', C=logregcv.C_[0],max_iter=1000,n_jobs=-1, random_state=seed)

logregl2.fit(X_train, y_train)

y_pred = logregl2.predict(X_test)

dirname = 'sklearnlogisticregressioncv'
path=util.makeresultsdir(dirname)

#save validation results
plot = util.plot_confusion_matrix(y_test,y_pred,y_train,pathtosave=path)
accuracyscore=accuracy_score(y_test,y_pred)*100.0
text_file = open(path+'/results.txt', "w")
text_file.write("Accuracy: {0:0.1f}%".format(accuracyscore))
text_file.close()
print("Accuracy: {0:0.1f}%".format(accuracyscore))

#predict with the actual test data for kaggle submission
y_pred = logregl2.predict(test_features)

#save results
prediction_results = pd.DataFrame(np.vstack((test_labels,y_pred)).T)
prediction_results.columns = ['Id','Prediction']
display(prediction_results)
prediction_results.to_csv(path+'/Yte.csv',header=True,index=False)

print("done!!!")





