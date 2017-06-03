import copy
import numpy as np
import pandas as pd
import math
from IPython.core.display import display
from datetime import datetime

import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt

import load_and_clean_data as lcd
import utilityfunctions as util

image_features_dir = r'C:\Users\sumabh\OneDrive\MLDS\UW-MSDS\DATA558\GitHub\kaggle\data_features'

train_features, train_labels, test_features, test_labels, labelnames = lcd.loadandextractcleandata(image_features_dir, standardize=1)

#display(labelnames)

#scikitlearn example from kernix page:
# https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_features, train_labels, test_size=0.2, random_state=42)
clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

dirname = 'sklearnlinearsvm'
path=util.makeresultsdir(dirname)

#save validation results
plot = util.plot_confusion_matrix(y_test,y_pred,labelnames,pathtosave=path)
accuracyscore=accuracy_score(y_test,y_pred)*100.0
text_file = open(path+'/results.txt', "w")
text_file.write("Accuracy: {0:0.1f}%".format(accuracyscore))
text_file.close()
print("Accuracy: {0:0.1f}%".format(accuracyscore))

#predict with the actual test data for kaggle submission
y_pred = clf.predict(test_features)

#save results
prediction_results = pd.DataFrame(np.vstack((test_labels,y_pred)).T)
prediction_results.columns = ['Id','Prediction']
display(prediction_results)
prediction_results.to_csv(path+'/Yte.csv',header=True,index=False)

print("done!!!")





