import copy
import numpy as np
import pandas as pd
import math
from IPython.core.display import display
from datetime import datetime
import multiprocessing

import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

import load_and_clean_data as lcd
import utilityfunctions as util
import logisticregression_l2regularization as logregl2

image_features_dir = r'https://s3.amazonaws.com/data558filessuman/DataCompetitionfiles/data'

train_features, train_labels, test_features, test_labels, labelnames = lcd.loadandextractcleandata(image_features_dir, standardize=1)

threads=multiprocessing.cpu_count()
folds=10

dirname = 'twoclass_logisticregression_L2reg'
path=util.makeresultsdir(dirname)

x_train, x_test, y_train, y_test = cross_validation.train_test_split(train_features, train_labels, random_state=0, test_size=0.25)

#logregl2CV_lambda, cvscores = logregl2.logisticregression_CV(x_train, y_train, folds=folds, max_iter=1000)
logregl2CV_lambda, cvscores = logregl2.logisticregression_parallelCV(x_train, y_train, folds=folds, max_iter=1000, threads=threads)
print("Best lambda found from cross validation: " + str(logregl2CV_lambda))
#save validation results
util.plotcrossvalidationresults(cvscores.iloc[1, :], cvscores.iloc[1, :], logbase=10, pathtosave=path)

logregl2_betas = logregl2.logisticregression(x_train, y_train, logregl2CV_lambda, max_iter=1000, method='fastgradientdescent')
print('fitted coefficients: ')
display(logregl2_betas)
validationscore = logregl2.logisticregressionscore(x_test, y_test, logregl2_betas)
print('Validation score on test-set = ' + str(validationscore))

text_file = open(path + '/results.txt', "w")
text_file.write('Validation score on test-set = ' + str(validationscore))
text_file.close()

# save results
pd.DataFrame(logregl2_betas).to_csv(path + '/betas.csv', header=False, index=False)
print("done!!!")



