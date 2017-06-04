import copy
import numpy as np
import pandas as pd
import math
from IPython.core.display import display
from datetime import datetime

import sklearn
from sklearn import svm as svm
import matplotlib.pyplot as plt

import load_and_clean_data as lcd
import utilityfunctions as util
from sklearn.decomposition import PCA

seed=0
targetvariance=0.75

image_features_dir = r'https://s3.amazonaws.com/data558filessuman/DataCompetitionfiles/data'

train_features, train_labels, test_features, test_labels = lcd.load_image_data(image_features_dir, standardize=0)

x_train = np.asarray(pd.DataFrame(train_features).apply(lambda x: (x - np.mean(x)) if (np.std(x) == 0) else (x - np.mean(x)) / np.std(x)))

x_test = np.asarray(pd.DataFrame(test_features).apply(lambda x: (x - np.mean(x)) if (np.std(x) == 0) else (x - np.mean(x)) / np.std(x)))

ncomponents=x_train.shape[1]

#from scikitlearn
pca = PCA(ncomponents, svd_solver='randomized', random_state=seed)
pca.fit(x_train)

cumsum_exp_var = np.cumsum(pca.explained_variance_ratio_)

targetcomponents = np.where(cumsum_exp_var > targetvariance)[0][0]+1

print('No of pca components that explain ' + str(targetvariance) + ': ' + str(targetcomponents))

pca = PCA(targetcomponents, svd_solver='randomized', random_state=seed)
pca.fit(x_train)

train_features = pca.transform(x_train)
test_features = pca.transform(x_test)

dirname = 'pca_linearsvmmulticlass'
path=util.makeresultsdir(dirname)

pd.DataFrame(train_features).to_csv(path+'/train_features_pca.csv',header=False,index=False)
pd.DataFrame(test_features).to_csv(path+'/test_features_pca.csv',header=False,index=False)

lambduh=0.5


svmrbf=svm.SVC(verbose=True,decision_function_shape='ovr', random_state=seed)
svmrbf.fit(train_features, train_labels)

#predict with the actual test data for kaggle submission
y_pred = svmrbf.predict(test_features)


#save results
prediction_results = pd.DataFrame(np.vstack((test_labels,y_pred)).T)
prediction_results.columns = ['Id','Prediction']
display(prediction_results)
prediction_results.to_csv(path+'/Yte.csv',header=True,index=False)

print("done!!!")





