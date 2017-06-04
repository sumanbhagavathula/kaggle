import numpy as np
import pandas as pd
from IPython.core.display import display
import copy
import math
import random
import multiprocessing
import load_and_clean_data as lcd
import linearsvm_sqhingeloss as lsvm
import linearsvmmulticlass_sqhingeloss as lsvmmc
from sklearn import cross_validation
import utilityfunctions as util

dirname = 'linearsvmmulticlass'
path=util.makeresultsdir(dirname)
seed = 0
image_features_dir = r'https://s3.amazonaws.com/data558filessuman/DataCompetitionfiles/data'

train_features, train_labels, test_features, test_labels = lcd.load_image_data(image_features_dir)

lambduh=0.5

x_train, x_test, y_train, y_test = cross_validation.train_test_split(train_features, train_labels, random_state=0, test_size=0.25)
#cross validation (Note: I am only doing the 2-class case for the CV here
#and also plot confusion matrix, log accuracy score
threads=multiprocessing.cpu_count()
folds=10

fitclasslabel = y_train[0]
uniqueclasses = np.unique(y_train).astype(int)
classindex = [x[0] for x in np.where(uniqueclasses == fitclasslabel)]
ysub = y_train - fitclasslabel
classidxs = (ysub == 0)
notclassidxs = np.logical_not(classidxs)
x1 = x_train[classidxs]
xrest = x_train[notclassidxs]
fraction = xrest.shape[0] / x1.shape[0]
while fraction <= 1:
    xrest = np.append(x1, xrest, axis=0)
    fraction = xrest.shape[0] / x1.shape[0]
np.random.seed(seed)
np.random.shuffle(xrest)
xrest = xrest[0:x1.shape[0], :]
x_train = np.append(x1, xrest, axis=0)
y1 = np.repeat(1, x1.shape[0])
yrest = np.repeat(-1, x1.shape[0])
y_train = np.append(y1, yrest, axis=0)

bestlambduhCV, cvscores = lsvm.linearSVM_parallelCV(x_train, y_train, folds=folds, max_iter=1000, threads=threads)
util.plotcrossvalidationresults(cvscores.iloc[1,:],cvscores.iloc[1,:], logbase = 10, pathtosave=path)


#final training using best lambda from cross validation
linearsvmmulticlassbetas = lsvmmc.linearsvmmulticlass_sqhingeloss(x=train_features, y=train_labels, lambduh=bestlambduhCV, max_iter=1000, method='fastgradientdescent', seed=0, threads=4)

classes=np.unique(train_labels)
y_pred = lsvmmc.linearsvmmulticlass_predict(linearsvmmulticlassbetas, test_features, classes)

prediction_results = pd.DataFrame(np.vstack((test_labels,y_pred)).T)
prediction_results.columns = ['Id','Prediction']
prediction_results.to_csv(path+'/Yte.csv',header=True,index=False)

print("done!!!")












