import copy
import numpy as np
import pandas as pd
import math
from IPython.core.display import display
from datetime import datetime
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

import sklearn
from sklearn import svm as svm
from sklearn import cross_validation
from sklearn import metrics
import matplotlib.pyplot as plt

import load_and_clean_data as lcd
import utilityfunctions as util

def runsvccvfolds(lambduh, num_obs, x, y, folds, max_iter, seed=0):
    np.random.seed(0)
    try:
        folds
    except NameError:
        folds = 5

    try:
        max_iter
    except NameError:
        max_iter = 1000

    subscores = []
    foldbetas = np.zeros(x.shape[1])

    fold_indexes = np.arange(0, num_obs)
    fold_indexes = np.random.permutation(fold_indexes)

    accuracyscore = np.zeros(folds)
    for fold in np.arange(0, folds):
        indexstart = fold * round(num_obs / folds)
        indexend = fold * round(num_obs / folds) + round(num_obs / folds)

        # print("fold: "+ str(fold) + " indexstart: " + str(indexstart) + ", indexend: " + str(indexend))

        x_test = x[indexstart:indexend]
        x_train = np.concatenate((x[0:indexstart], x[indexend:]), axis=0)
        y_test = y[indexstart:indexend]
        y_train = np.concatenate((y[0:indexstart], y[indexend:]), axis=0)

        x_train = np.asarray(pd.DataFrame(x_train).apply(
            lambda x: (x - np.mean(x)) if (np.std(x) == 0) else (x - np.mean(x)) / np.std(x)))
        x_test = np.asarray(pd.DataFrame(x_test).apply(
            lambda x: (x - np.mean(x)) if (np.std(x) == 0) else (x - np.mean(x)) / np.std(x)))

        svmrbf = svm.SVC(C= lambduh, shrinking=True, verbose=False, decision_function_shape='ovr', random_state=seed)
        svmrbf.fit(train_features, train_labels)
        # display(linearsvmbetas[-1])

        y_pred = svmrbf.predict(x_test)

        accuracyscore[fold] = metrics.accuracy_score(y_test, y_pred) * 100.0
        #print("lambda: "+ str(lambduh) + ", fold: " + str(fold) + ", meanscore: " + str(newsubscore))

    #print("lambda: " + str(lambduh) + ", meanscore after " + str(folds) + " folds is: " + str(np.mean(subscores)))

    return np.mean(accuracyscore)

def runsvccvfoldswrapper(args):
    return runsvccvfolds(*args)


def runsvcparallelcrossvalidation(x, y, lambda_vals, folds=10, max_iter=1000, seed=0, threads=4):
    np.random.seed(seed)
    num_obs = x.shape[0]

    print("Trying with the following lambdas (and " + str(folds) + " folds each): ")
    display(lambda_vals)
    k = len(lambda_vals)
    scores = np.zeros(k)

    pool = ThreadPool(threads)
    params=[(lambduh, num_obs, x, y, folds, max_iter) for lambduh  in lambda_vals]

    scores = pool.map(runsvccvfoldswrapper, params)
    cvresults = pd.DataFrame(np.vstack((lambda_vals, scores)).T)
    cvresults.columns = ['lambdas','CV scores']

    pool.close()
    pool.join()

    return lambda_vals[np.argmin(scores)], cvresults

#run starts here
dirname = 'sklearnsvcrbfkernelcv'
path=util.makeresultsdir(dirname)

seed=0
image_features_dir = r'https://s3.amazonaws.com/data558filessuman/DataCompetitionfiles/data'

train_features, train_labels, test_features, test_labels = lcd.load_image_data(image_features_dir, standardize=1)

#add train_test_split
x_train, x_test, y_train, y_test = cross_validation.train_test_split(train_features, train_labels, random_state=0, test_size=0.25)

#add cross validation steps
lambda_start = -10
lambda_end = 2
lambdas = np.logspace(lambda_start, lambda_end, 20, base=10)

threads=multiprocessing.cpu_count()
folds=10

bestlambduhcv, cvscores = runsvcparallelcrossvalidation(x_train, y_train, lambdas, folds=10, max_iter=1000, seed=0, threads=4)

print('best lambduh from cross validation: ' + str(bestlambduhcv))

svmrbf = svm.SVC(C=bestlambduhcv, verbose=False, shrinking=True, decision_function_shape='ovr', random_state=seed)
svmrbf.fit(x_train, y_train)
# display(linearsvmbetas[-1])

y_pred = svmrbf.predict(x_test)

#save validation results
plot = util.plot_confusion_matrix(y_test,y_pred,y_train,pathtosave=path)
accuracyscore=metrics.accuracy_score(y_test,y_pred)*100.0
text_file = open(path+'/results.txt', "w")
text_file.write("Best lambda from cross-validation: ")
text_file.write(str(bestlambduhcv))
text_file.write("Accuracy: {0:0.1f}%".format(accuracyscore))
text_file.close()
print("Accuracy: {0:0.1f}%".format(accuracyscore))

#predict with the actual test data for kaggle submission
svmrbf.fit(train_features, train_labels)
# display(linearsvmbetas[-1])

y_pred = svmrbf.predict(test_features)

#save results
prediction_results = pd.DataFrame(np.vstack((test_labels,y_pred)).T)
prediction_results.columns = ['Id','Prediction']
display(prediction_results)
prediction_results.to_csv(path+'/Yte.csv',header=True,index=False)

print("done!!!")



#final training and fit
svmrbf=svm.SVC(C=bestlambduhcv,verbose=True,shrinking=True, decision_function_shape='ovr', random_state=seed)
svmrbf.fit(train_features, train_labels)

#predict with the actual test data for kaggle submission
y_pred = svmrbf.predict(test_features)

#save results
prediction_results = pd.DataFrame(np.vstack((test_labels,y_pred)).T)
prediction_results.columns = ['Id','Prediction']
display(prediction_results)
prediction_results.to_csv(path+'/Yte.csv',header=True,index=False)

print("done!!!")





