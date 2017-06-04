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

seed=0
image_features_dir = r'https://s3.amazonaws.com/data558filessuman/DataCompetitionfiles/data'

train_features, train_labels, test_features, test_labels = lcd.load_image_data(image_features_dir, standardize=1)


svmrbf=svm.SVC(verbose=True,decision_function_shape='ovr', random_state=seed)
svmrbf.fit(train_features, train_labels)

#predict with the actual test data for kaggle submission
y_pred = svmrbf.predict(test_features)


dirname = 'sklearnpcasvcrbfkernel'
path=util.makeresultsdir(dirname)


#save results
prediction_results = pd.DataFrame(np.vstack((test_labels,y_pred)).T)
prediction_results.columns = ['Id','Prediction']
display(prediction_results)
prediction_results.to_csv(path+'/Yte.csv',header=True,index=False)

print("done!!!")





