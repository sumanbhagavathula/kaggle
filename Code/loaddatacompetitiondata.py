import utilityfunctions as util
import load_and_clean_data as lcd
import numpy as np
import pandas as pd
from IPython.core.display import display

#image_features_dir = r'C:\Users\sumabh\OneDrive\MLDS\UW-MSDS\DATA558\GitHub\kaggle\sampledata_features'
#dirname = 'cleaneddata'
#path=util.makeresultsdir(dirname)

#train_features, train_labels, test_features, test_labels, labelnames = lcd.loadandextractcleandata(image_features_dir, standardize=1, savetopath=path)

image_features_dir = r'https://s3.amazonaws.com/data558filessuman/DataCompetitionfiles/sampledata'

#http(s)://s3.amazonaws.com/data558filessuman/DataCompetitionFiles/sampledata/labelnames.csv

train_features, train_labels, test_features, test_labels, labelnames = lcd.loadandextractcleandata(image_features_dir)
