import sys
print (sys.version)


# In[2]:

import os
import re

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import pickle
#import sklearn
#from sklearn import cross_validation
#from sklearn.metrics import accuracy_score, confusion_matrix
#from sklearn.svm import SVC, LinearSVC
#import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

# In[31]:

def create_graph():
    with gfile.FastGFile(os.path.join(model_dir,'classify_image_graph_def.pb'),'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ =tf.import_graph_def(graph_def,name='')



def list_all_files(directory,datatype='train'):
    if datatype != 'train':
        datatype='test'
    
    list_images_start = []    
    list_images=[]
    
    if datatype == 'train':
        print(directory)
        directory += 'train/'
        list_images_start = [directory+f for f in os.listdir(directory)]
        for i in range(len(list_images_start)):
            list_images += [list_images_start[i]+'/'+f for f in os.listdir(list_images_start[i]) if re.search('jpg|JPG', f)]
    else:    
        directory += 'test/'
        for root,directory,filenames in os.walk(directory):
            for file in filenames:
                list_images.append(os.path.join(root, file))

    #print(list_images)
    return list_images


# #local check, make this a code cell to run the test
# cwd = os.getcwd()
# 
# print(cwd)
# 
# directory = 'sampledata/'
# 
# print("train image files list: ")
# print(list_all_files(directory,'train'))
# print("")
# print("test image files list: ")
# print(list_all_files(directory,'test'))
# 

# In[ ]:

#on AWS with tensorflow
model_dir = '/home/ubuntu/src/tensorflow/tensorflow/models/image/imagenet/TUTORIAL_DIR/imagenet'
images_dir = '/home/ubuntu/src/tensorflow/tensorflow/models/image/imagenet/TUTORIAL_DIR/images/'

list_train_images = list_all_files(images_dir, 'train')

list_test_images = list_all_files(images_dir, 'test')


# In[ ]:

def extract_features(list_images):
    nb_features = 2048
    features = np.empty((len(list_images),nb_features))
    labels = []

    create_graph()

    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

    for ind, image in enumerate(list_images):
        if (ind%100 == 0):
            print('Processing %s...' % (image))
        if not gfile.Exists(image):
            tf.logging.fatal('File does not exist %s', image)

        image_data = gfile.FastGFile(image, 'rb').read()
        predictions = sess.run(next_to_last_tensor,
        {'DecodeJpeg/contents:0': image_data})
        features[ind,:] = np.squeeze(predictions)
        labels.append(re.split('_\d+',image.split('\\')[-1])[0])
        #break 

    return features, labels


# In[ ]:

train_features,train_labels = extract_features(list_train_images)

test_features,test_labels = extract_features(list_test_images)


# In[ ]:

pickle.dump(train_features, open('train_features', 'wb'))
pickle.dump(train_labels, open('train_labels', 'wb'))

pickle.dump(test_features, open('test_features', 'wb'))
pickle.dump(test_labels, open('test_labels', 'wb'))

