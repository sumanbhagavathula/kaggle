import pickle
import numpy
import pandas as pd

test_features = pickle.load(open('./data_features/test_features','rb'))
print(test_features)

tfdf=pd.DataFrame(test_features)

tfdf.to_csv("./data_features/test_features.csv")


test_labels = pickle.load(open('./data_features/test_labels','rb'))
print(test_labels)

tldf=pd.DataFrame(test_labels)

tldf.to_csv("./data_features/test_labels.csv")


train_features = pickle.load(open('./data_features/train_features','rb'))
print(train_features)

trfdf=pd.DataFrame(train_features)

trfdf.to_csv("./data_features/train_features.csv")


train_labels = pickle.load(open('./data_features/train_labels','rb'))
print(train_labels)

trldf=pd.DataFrame(train_labels)

trldf.to_csv("./data_features/train_labels.csv")
