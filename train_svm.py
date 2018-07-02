from __future__ import print_function
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import random
import pickle
from functions import store_data

filename = "data_processing/data.p"
keypoints = pickle.load(open(filename, "rb"))

#Create dataset for svm
X = list()
Y = list()
for clas in keypoints:
  for instance in keypoints[clas]:
    Y.append(clas)
    X.append(np.nan_to_num(np.concatenate((instance[0], instance[1]))))

#Shuffle dataset
combined = list(zip(X, Y))
random.shuffle(combined)
X[:], Y[:] = zip(*combined)

#Train
clf = svm.SVC(kernel='linear')
dataset_size = len(Y)
train_size = int(dataset_size*0.8)
clf.fit(X[0:train_size], Y[0:train_size])

#Predict
eval_size = dataset_size - train_size
y_pred = clf.predict(X[train_size:])
y_true = Y[train_size:]

print("Accuracy:", accuracy_score(y_true, y_pred))

store = raw_input("Store model?(y/n): ")
if store == "y":
    store_data(clf, "svm.p")