# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:11:59 2020

@author: 138709
"""

# Random Forest Algorithm

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values


# Splitting the data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)



# Fitting the randomforest model here
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
model.fit(X_train, Y_train)


# Predicting the test set results
Y_pred = model.predict(X_test)


# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
