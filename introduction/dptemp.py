# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 11:48:36 2019

@author: marvin
"""

# importing the libraries
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd

#importing dataset
dataset = pd.read_csv('dataset.csv')
# creating a matrix of features(matrix of independent variables)
x = dataset.iloc[:, :-1].values
# creating a dependent variable vector
y = dataset.iloc[:, 3].values

#splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
# x_test and train dataset
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""