# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:30:07 2019

@author: marvin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('salary_data.csv')
# creating a matrix of features(matrix of independent variables)
x = data.iloc[:, :-1].values
# creating a dependent variable vector
y = data.iloc[:, 1].values

#splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.333, random_state = 0)

# Fitting Simple  linear regression into the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) 

# Predicting the test set results
y_pred = regressor.predict(x_test)

# Visualising the training set results
plt.scatter(x_train, y_train, color='green')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience[training set]')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show() 

# Visualising the test set results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience[test set]')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show() 