# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# importing the libraries
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd

#importing dataset
dataset = pd.read_csv('50_Startups.csv')
# creating a matrix of features(matrix of independent variables)
x = dataset.iloc[:, :-1].values
# creating a dependent variable vector
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Country category
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])

# creating dummy variables for country catergory
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

# Avioding the dummy variable trap
x = x[:, 1:] 

#splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Fitting multi linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predicting test results
y_pred = regressor.predict(x_test)

#Building optimal model using backward elimination
import statsmodels.formula.api as sn
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sn.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 3, 4, 5]]
regressor_OLS = sn.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 3, 4, 5]]
regressor_OLS = sn.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 3, 5]]
regressor_OLS = sn.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 3]]
regressor_OLS = sn.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()