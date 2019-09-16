# Polynomial Regression

#libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =pd.read_csv('pos.csv')

# creating a matrix of features(matrix of independent variables)
x = dataset.iloc[:, :-1].values

# creating a dependent variable vector
y = dataset.iloc[:, 3].values

#splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)