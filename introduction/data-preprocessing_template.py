# data preprocessing

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

# Taking care of missing data

# import necessary library
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
  

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Country category
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

# creating dummy variables for country catergory
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

# purchased catergory
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
# x_test and train dataset
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)



