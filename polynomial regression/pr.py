# Polynomial Regression

#libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('pos.csv')


# creating a matrix of features(matrix of independent variables)
x = dataset.iloc[:, 1:2].values

# creating a dependent variable vector
y = dataset.iloc[:, 2].values

#fit Linear regression model to dataset
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(x,y)
# fit polynomial regression modelto dataset
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4)
xpoly= poly.fit_transform(x)
lin2 = LinearRegression()
lin2.fit(xpoly, y)

# visualize the linear regression results
plt.scatter(x, y, color = 'red')
plt.plot(x,lin.predict(x), color = 'green')
plt.title('Linear reg results')
plt.xlabel('position level')
plt.ylabel('salary')

# visualize the polynomial regression results 
x_grid= np.arange(min(x),max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid,lin2.predict(poly.fit_transform(x_grid)), color = 'green')
plt.title('polynomial reg results')
plt.xlabel('position level')
plt.ylabel('salary')

# predict salary with a 6.5 level (linear reg)
lin.predict(6.5)
# predict salary with a 6.5 level (linear reg)
lin2.predict(poly.fit_transform(6.5))