# Jason Wein
# Machine Learning CS 4267
# 4 February 2019

import matplotlib.pyplot as plot
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# Read from the provided excel doc
df = pd.read_csv('Propose-Salaries-Polynomial.csv')

Y = df['Salary']
X = df['Level']

X = X.values.reshape(len(X), 1)
Y = Y.values.reshape(len(Y), 1)

# Create a training set and test set for the x variables.
X_train=X[:-5]
X_test=X[-10:]
# Create a training set and test set for the y variables.
Y_train = Y[:-5]
Y_test = Y[-10:]

# Plot outputs
plot.scatter(X_test, Y_test, color='black')
plot.title('Test Data')
plot.xlabel('Level')
plot.ylabel('Salary')
plot.xticks(())
plot.yticks(())

# Create a linear regression object and fit the X and Y values to it.
regr = linear_model.LinearRegression()
regr.fit(X,Y)

# Create a polynomial object and set its degree to 5.
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)


poly.fit(X_poly,Y)
regr2 = linear_model.LinearRegression()
regr2.fit(X_poly, Y)

# Train the polynomial regression model
poly.fit(X_train, Y_train)

# Plot the data points
plot.plot(X, regr2.predict(poly.fit_transform(X)), color='red', linewidth=3)
plot.show()

print(poly.fit_transform([[6.5]]))
