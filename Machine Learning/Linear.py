# Jason Wein
# Machine Learning CS 4267
# 4 February 2019

import matplotlib.pyplot as plot
from sklearn import linear_model
import pandas as pd

# Read from the provided excel doc
df = pd.read_csv("Salaries-Simple_Linear.csv")

Y = df['Years_of_Expertise']
X = df['Salary']
print(X)

X = X.values.reshape(len(X), 1)
Y = Y.values.reshape(len(Y), 1)

# Create a training set and test set for the x variables.
X_train = X[:-14]
X_test = X[-30:]

# Create a training set and test set for the y variables.
Y_train = Y[:-14]
Y_test = Y[-30:]

# Plot the data points
plot.scatter(X_test, Y_test, color='black')
plot.title('Test Data')
plot.xlabel('Salary')
plot.ylabel('Years_of_Expertise')
plot.xticks(())
plot.yticks(())

# Create a linear regression object
regr = linear_model.LinearRegression()

# Train the linear regression model
regr.fit(X_train, Y_train)

# Plot the outputs of the graph
plot.plot(X_test, regr.predict(X_test), color='red', linewidth=3)
plot.show()