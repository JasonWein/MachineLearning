# Jason Wein
# Machine Learning CS 4267
# 4 February 2019
import matplotlib.pyplot as plot
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("3-Products-Multiple.csv")

Y = df['Profit']
X = df[['Product_1', 'Product_2', 'Product_3']]


#Y = Y.values.reshape(len(Y), 10)
#X = X.values.reshape(len(X), 10)

X_train=X[:-20]
X_test=X[-45:]

Y_train = Y[:-20]
Y_test = Y[-45:]

# Plot the data points
#plot.scatter(X_test, Y_test, color='black')
#plot.title('Test Data')
#plot.xlabel('Products')
#plot.ylabel('Profit')
#plot.xticks(())
#plot.yticks(())

# Create a linear regression object
regr = linear_model.LinearRegression(normalize=True)

# Train the linear regression model
regr.fit(X_train, Y_train)

# Plot the outputs of the graph
plot.plot(X_test, regr.predict(X_test), color='red', linewidth=3)
plot.show()

###############################

#info = pd.read_csv('3-Products-Multiple.csv')
#df = pd.DataFrame(info,columns=['info','Product_1','Product_2','Product_3', 'Location', 'Profit'])

#X = info[['Product_1','Product_2', 'Product_3']]
#Y = info['Profit']

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#regr = linear_model.LinearRegression()

#regr.fit(X_train, y_train)
#coeff_df = pd.DataFrame(regr.coef_, X.columns, columns=['Coefficient'])

#print(coeff_df)

#plot.scatter(X_test, y_test, color='black')
#plot.title('Test Data')
#plot.xlabel('Products')
#plot.ylabel('Profit')
#plot.xticks(())
#plot.yticks(())

#plot.plot(X_test, regr.predict(X_test), color='red', linewidth=3)
#plot.show()