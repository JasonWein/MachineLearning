#Jason Wein
# Machine Learning
# 11 Feb 2019

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

#load the data from the given CSV file.
data = pd.read_csv("advertising_appealsUpdated.csv", None)

#X-features (Age, Salary)
X = data.iloc[:, :2]
#Y- feature (purchased)
y = data.iloc[:, -1]
#Filter those who purchased and those who did not.
purchased = data.loc[y==1]

no_purchase = data.loc[y==0]
#Create a scatter plot with red dots indicating who did not purchase and blue for those who did.
plot.scatter(purchased.iloc[:, 0], purchased.iloc[:, 1], s=10, label='Purchased')
plot.scatter(no_purchase.iloc[:, 0], no_purchase.iloc[:, 1], s=10, label='Did not purchase')
plot.legend()
plot.show()