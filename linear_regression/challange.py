import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
#read data
dataframe = pd.read_csv('challenge_dataset.txt')
print dataframe.head()
x_values = dataframe[[0]]
y_values = dataframe[[1]]
#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
# The coefficients
print('Coefficients: ', body_reg.coef_)
# The mean squared error
print('Mean squared error: %.2f ' % np.mean((body_reg.predict(x_values) - y_values) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % body_reg.score(x_values, y_values))