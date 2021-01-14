# Decision Tree Regression
# NOTE: DTR No need for feature scaling
# NOTE: DTR is not good at low feature datasets; works a lot better on datasets with many features 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor 
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting a new result
print(regressor.predict([[6.5]]))

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)                                               
X_grid = X_grid.reshape((len(X_grid), 1))   
plt.scatter(X, y, color='red')                       
plt.plot(X_grid, regressor.predict(X_grid), color='blue') 
plt.title('Salary vs Position Level (DTR)')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()