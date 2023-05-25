# Q5

# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)
# Preprocessing Input data
data = pd.read_csv('ML\datasets\Salary_Data.csv')
X = data.iloc[:, 0]

Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()

#Stochastic gradient descent
# Building the model
m = 0
c = 0

L = 0.0001 # The learning Rate
epochs = 1000 # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing stochastic Gradient Descent 
for i in range(epochs): 
    Y_pred = m*X + c # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred)) # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred) # Derivative wrt c
    m = m - L * D_m # Update m
    c = c - L * D_c # Update c
    
print (f'm = {m} \nc = {c}')

# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y) 
plt.plot(X, Y_pred, color='red') # regression line
plt.show()

#Batch gradient descent
# Building the model
#Batch gradient descent
# Building the model
m = 0
c = 0

L = 0.0002 # The learning Rate
epochs = 1000 # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X


# Performing Batch Gradient Descent 
for i in range(epochs): 
    Y_pred = m*X + c # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred)) # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred) # Derivative wrt c
    
    m = m - L * D_m # Update m
    c = c - L * D_c # Update c
    
# Calculate R-squared
Y_mean = np.mean(Y)
ss_res = sum((Y - Y_pred) ** 2)
ss_tot = sum((Y - Y_mean) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(f"R-squared: {r2}")

# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y) 
plt.plot(X, Y_pred, color='red') # regression line
plt.show()

