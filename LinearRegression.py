# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8.0, 6.0)

# Preprocessing Input data
# data = pd.read_csv('datasets/Salary_Data.csv')
data = pd.read_csv('ML\datasets\Salary_Data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.title("Datapoints - Linear regression")
plt.scatter(X, Y)
plt.show()

# print("X: ",X,Y)
# Stochastic Gradient Descent
# Building the model
m = 0
c = 0

L = 0.0001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing stochastic Gradient Descent 
for i in range(epochs): 
    # print("X: ",X)
    Y_pred = m*X + c  # The current predicted value of Y
    # print("y pred : ",Y_pred)
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print("Stochastic Gradient : ")
print (f'm = {m} \nc = {c}')

# Batch gradient descent
# Building the model
m = 0
c = 0

L = 0.0002  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Batch Gradient Descent 
for i in range(epochs): 
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c 
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c


print("Batch Gradient  : ")
print (f'm = {m} \nc = {c}')

# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y) 
plt.title("Line of Regression ")
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()
