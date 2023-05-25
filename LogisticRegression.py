# Import the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load the classification data
df = pd.read_csv('ML\datasets\Logistic-Regression-Data.csv')
df.head()

# vizualizing the given data
sns.scatterplot(x='BloodPressure', y='Diabetes', data=df)  

# Separate the features and label
x = df[['Glucose','BloodPressure']]
y = df['Diabetes']
x

# Define the sigmoid function
def sigmoid(input):    
    output = 1 / (1 + np.exp(-input))
    return output

def optimize(x, y,learning_rate,iterations,parameters): 
   size = x.shape[0]
   weight = parameters["weight"] 
   bias = parameters["bias"]
   for i in range(iterations): 
    sigma = sigmoid(np.dot(x, weight) + bias)
    try:
        if sigma != 0:
            loss = -1/size * np.sum(y * np.log(sigma)) + (1 - y) * np.log(1-sigma)
    except:
        pass
    dW = 1/size * np.dot(x.T, (sigma - y))
    db = 1/size * np.sum(sigma - y)
    weight -= learning_rate * dW
    bias -= learning_rate * db 
    
   parameters["weight"] = weight
   parameters["bias"] = bias
   return parameters

# Initialize the weight and bais
init_parameters = {} 
init_parameters["weight"] = np.zeros(x.shape[1])
init_parameters["bias"] = 0

# Define the train function
def train(x, y, learning_rate,iterations):
    parameters_out = optimize(x, y, learning_rate, iterations ,init_parameters)
    return parameters_out

# Train the model
parameters_out = train(x, y, learning_rate = 0.02, iterations = 346)

# Predict using the trained model
# output_values = np.dot(x[:10], parameters_out["weight"]) + parameters_out["bias"]
# predictions = sigmoid(output_values) >= 1/2
# # predictions
# print("Predictions : ",predictions)


# Predict using the trained model
output_values = np.dot(x[:10], parameters_out["weight"]) + parameters_out["bias"]
print("ot : ",output_values)
sns.lineplot()
opData=pd.DataFrame()
opData['Diabetes']= [ sigmoid(x)  for x in output_values ]
opData['Bloodpressure']= df['BloodPressure'] 
sns.lineplot(x="Bloodpressure", y="Diabetes", data=opData)
plt.title("Diabetes Vs Bloodpressure graph - Logistic Regression")
plt.show()
predictions = sigmoid(output_values) >= 1/2
# predictions
print("Predictions : ",predictions)
