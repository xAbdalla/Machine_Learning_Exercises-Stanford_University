#%%
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# read data
path = 'ex1data2.txt'
data = pd.read_csv(path, header = None, names= ['Size', 'Bedrooms', 'Price'])

#%%
# show imported data details
# print('data = \n', data.head(10))
# print('**************************************')
# print('data.describe = \n', data.describe())
# print('**************************************')

#%%
#Normalizes the features in X
def featureNormalize(data):
    data_norm = data.copy()
    data_norm.iloc[:, 0:-1] = (data.iloc[:, 0:-1] - data.iloc[:, 0:-1].mean()) / data.iloc[:, 0:-1].std()
    return data_norm, data.mean(), data.std()

data_norm, mu, sigma = featureNormalize(data)

#%%
# show normalized data details
# print('norm data = \n', data_norm.head(10))
# print('**************************************')
# print('data_norm.describe = \n', data_norm.describe())
# print('**************************************')

#%%
# adding a new column called ones before the data
data.insert(0, 'Ones', 1)
data_norm.insert(0, 'Ones', 1)
# print('new data = \n', data_norm.head(10))
# print('**************************************')

#%%
# separate X (training data) from y (target variable)
cols = data_norm.shape[1]
X = data_norm.iloc[:, 0 : cols-1]
y = data_norm.iloc[:, cols-1 : cols]

# print('X data = \n', X.head(10))
# print('**************************************')
# print('y data = \n', y.head(10))
# print('**************************************')

#%%
# draw data
#3D
fig, ax = plt.subplots(figsize= (6,6), subplot_kw={'projection' : '3d'})
ax.scatter3D(data_norm['Size'], data_norm['Bedrooms'], data_norm['Price'], s= 30)
plt.show()

#2D
fig, ax = plt.subplots(figsize= (6,6))
ax.scatter(data_norm['Size'], data_norm['Price'], s= 30, label= 'Dataset')
ax.set(xlabel= 'Size (Normalized)', ylabel= 'Price', title= 'Dataset')
plt.legend(loc= 1, shadow=True)
plt.show()

#%%
# Convert data from data frames to numpy matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array(np.zeros((cols-1, 1))))

# print('X \n', X)
print('\nX.shape = ', X.shape)
print('**************************************')
# print('y \n', y)
print('y.shape = ', y.shape)
print('**************************************')
# print('theta \n', theta)
print('theta.shape = ', theta.shape)
print('**************************************')

#%%
# Cost function
def computeCost(X, y, theta):
    J = np.sum(np.power(((X * theta) - y), 2)) / (2 * len(y))
    return float(J)

#%%
# Running Gradient Descent
def gradientDescent(X, y, theta, alpha, num_iters):
    temp = np.matrix(np.zeros(theta.shape))
    J_history = np.zeros((num_iters, 1))
    
    for i in range(num_iters):
        for j in range(len(theta)):
            temp[j, 0] = theta[j, 0] - ((alpha / len(y)) * np.sum(np.multiply(((X * theta) - y), X[:, j])))
        
        theta = temp.copy()
        J_history[i,0] = computeCost(X, y, theta)
    
    return theta, J_history

iterations = 400
alpha = 0.01

theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

print('Theta computed from gradient descent: ')
for i in range(len(theta)): print(theta[i,0])
print('**************************************')

test = np.array([[1.], [1650.], [3.]])
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $', (theta.T * test)[0,0])
print('Expected Price (approx)\n $ 165489064.118993')
print('**************************************')
# print('J_history\n', J_history)
# print('**************************************')

#%%
# Normalize the test point
print(test)
test[1,:] = (test[1,:] - np.mean(np.matrix(data['Size']), axis= 1)) / np.std(np.matrix(data['Size']), axis= 1)
test[2,:] = (test[2,:] - np.mean(np.matrix(data['Bedrooms']), axis= 1)) / np.std(np.matrix(data['Bedrooms']), axis= 1)
print(test)

# draw the best fit line from Gradient Descent
fig, ax = plt.subplots(figsize= (6,6))
ax.scatter(data_norm['Size'], data_norm['Price'], s= 30, label= 'Dataset')
ax.scatter(test[1,:], (theta.T * test)[0,0], s= 30, marker= 'x', color= 'r', linewidths=2, label= 'Test Data')
ax.plot(X[:,1], (X[:,1]*theta[1,:]) + theta[0,:], 'r-', linewidth= 2, label= 'Best Fit line')
ax.set(xlabel= 'Size (Normalized)', ylabel= 'Price', title= 'Best Fit line from Gradient Descent')
plt.legend(loc= 2, shadow=True, borderpad= 1)
plt.show()

#%%
# draw error graph
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(np.arange(1, iterations+1), J_history, 'b', linewidth=2)
ax.set(xlabel= 'No. of Iterations', ylabel= 'Cost', title= 'Error vs. Training Data')
plt.show()

#%%
# reset the data
del X,y,theta
cols = data.shape[1]
X = data.iloc[:, 0 : cols-1]
y = data.iloc[:, cols-1 : cols]
X = np.matrix(X.values)
y = np.matrix(y.values)

#%%
# Calculate the parameters from the normal equation
def normalEqn(X, y):
    theta = np.matrix(np.zeros((X.shape[1], 1)))
    theta = np.linalg.inv(X.T * X) * X.T * y
    return theta

theta = normalEqn(X, y)

#%%
# Display normal equation's result
print('\nTheta computed from the normal equations: ')
for i in range(len(theta)): print(theta[i,0])
print('**************************************')

test = np.array([[1.], [1650.], [3.]])
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $', (theta.T * test)[0,0])
print('Expected Price (approx)\n $ 293081.464335')
print('**************************************')

#%%
# draw the best fit line from normal equation
fig, ax = plt.subplots(figsize= (6,6))
ax.scatter(data['Size'], data['Price'], s= 30, label= 'Dataset')
ax.scatter(test[1,:], (theta.T * test)[0,0], s= 30, marker= 'x', color= 'r', linewidths=2, label= 'Test Data')
ax.plot(X[:,1], (X[:,1]*theta[1,:]) + theta[0,:], 'r-', linewidth= 2, label= 'Best Fit line')
ax.set(xlabel= 'Size', ylabel= 'Price', title= 'Best Fit line from Normal Equation')
plt.legend(loc= 2, shadow=True, borderpad= 1)
plt.show()
