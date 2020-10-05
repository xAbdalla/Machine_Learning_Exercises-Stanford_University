#%%
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# read data
path = 'ex1data1.txt'
data = pd.read_csv(path, header = None, names= ['Population', 'Profit'])

#%%
# show imported data details
# print('data = \n', data.head(10))
# print('**************************************')
# print('data.describe = \n', data.describe())
# print('**************************************')

#%%
# draw data
fig, ax = plt.subplots(figsize= (6,6))
plt.scatter(data['Population'], data['Profit'], marker= 'o', s= 12, label= 'Training data')
ax.set(title= 'Best Fit Line', xlabel= 'Population', ylabel= 'Profit')

#%%
# adding a new column called ones before the data
data.insert(0, 'Ones', 1)
# print('new data = \n', data.head(10))
# print('**************************************')

#%%
# separate X (training data) from y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0 : cols-1]
y = data.iloc[:, cols-1 : cols]

# print('X data = \n', X.head(10))
# print('**************************************')
# print('y data = \n', y.head(10))
# print('**************************************')

#%%
# Convert data from data frames to numpy matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array(np.zeros((cols-1, 1))))

# print('X \n', X)
print('X.shape = ', X.shape)
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

print('With theta = [0 : 0]\nCost computed =', computeCost(X, y, theta))
print('Expected cost value (approx) 32.07')
print('**************************************')
print('With theta = [-1 : 2]\nCost computed =', computeCost(X, y, np.array([[-1],[2]])))
print('Expected cost value (approx) 54.24')
print('**************************************')

#%%
# Running Gradient Descent
def gradientDescent(X, y, theta, alpha, num_iters):
    temp = np.matrix(np.zeros(theta.shape))
    J_history = np.zeros((num_iters, 1))
    
    for i in range(num_iters):
        for j in range(len(theta)):
            temp[j, 0] = theta[j, 0] - ((alpha / len(y)) * np.sum(np.multiply(((X * theta) - y), X[:, j])))
        
        theta = temp.copy()
        J_history[i] = computeCost(X, y, theta)
    
    return theta, J_history

iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

print('Theta found by gradient descent:\n', theta[0,0], '\n ', theta[1,0])
print('Expected theta values (approx)\n -3.6303\n  1.1664')
print('**************************************')
# print('J_history\n', J_history)
# print('**************************************')

#%%
# Plot the linear fit
plt.plot(X[:, 1], X*theta, 'r-', linewidth= 1.5, label= 'Linear regression')
plt.legend()

#%%
# Predict values for population sizes of 35,000 and 70,000
predict1 = [1., 3.5] * theta
print('For population = 35,000, we predict a profit of %f' %(predict1 * 10000))
predict2 = [1., 7.] * theta
print('For population = 70,000, we predict a profit of %f' %(predict2 * 10000))
print('**************************************')

#%%
# draw error graph
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(np.arange(1, iterations+1), J_history, 'r', linewidth=2)
ax.set(xlabel= 'No. of Iterations', ylabel= 'Cost', title= 'Error vs. Training Epoch')

#%%
# Visualizing J(theta_0, theta_1)

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i,j] = computeCost(X, y, t)
    
J_vals = J_vals.T
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
fig, ax = plt.subplots(figsize= (7,7), subplot_kw={'projection' : '3d'})
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap= 'jet')
# ax.view_init(0,90)
ax.set(xlabel= '$\mathrm{\\theta_{0}}$', ylabel= '$\mathrm{\\theta_{1}}$', zlabel= 'Cost Function  $\mathrm{J(\\theta)}$')
plt.plot(theta[0,0], theta[1,0], 'rx', markersize=10, linewidth=2)