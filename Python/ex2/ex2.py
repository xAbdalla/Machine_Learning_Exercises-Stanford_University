import numpy as np
import pandas as pd
import scipy.optimize as op
import matplotlib.pyplot as plt

# read data
data = pd.read_csv('ex2data1.txt', header=None)

# show imported data details
# print('data = \n', data.head(10))
# print('**************************************')
# print('data.describe = \n', data.describe())
# print('**************************************')

# draw data
y = data.iloc[:, -1]

fig, ax = plt.subplots(figsize= (8,5))
plt.scatter(data.loc[y == 1, 0], data.loc[y == 1, 1], marker= '+', c= 'black', 
            linewidths= 3, s= 100, label= 'Admitted')

plt.scatter(data.loc[y == 0, 0], data.loc[y == 0,1], marker= 'o', c= 'yellow', 
            linewidths= 1, edgecolors= 'k', s= 80, label= 'Not admitted')

ax.set(title= 'Training Dataset', xlabel= 'Exam 1 score', ylabel= 'Exam 2 score')
ax.legend(loc= 1, shadow= True, borderpad= 1)
plt.show()

# Setup the data matrix appropriately, and add ones for the intercept term
data.insert(0, -1 , 1)
# print('\nnew data = \n', data.head(10))
# print('\n**************************************')

# separate X (training data) from y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0 : cols-1]
y = data.iloc[:, cols-1 : cols]

# print('\nX data = \n', X.head(10))
# print('\n**************************************')
# print('\ny data = \n', y.head(10))
# print('\n**************************************')

# Convert data from data frames to numpy matrices
X = np.matrix(X.values)
y = np.matrix(y.values)

# Initialize fitting parameters
m, n = X.shape
initial_theta = np.zeros((n, 1))

# print('X \n', X)
# print('\nX.shape = ', X.shape)
# print('\n**************************************')
# print('y \n', y)
# print('\ny.shape = ', y.shape)
# print('\n**************************************')
# print('initial_theta \n', initial_theta)
# print('\ninitial_theta.shape = ', initial_theta.shape)
# print('\n**************************************')

# Compute and display initial cost and gradient
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

nums = np.arange(-10, 10, 0.5)

fig, ax = plt.subplots(figsize= (8,5))
ax.plot(nums, sigmoid(nums), 'r')
ax.grid()
plt.show()

def costFunction(theta, X, y):
    m = len(y)
    theta = theta.reshape((X.shape[1],1))
    
    h = sigmoid(X * theta)
    J = ((y.T * np.log(h)) + ((1 - y).T * np.log(1 - h))) / (-m)
    return J

def Gradient(theta, X, y):
    theta = theta.reshape((X.shape[1],1))
    return ((X.T * (sigmoid(X * theta) - y)) / len(y))

cost = costFunction(initial_theta, X, y)
grad = Gradient(initial_theta, X, y)

print('\nCost at initial theta (zeros): %.3f' %cost)
print('Expected cost (approx): 0.693')
print('\n**************************************')

print('\nGradient at initial theta (zeros): ')
for i in grad.tolist(): print(' %.4f' %i[0])
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628')
print('\n**************************************')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24], [0.2], [0.2]])
cost = costFunction(test_theta, X, y)
grad = Gradient(test_theta, X, y)

print('\nCost at test theta: %.3f' %cost)
print('Expected cost (approx): 0.218')
print('\n**************************************')

print('\nGradient at test theta: ')
for i in grad.tolist(): print(' %.3f' %i[0])
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647')
print('\n**************************************')

# Run fminunc to obtain the optimal theta
# This function will return theta and the cost
Result = op.fmin_tnc(func= costFunction,
                     x0= initial_theta,
                     args= (X, y),
                     fprime= Gradient,
                     disp= False)

theta = Result[0].reshape((X.shape[1], 1))

print('\nNew Theta After Optimization = ')
for i in range(theta.shape[0]): print(' %.3f' %theta[i, 0])
print('Expected theta (approx):')
print(' -25.161\n 0.206\n 0.201')
print('\n**************************************')

newCost = costFunction(Result[0], X, y)
print('\nNew Cost After Optimization = %.3f' %newCost)
print('Expected cost (approx): 0.203')
print('\n**************************************')

# Plot Boundary
# Plot Data
fig, ax = plt.subplots(figsize= (8,5))
plt.scatter(data.loc[y == 1, 0], data.loc[y == 1, 1], marker= '+', c= 'black', 
            linewidths= 3, s= 100, label= 'Admitted')

plt.scatter(data.loc[y == 0, 0], data.loc[y == 0,1], marker= 'o', c= 'yellow', 
            linewidths= 1, edgecolors= 'k', s= 80, label= 'Not admitted')


# Only need 2 points to define a line, so choose two endpoints
plot_x = np.array([np.min(X[:,1])-2, np.max(X[:,2])+2])

# Calculate the decision boundary line
theta = np.array(Result[0]).reshape((X.shape[1], 1))
plot_y = np.multiply(((-1.) / theta[2]) , (np.multiply(theta[1], plot_x) + theta[0]))

# print('x =\n', plot_x)
# print('y =\n', plot_y)

# Plot, and adjust axes for better viewing
plt.plot(plot_x, plot_y, linewidth= 2, label= 'Decision Boundary')

ax.set(title= 'Training Dataset with Decision Boundary', xlabel= 'Exam 1 score', ylabel= 'Exam 2 score',
        xlim= (28, 101), ylim= (21, 101))
ax.legend(loc= 1, shadow= True)
plt.show()

# Predict and Accuracies
def predict(theta, X):
    # PREDICT Predict whether the label is 0 or 1 using learned logistic
    # regression parameters theta
    p = np.zeros((X.shape[0],1))
    
    for i in range(X.shape[0]):
        x = X[i, :].T
        if sigmoid(theta.T * x) >= 0.5 :
            p[i] = 1
        else:
            p[i] = 0
    
    return p

prob = sigmoid(np.matrix([1, 45, 85]) * theta)

print('\nFor a student with scores 45 and 85, we predict an admission probability of %.3f' %prob[0])
print('Expected value: 0.775 +/- 0.002')
print('\n**************************************')

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %% %.1f' %(np.mean(p == y) * 100))
print('Expected accuracy (approx): % 89.0')
print('\n**************************************')