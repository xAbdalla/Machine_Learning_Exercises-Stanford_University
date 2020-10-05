import numpy as np
import pandas as pd
import scipy.optimize as op
import matplotlib.pyplot as plt

# read data
data = pd.read_csv('ex2data2.txt', header=None)

# show imported data details
# print('data = \n', data.head(10))
# print('**************************************')
# print('data.describe = \n', data.describe())
# print('**************************************')

# draw data
y = data.iloc[:, -1]

fig, ax = plt.subplots(figsize= (8,6))
plt.scatter(data.loc[y == 1, 0], data.loc[y == 1, 1], marker= '+', c= 'black', 
            linewidths= 3, s= 100, label= 'y = 1')

plt.scatter(data.loc[y == 0, 0], data.loc[y == 0,1], marker= 'o', c= 'yellow', 
            linewidths= 1, edgecolors= 'k', s= 80, label= 'y = 0')

ax.set(title= 'Training Dataset', xlabel= 'Microchip Test 1', ylabel= 'Microchip Test 2')
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
X = np.array(X.values)
y = np.array(y.values)

# Regularized Logistic Regression
def mapFeature(X1, X2):
    deg = 6
    out = np.ones((len(X1), 1))
    
    for i in range(1, deg+1):
        for j in range(0, i+1):
            out = np.append(out, (np.multiply( np.power(X1, (i-j)), np.power(X2, j)) ).reshape((len(X1), 1)), axis= 1)
            
    return out

X = mapFeature(X[:,1], X[:,2])

# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))

# Set regularization parameter lambda to 1
rlambda = 1

# print(X.shape)
# print(initial_theta.shape)

# Compute and display initial cost and gradient for regularized logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def costFunctionReg(theta, X, y, rlambda):
    m = len(y)
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    theta = theta.reshape((X.shape[1],1))
    
    h = sigmoid(X.dot(theta))
    unreg_cost = ((-y).T.dot(np.log(h)) - ((1 - y).T.dot(np.log(1.0 - h)))) / m
    theta[0,0] = 0
    reg_cost = theta.T.dot(theta) * rlambda / (2*m)
    J = unreg_cost + reg_cost
    
    return J

def Gradient(theta, X, y, rlambda):
    m = len(y)
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    theta = theta.reshape((X.shape[1],1))
    
    grad = np.zeros(theta.shape)
    h = sigmoid(X.dot(theta))
    theta[0,0] = 0
    
    grad = (X.T.dot(h - y) + (theta * rlambda)) / m
    
    return grad

cost = costFunctionReg(initial_theta, X, y, rlambda)
grad = Gradient(initial_theta, X, y, rlambda)


# print('\n**************************************')
# print(cost.shape)
print('\nCost at initial theta (zeros): %.3f' %cost)
print('Expected cost (approx): 0.693')
print('\n**************************************')

# print('grad.shape', grad.shape)
print('\nGradient at initial theta (zeros) - first five values only:')
for i in grad.tolist()[:5]: print(' %.4f' %i[0])
print('Expected gradients (approx):\n 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115')
print('\n**************************************')

# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones((X.shape[1], 1))
cost = costFunctionReg(test_theta.copy(), X, y, 10)
grad = Gradient(test_theta.copy(), X, y, 10)

print('\nCost at test theta (with lambda = 10): %.2f' %cost)
print('Expected cost (approx): 3.16')
print('\n**************************************')

print('\nGradient at test theta - first five values only:')
for i in grad.tolist()[:5]: print(' %.4f' %i[0])
print('Expected gradients (approx) - first five values only:\n 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922')
print('\n**************************************')

# Regularization and Accuracies
# Initialize fitting parameters
itheta = np.zeros((X.shape[1], 1))

# Set regularization parameter lambda to 1
rlambda = 0.1

Result = op.fmin_tnc(func= costFunctionReg,
                     x0= itheta,
                     args= (X, y, rlambda),
                     fprime= Gradient,
                     disp= False)

theta = Result[0].reshape((X.shape[1], 1))

# Plot Boundary
# Plot Data
fig, ax = plt.subplots(figsize= (8,6))
plt.scatter(data.loc[y == 1, 0], data.loc[y == 1, 1], marker= '+', c= 'black', 
            linewidths= 3, s= 100, label= 'y = 1')

plt.scatter(data.loc[y == 0, 0], data.loc[y == 0,1], marker= 'o', c= 'yellow', 
            linewidths= 1, edgecolors= 'k', s= 80, label= 'y = 0')


# Here is the grid range
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)

z = np.zeros((len(u), len(v)))

# Evaluate z = theta*x over the grid
for i in range(0, len(u)):
    for j in range(0, len(v)):
        mapped = mapFeature(np.array([u[i]]), np.array([v[j]]))
        z[i,j] = mapped.dot( theta )

z = z.T         # important to transpose z before calling contour
u, v = np.meshgrid(u, v)
# print(u.shape)
# print(v.shape)
# print(z.shape)

# Plot z = 0
# Notice you need to specify the range [0, 0]
contour = plt.contour(u, v, z, levels=[0], linewidths= 2)
plt.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.1)
plt.clabel(contour, inline=True, fontsize=12, fmt= 'Decision Boundary')

ax.set(title= 'Training Dataset', xlabel= 'Microchip Test 1', ylabel= 'Microchip Test 2')
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

# Compute accuracy on our training set
p = predict(theta, np.matrix(X))

print('\nTrain Accuracy: %% %.1f' %(np.mean(p == y) * 100))
print('Expected accuracy (with lambda = 1): % 83.1 (approx)')
print('\n**************************************')