import numpy as np
from scipy.io import loadmat
from scipy.optimize import fmin_cg

# Ignore overflow and divide by zero of np.log() and np.exp()
# np.seterr(divide = 'ignore')
# np.seterr(over = 'ignore') 

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def computeCost(theta, X, y, lamba=1):
    m = len(y)
    
    h = sigmoid( X.dot( theta ) )
    unreg_cost = (( np.log( h ).dot( -y ) ) - (( np.log( 1. - h ).dot( 1. - y ) ))) / m
    theta[0] = 0
    reg_cost = theta.T.dot( theta ) * lamba / (2*m)
    
    return unreg_cost + reg_cost

def gradientCost(theta, X, y, lamba=1):
    m = len(y)
    grad = X.T.dot( sigmoid(X.dot(theta)) - y) / m
    grad[1:] += (theta[1:] * lamba) / m
    
    return grad

def oneVsAll(X, y, num_labels, lamba):
    # Some useful variables
    m, n = X.shape
    
    # Add ones to the X data matrix
    X = np.insert(X, 0, 1, axis= 1)
    
    # need to return the following variables correctly 
    all_theta = np.zeros((n+1, num_labels))
    
    # labels are 1-indexed instead of 0-indexed
    for i in range(0, num_labels):
        theta   = np.zeros(( n+1, 1 )).reshape(-1)
        y_i     = ((y == (i+1)) + 0).reshape(-1)
        
        # minimize the objective function
        fmin = fmin_cg(computeCost,
                       x0= theta,
                       args= (X, y_i, lamba),
                       fprime= gradientCost,
                       maxiter= 300,
                       disp= False,
                       full_output= True)
        
        all_theta[:, i] = fmin[0]
        
        # np.save( "all_theta.txt", all_theta )
        print ("%2d Cost: %.5f" % (i+1, fmin[1]))
    print('===================================================')
    return all_theta

def predictOneVsAll(X, all_theta):
    # Add ones to the X data matrix
    m, n = X.shape
    X = np.insert(X, 0, 1, axis= 1)
    
    p = sigmoid(X.dot(all_theta))   # 1-D Array
    # print(p.shape)
    p_argmax = np.matrix(p.shape)   # 1-D Array
    p_argmax = np.argmax(p, axis= 1) + 1
    
    return p_argmax.reshape(m, 1) # it's important to reshape to convert it to 2-D Array.

# read data
data = loadmat('ex3data1.mat')
X, y = data['X'], data['y']

m, n = X.shape
num_labels = len(np.unique(y).tolist())
input_layer_size = n

print('\nDataset Details:\n')
print('X Shape = ' ,  X.shape, type(X)) 
print('Y Shape = ', y.shape, ' ', type(y))
print('===================================================')

lamda = 0.1
all_theta = oneVsAll(X, y, num_labels, lamda)

print('        X.shape =  ', X.shape)
print('        y.shape =  ', y.shape)
print('all_theta.shape =  ', all_theta.shape)
print('  no. of labels =   ', num_labels)
print('     data array =  ', np.unique(data['y']))
print('===================================================')

# Compute accuracy on our training set
p = predictOneVsAll(X, all_theta)
print('Training Set Accuracy: %.4f%%' %(np.mean(y == p) * 100))
print('===================================================')