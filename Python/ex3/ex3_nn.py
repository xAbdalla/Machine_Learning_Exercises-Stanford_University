import numpy as np
from scipy.io import loadmat
# from scipy.optimize import fmin_cg

# Ignore overflow and divide by zero of np.log() and np.exp()
# np.seterr(divide = 'ignore')
# np.seterr(over = 'ignore') 

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def predict(Theta1, Theta2, X):
    # Useful values
    m, n = X.shape
    X = np.insert(X, 0, 1, axis= 1)
    
    a2 = np.c_[np.ones((m, 1)), sigmoid( X.dot( Theta1.T ) )]
    h_theta = sigmoid( a2.dot( Theta2.T ) )
    
    return  (np.argmax( h_theta, axis= 1) + 1).reshape(m, 1)

# read data
data = loadmat('ex3data1.mat')
X, y = data['X'], data['y']

weights = loadmat('ex3weights.mat')
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

m, n = X.shape
num_labels = len(np.unique(y).tolist())
input_layer_size = n

print('\nDataset Details:\n')
print('X Shape = ' ,  X.shape, type(X)) 
print('Y Shape = ', y.shape, ' ', type(y))
print('Theta1 Shape = ' ,  Theta1.shape, type(Theta1)) 
print('Theta2 Shape = ', Theta2.shape, ' ', type(Theta2))
print('===================================================')

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: %f' %(np.mean(y == pred) * 100))
print('===================================================')

rp =list(range(m))
np.random.shuffle(rp)
np.random.shuffle(rp)
np.random.shuffle(rp)

for i in rp:
    pred = predict(Theta1, Theta2, np.matrix(X[i, :]))
    print('Neural Network Prediction: %d (digit %d): y = %d' %(pred, np.mod(pred, 10), y[i,:]))
    print('===================================================')
    
    # Pause with quit option
    s = input('Paused - press enter to continue, q to exit: ');
    if s == 'q' :
      break