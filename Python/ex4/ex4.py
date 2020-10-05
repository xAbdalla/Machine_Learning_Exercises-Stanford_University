import numpy as np
from scipy.io import loadmat
from scipy.optimize import fmin_cg

def sigmoid( z ):
    return ( 1.0 / ( 1.0 + np.exp( -z ) ) )

def sigmoidGradient( z ):
    return np.multiply( sigmoid( z ) , ( 1.0 - sigmoid( z ) ) )

def recodeLabel( y, num_labels ):
    y = np.matrix(y)
    m = y.shape[0]
    ry = np.zeros((m, num_labels))
    
    for i in range(m):
        ry[i, y[i]-1] = 1
    
    return ry

def feedForward( Theta1, Theta2, X, X_bias = False ):
    # Input Layer
    a1 = np.c_[ np.ones((X.shape[0], 1)), X ] if X_bias is False else X
    
    # Hidden Layer
    z2 = a1.dot( Theta1 )
    a2 = sigmoid( z2 )
    a2 = np.c_[ np.ones((X.shape[0], 1)), a2 ]
    
    # Output Layer
    z3 = a2.dot( Theta2 )
    a3 = sigmoid( z3 )
    
    return (a1, a2, a3, z2, z3)

def paramUnroll( nn_params, input_layer_size, hidden_layer_size, num_labels ):
    theta1_elems = ( input_layer_size + 1 ) * hidden_layer_size
    theta1_size  = ( input_layer_size + 1, hidden_layer_size  )
    theta2_size  = ( hidden_layer_size + 1, num_labels )
    
    # Reshape nn_params back into the parameters Theta1 and Theta2
    # the weight matrices for our 2 layer neural network
    theta1 = nn_params[:theta1_elems].reshape( theta1_size )
    theta2 = nn_params[theta1_elems:].reshape( theta2_size )
   
    return (theta1, theta2)

def randInitializeWeights(L_in, L_out, INIT_EPSILON= 0.01):
    return (np.random.randn(L_in + 1, L_out) * 2 * INIT_EPSILON) - INIT_EPSILON

def debugInitializeWeights(L_in, L_out):
    num_elements = L_out * ( 1 + L_in )
    W = np.array([ np.sin( x ) / 10 for x in range( 1, num_elements + 1 ) ])
    return W.reshape( L_out, L_in + 1 )

def computeNumericalGradient( J, theta, input_layer_size, hidden_layer_size, num_labels, X, y, lamba ):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    num_elements = theta.shape[0]
    
    for p in range(num_elements) :
        perturb[p] = e
        loss1 = nnCostFunction( theta - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lamba )[0]
        loss2 = nnCostFunction( theta + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lamba )[0]
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    return numgrad

def checkNNGradients( lamba= 0 ):
    d_input_layer_size    = 3
    d_hidden_layer_size   = 5
    d_num_labels          = 3
    d_m                   = 5
    
    # We generate some 'random' test data
    theta1 = debugInitializeWeights( d_hidden_layer_size, d_input_layer_size )
    theta2 = debugInitializeWeights( d_num_labels, d_hidden_layer_size )
    
    # Reusing debugInitializeWeights to generate X
    X = debugInitializeWeights( d_input_layer_size - 1, d_m )
    y = 1 + np.mod( d_m, d_num_labels )
    
    # Unroll parameters
    nn_params = np.r_[ theta1.T.flatten(), theta2.T.flatten() ]
    
    J, grad = nnCostFunction(nn_params, d_input_layer_size, d_hidden_layer_size, d_num_labels, X, y, lamba)
    numgrad = computeNumericalGradient( J, nn_params, d_input_layer_size, d_hidden_layer_size, d_num_labels, X, y, lamba )
    
    for i in range(nn_params.shape[0]):
        print('%15f' %numgrad[i], '%15f' %grad[i])
    
    print('The above two columns you get should be very similar.\n'+
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)')
    
    

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamba):
    # Reshape nn_params back into the parameters Theta1 and Theta2
    # the weight matrices for our 2 layer neural network
    Theta1, Theta2 = paramUnroll( nn_params, input_layer_size, hidden_layer_size, num_labels )
    # print(Theta1.shape)
    # print(Theta2.shape)
    
    # need to return the following variables correctly
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    # Part 1: CostFunction
    # -------------------------------------------------------------
    a1, a2, a3, z2, z3     = feedForward( Theta1, Theta2, X )
    ry = recodeLabel( y, num_labels )
    # print(ry.shape)
    # print(a3.shape)
    # assert np.shape(ry) == np.shape(a3), "Error, shape of recoded y is different from a3"
    
    unreg_cost = ( ry * np.log( a3 ) ) + ( ( 1 - ry ) * np.log( 1 - a3 ) )
    unreg_cost = - np.sum( unreg_cost ) / m
    
    reg_cost   = np.sum(Theta1[1:, :] ** 2) + np.sum(Theta2[1:, :] ** 2)
    reg_cost   = (reg_cost * lamba) / (2 * m)
    
    J          = unreg_cost + reg_cost
    # -------------------------------------------------------------
    
    # Part 2: Backpropagation algorithm
    # -------------------------------------------------------------
    sigma3 = a3 - ry
    sigma2 = np.multiply( sigma3.dot( Theta2.T )[:, 1:], sigmoidGradient( z2 ) )
    
    accum1 = sigma2.T.dot( a1 )
    accum2 = sigma3.T.dot( a2 )
    
    Theta1_grad = ( accum1 + lamba * np.insert(Theta1.T[1:,:], 0, 0, axis= 0) ) / m
    Theta2_grad = ( accum2 + lamba * np.insert(Theta2.T[1:,:], 0, 0, axis= 0) ) / m
    
    grad = np.r_[Theta1_grad.T.flatten(), Theta2_grad.T.flatten()]
    # -------------------------------------------------------------
    
    return J, grad

def computeCost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamba):
    return nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamba)[0]

def computeGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamba):
    return nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamba)[1]

def predict(Theta1, Theta2, X, m):
    pred = np.zeros(( m, 1 ))
    
    h1 = sigmoid( np.c_[ np.ones(( m, 1 )), X].dot( Theta1 ))
    h2 = sigmoid( np.c_[ np.ones(( m, 1 )), h1].dot( Theta2 ))
    
    pred = np.argmax(h2, axis= 1) + 1
    
    return pred.reshape(m, 1) # it's important to reshape to convert it to 2-D Array.

def part3():
    print('\n' + ' Part 3: Compute Cost (Feedforward) '.center(80, '=') + '\n')
    print('Feedforward Using Neural Network ...')
    
    # Weight regularization parameter (we set this to 0 here).
    lamba = 0
    J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamba)
    
    print('Cost at parameters (loaded from ex4weights.mat): %f' %J)
    print('(this value should be about 0.287629)')

def part4():
    print('\n' + ' Part 4: Implement Regularization '.center(80, '=') + '\n')
    print('Checking Cost Function (w/ Regularization) ...')
    
    # Weight regularization parameter (we set this to 1 here).
    lamba = 1
    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamba)[0]
    
    print('Cost at parameters (loaded from ex4weights.mat): %f' %J)
    print('(this value should be about 0.383770)')

def part5():
    print('\n' + ' Part 5: Sigmoid Gradient '.center(80, '=') + '\n')
    print('Evaluating sigmoid gradient...')
    
    g = sigmoidGradient( np.array( [-1, -0.5, 0, 0.5, 1] ) )
    
    print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:')
    print(g)

def part6():
    print('\n' + ' Part 6: Initializing Pameters '.center(80, '=') + '\n')
    print('Initializing Neural Network Parameters ...')
    
    initial_Theta1 = randInitializeWeights( input_layer_size, hidden_layer_size )
    initial_Theta2 = randInitializeWeights( hidden_layer_size, num_labels )
    
    # Unroll parameters
    initial_nn_params = np.r_[ initial_Theta1.T.flatten(), initial_Theta2.T.flatten() ]
    print('initial_nn_params.shape =', initial_nn_params.shape)

def part7():
    print('\n' + ' Part 7: Implement Backpropagation '.center(80, '=') + '\n')
    print('Checking Backpropagation...')
    
    # Check gradients by running checkNNGradients
    checkNNGradients()

def part8():
    print('\n' + ' Part 8: Implement Regularization '.center(80, '=') + '\n')
    print('Checking Backpropagation (w/ Regularization) ... ')
    
    # Check gradients by running checkNNGradients
    lamba = 3
    checkNNGradients(3)
    print('===================================================')
    
    # Also output the costFunction debugging values
    debug_J = nnCostFunction( nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamba )[0]
    
    print('Cost at (fixed) debugging parameters (w/ lambda = {}): {:.6f}'.format(lamba, debug_J))
    print('(for lambda = 3, this value should be about 0.576051)')

def part9():
    print(' Part 9: Training NN '.center(50, '=').center(80) + '\n')
    print('Training Neural Network...')
    
    initial_Theta1 = randInitializeWeights( input_layer_size, hidden_layer_size )
    initial_Theta2 = randInitializeWeights( hidden_layer_size, num_labels )
    
    # Unroll parameters
    initial_nn_params = np.r_[ initial_Theta1.T.flatten(), initial_Theta2.T.flatten() ]
    
    lamba = 1
    result = fmin_cg( computeCost,
                     fprime= computeGradient,
                     x0= initial_nn_params,
                     args= (input_layer_size, hidden_layer_size, num_labels, X, y, lamba),
                     maxiter= 100,
                     disp= False,
                     full_output= True )
    
    nn_params = result[0]
    return nn_params

def part10():
    print('\n' + ' Part 10: Implement Predict '.center(80, '='))
    
    # Initializing All Inputs
    data = loadmat('ex4data1.mat')
    X, y = data['X'], data['y']                 # X (5000, 400), y (5000, 1)
    
    m, n = X.shape
    input_layer_size = n                        # = 400 (20x20 Input Images of Digits)
    hidden_layer_size = 25                      # 25 hidden units
    num_labels = len(np.unique(y).tolist())     # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
    
    nn_params = part9()
    Theta1, Theta2 = paramUnroll( nn_params, input_layer_size, hidden_layer_size, num_labels )
    
    pred = predict(Theta1, Theta2, X, m)
    
    print('Training Set Accuracy: %.2f' %(np.mean(pred == y) * 100))

def main():
    part3()
    part4()
    part5()
    part6()
    part7()
    part8()
    # part9()               # Called in part10()
    part10()

if (__name__ == '__main__'):
    # Loading the dataset.
    # working with a dataset that contains handwritten digits.
    
    # Load Training Data
    print('\n' + ' Part 1: Loading Data '.center(80, '=') + '\n')
    print('Loading Data ...')
    data = loadmat('ex4data1.mat')
    print('Data Loaded!')
    X, y = data['X'], data['y']                                 # X (5000, 400), y (5000, 1)
    m, n = X.shape
    
    # Loading Parameters
    # Load the weights into variables Theta1 and Theta2
    
    # Load the weights into variables Theta1 and Theta2
    print('\n' + ' Part 2: Loading Parameters '.center(80, '=') + '\n')
    print('Loading Saved Neural Network Parameters ...')
    weights = loadmat('ex4weights.mat')
    print('Neural Network Parameters Loaded!')
    Theta1, Theta2 = weights['Theta1'], weights['Theta2']       # Theta1 (25, 401), Theta2 (10, 26)
    
    # Unroll parameters
    nn_params = np.r_[Theta1.T.flatten(), Theta2.T.flatten()]   # (10285,) Vector
    # print(nn_params.shape)
    
    input_layer_size = n                        # = 400 (20x20 Input Images of Digits)
    hidden_layer_size = 25                      # 25 hidden units
    num_labels = len(np.unique(y).tolist())     # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
    
    main()
