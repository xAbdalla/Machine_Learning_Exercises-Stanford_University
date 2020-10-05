# -*- coding: utf-8 -*-
"""
Creator: Abdalla H.
Created on: Mon Sep 28 05:27:06 2020
"""

# Import Libraries
import numpy as np
import scipy.optimize as op
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Ignore overflow and divide by zero of np.log() and np.exp()
# np.seterr(divide = 'ignore')
# np.seterr(over = 'ignore') 

def plotData(X, y, s= 50, linewidth= 1, x_label= 'x', y_label= 'y', label= 'Data'):# Plot Examples
    fig, ax = plt.subplots(figsize=(6,6))
    plt.scatter(X, y, s= s, linewidth= linewidth, c= 'red', marker= 'x', label= label)
    ax.set(xlabel= x_label, ylabel= y_label)

def plotLine(X, y, line= '--', linewidth= 1, label= 'Boundary'):
    plt.plot(X, y, line, linewidth= linewidth, label= label)

def linearRegCostFunction(theta, X, y, lamba, m):
    theta = theta.reshape((X.shape[1], 1))
    return (sum(np.power( X.dot( theta ) - y, 2 )) + sum(lamba * np.power(theta[1:, :], 2))) / (2*m)

def linearRegGradientFunction(theta, X, y, lamba, m):
    theta = theta.reshape((X.shape[1], 1))
    return (((X.T.dot( X.dot( theta ) - y )) + ( np.r_[np.zeros((1, 1)), theta[1:, :]] * lamba )) / m).ravel()

def trainLinearReg(X, y, lamba, m):
    # Initialize Theta
    initial_theta = np.zeros((X.shape[1], 1))
    
    result = op.fmin_cg(f= linearRegCostFunction,
                        x0= initial_theta,
                        fprime= linearRegGradientFunction,
                        args= (X, y, lamba, m),
                        maxiter= 200,
                        disp= 1)
    
    return result

def learningCurve(X, y, Xval, yval, lamba, m):
    error_train, error_val = np.zeros((m, 1)), np.zeros((m, 1))
    
    for i in range(m):
        Xs = X[:i, :]
        ys = y[:i, :]
        theta = trainLinearReg(Xs, ys, lamba, m)
        error_train[i, 0] = linearRegCostFunction(theta, Xs, ys, 0, m)
        error_val[i, 0] = linearRegCostFunction(theta, Xval, yval, 0, m)
    
    return error_train, error_val

def polyFeatures(X, p):
    X_poly = X


    # if p is equal or greater than 2
    if p >= 2:

        # for each number between column 2 (index 1) and last column
        for k in range(1,p):

            # add k-th column of polynomial features where k-th column is X.^k
            X_poly = np.column_stack((X_poly, np.power(X,k+1)))
            

    return X_poly

def featureNormalize(X):
    mu = np.mean( X , axis= 0)
    X_norm = X - mu
    sigma = np.std(X_norm, axis= 0, ddof= 1)
    X_norm = X_norm / sigma
    
    return X_norm, mu, sigma

def plotFit(min_x, max_x, mu, sigma, theta, p):
    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.array(np.arange(min_x - 15, max_x + 25, 0.05)) # 1D vector

    # Map the X values 
    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma

    # Add ones
    X_poly = np.column_stack((np.ones((x.shape[0], 1)), X_poly))

    # Plot
    plt.plot(x, np.dot(X_poly, theta), '--', linewidth=2, label= 'Fit Line')

def validationCurve(X, y, Xval, yval, m):
    # Selected values of lambda (you should not change this)
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).T
    
    error_train, error_val = np.zeros(lambda_vec.shape), np.zeros(lambda_vec.shape)
    
    for i in range(len(lambda_vec)):
        lamba = lambda_vec[i]
        theta = trainLinearReg(X, y, lamba, m)
        error_train[i] = linearRegCostFunction(theta, X, y, 0, m)
        error_val[i] = linearRegCostFunction(theta, Xval, yval, 0, m)
    
    return lambda_vec, error_train, error_val

def part1():
    print('\n' + ' Part 1: Loading and Visualizing Data '.center(80, '='), end= '\n\n')
    
    # Load Training Data
    print('Loading and Visualizing Data ...')
    # Load from ex5data1
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    data = loadmat('ex5data1.mat')
    X, y, Xval, yval, Xtest, ytest = data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']
    
    # m = Number of examples
    m = X.shape[0]
    
    # Plot training data
    plotData(X, y, 60, 1.5, x_label= 'Change in water level (x)', y_label= 'Water flowing out of the dam (y)', label= 'Xy Data')
    plt.legend(loc= 2, shadow= True, borderpad= 1)
    plt.show()
    
    return X, y, Xval, yval, Xtest, ytest, m
    
def part2(X, y, m):
    print('\n' + ' Part 2: Regularized Linear Regression Cost '.center(80, '='), end= '\n\n')
    
    theta = np.ones((X.shape[1] + 1, 1))
    lamba = 1.0
    J = linearRegCostFunction(theta, np.c_[np.ones((m, 1)), X], y, lamba, m)
    
    print('Cost at theta = [1 ; 1]: %f' %J)
    print('(this value should be about 303.993192)')
    
    return J

def part3(X, y, m):
    print('\n' + ' Part 3: Regularized Linear Regression Gradient '.center(80, '='), end= '\n\n')
    
    theta = np.ones((X.shape[1] + 1, 1))
    lamba = 1.0
    grad = linearRegGradientFunction(theta, np.c_[np.ones((m, 1)), X], y, lamba, m).reshape(theta.shape)
    
    
    print('Gradient at theta = [1 ; 1]:  [%f; %f]' %(grad[0,0], grad[1,0]))
    print('(this value should be about [-15.303016; 598.250744])')
    
    return grad

def part4(X, y, m):
    print('\n' + ' Part 4: Train Linear Regression '.center(80, '='), end= '\n\n')
    
    # Train linear regression with lambda = 0
    lamba = 0.0
    theta = trainLinearReg(np.c_[np.ones((m, 1)), X], y, lamba, m)
    
    # Plot fit over the data
    plotData(X, y, 60, 1.5, x_label= 'Change in water level (x)', y_label= 'Water flowing out of the dam (y)', label= 'Xy Data')
    plotLine(X, np.c_[np.ones((m, 1)), X].dot( theta ), line= '--', linewidth= 2, label= 'Best Fit Line')
    plt.legend(loc= 2, shadow= True, borderpad= 1)
    plt.show()
    
    return theta

def part5(X, y, Xval, yval, m):
    print('\n' + ' Part 5: Learning Curve for Linear Regression '.center(80, '='), end= '\n\n')
    
    lamba = 0.0
    error_train, error_val = learningCurve(np.c_[np.ones((m, 1)), X],
                                           y,
                                           np.c_[np.ones((Xval.shape[0], 1)), Xval],
                                           yval,
                                           lamba,
                                           m)
    
    # Plotting the Error
    fig, ax = plt.subplots(figsize=(6,6))
    plotLine(list(range(m)), error_train, line= '-', linewidth= 2, label= 'Train')
    plotLine(list(range(m)), error_val, line= '-', linewidth= 2, label= 'Cross Validation')
    ax.set(title= 'Learning curve for linear regression',
           xlabel= 'Number of training examples',
           ylabel= 'Error',
           xlim= (0, 13),
           ylim= (0, 400))
    plt.legend(loc= 1, shadow= True, borderpad= 1)
    plt.show()
    
    print('# Training Examples\t\tTrain Error\t\tCross Validation Error')
    for i in range(m):
        print('%15d%19f%22f' %(i+1, error_train[i, 0], error_val[i, 0]))

def part6(X, Xval, Xtest, m):
    print('\n' + ' Part 6: Feature Mapping for Polynomial Regression '.center(80, '='), end= '\n\n')
    
    p = 8
    
    # Map X onto Polynomial Features and Normalize
    X_poly = polyFeatures(X, p)
    X_poly, mu, sigma = featureNormalize(X_poly) # Normalize
    X_poly = np.c_[np.ones((m, 1)), X_poly] # Add Ones
    
    # Map X_poly_test and normalize (using mu and sigma)
    X_poly_test = polyFeatures( Xtest, p )
    X_poly_test = X_poly_test - mu
    X_poly_test = X_poly_test / sigma
    X_poly_test = np.c_[np.ones(( np.shape(X_poly_test)[0], 1)), X_poly_test]
    
    # Map X_poly_val and normalize (using mu and sigma)
    X_poly_val = polyFeatures( Xval, p )
    X_poly_val = X_poly_val - mu
    X_poly_val = X_poly_val / sigma
    X_poly_val = np.c_[np.ones(( np.shape(X_poly_val)[0], 1)), X_poly_val]
    
    print('Normalized Training Example 1:')
    for i in range(X_poly.shape[1]):
        print('%f' %X_poly[0, i])
    
    return X_poly, X_poly_test, X_poly_val, mu, sigma, p

def part7(X, y, X_poly, X_poly_val, yval, mu, sigma, p, m):
    print('\n' + ' Part 7: Learning Curve for Polynomial Regression '.center(80, '='), end= '\n\n')
    
    lamba = 0.0
    theta = trainLinearReg(X_poly, y, lamba, m)
    
    # Plot training data and fit
    plotData(X, y, 60, 1.5, x_label= 'Change in water level (x)', y_label= 'Water flowing out of the dam (y)', label= 'Xy Data')
    plotFit( min(X), max(X), mu, sigma, theta, p )
    plt.legend(loc= 2, shadow= True, borderpad= 1)
    plt.title('Polynomial Regression Fit (lambda = {:.4f})'.format(lamba))
    plt.show()
    
    error_train, error_val = learningCurve(X_poly,
                                           y,
                                           X_poly_val,
                                           yval,
                                           lamba,
                                           m)
    
    # Plotting the Error
    fig, ax = plt.subplots(figsize=(6,6))
    plotLine(list(range(m)), error_train, line= '-', linewidth= 2, label= 'Train')
    plotLine(list(range(m)), error_val, line= '-', linewidth= 2, label= 'Cross Validation')
    ax.set(title= 'Polynomial Regression Fit (lambda = {:.4f})'.format(lamba),
           xlabel= 'Number of training examples',
           ylabel= 'Error',
           xlim= (0, 13),
           ylim= (0, 300))
    plt.legend(loc= 1, shadow= True, borderpad= 1)
    plt.show()
    
    print('Polynomial Regression (lambda = {:.4f})'.format(lamba))
    print('# Training Examples\t\tTrain Error\t\tCross Validation Error')
    for i in range(m):
        print('%15d%19f%22f' %(i+1, error_train[i, 0], error_val[i, 0]))

def part8(X_poly, y, X_poly_val, yval, m):
    print('\n' + ' Part 8: Validation for Selecting Lambda '.center(80, '='), end= '\n\n')
    lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval, m)
    
    fig, ax = plt.subplots(figsize=(6,6))
    plotLine(lambda_vec, error_train, line= '-', linewidth= 2, label= 'Train')
    plotLine(lambda_vec, error_val, line= '-', linewidth= 2, label= 'Cross Validation')
    ax.set(xlabel= 'lambda', ylabel= 'Error')
    plt.legend(loc= 1, shadow= True, borderpad= 1)
    plt.show()
    
    print('# lambda\t\tTrain Error\t\tCross Validation Error')
    for i in range(len(lambda_vec)):
        print('%8.3f%18f%22f' %(lambda_vec[i], error_train[i], error_val[i]))

def main():
    print(' Exercise 5 | Regularized Linear Regression and Bias-Variance '.center(80, '='))
    
    X, y, Xval, yval, Xtest, ytest, m = part1()
    part2(X, y, m)
    part3(X, y, m)
    part4(X, y, m)
    part5(X, y, Xval, yval, m)
    X_poly, X_poly_test, X_poly_val, mu, sigma, p = part6(X, Xval, Xtest, m)
    part7(X, y, X_poly, X_poly_val, yval, mu, sigma, p, m)
    part8(X_poly, y, X_poly_val, yval, m)

if __name__ == '__main__' :
    main()