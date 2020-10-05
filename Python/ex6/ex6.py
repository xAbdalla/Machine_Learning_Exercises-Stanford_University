# -*- coding: utf-8 -*-
"""
Creator: Abdalla H.
Created on: 26 Sep. 2020
"""

# Import Libraries
import numpy as np
from sklearn import svm
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Ignore overflow and divide by zero of np.log() and np.exp()
# np.seterr(divide = 'ignore')
# np.seterr(over = 'ignore') 

# Exercise 6 | Support Vector Machines

def plotData(X, y):
    # Find Indices of Positive and Negative Examples
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    
    # Plot Examples
    fig, ax = plt.subplots(figsize=(8,8))
    plt.scatter(X[pos, 0], X[pos, 1], marker= '+', s= 80, c= 'black',
                linewidths= 3, label= 'y = 1')
    plt.scatter(X[neg, 0], X[neg, 1], marker= 'o', s= 60, c= 'red',
                linewidths= 1, edgecolors= 'darkred', label= 'y = 0')
    ax.legend(loc= 3, shadow= True, borderpad= 1)

def visualizeBoundaryLinear(X, y, svc, h=0.01, pad=0.25, color= 'yellow'):
    plotData(X, y)
    
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    # plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c= color, marker='|', s=60, linewidths=1)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('$\mathrm{X_1}$')
    plt.ylabel('$\mathrm{X_2}$')
    plt.title('This is the Plot of the Kernal when C = %d' %svc.get_params()['C'])
    plt.show()
    print('Number of support vectors: ', svc.support_.size)

def gaussianKernel(x1, x2, sigma= 1.0):
    # Ensure that x1 and x2 are column vectors
    x1 = x1.ravel()
    x2 = x2.ravel()
    
    return np.exp( -((x1 - x2).T.dot(x1 - x2) / (2 * sigma * sigma)) )

def dataset3Params(X, y, Xval, yval, train_set= [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]):
    error_t = 999
    
    for i in range(len(train_set)):
        for j in range(len(train_set)):
            # train the SVM first
            rbf_svm = svm.SVC(C= train_set[i], kernel= 'rbf', gamma= 1.0/train_set[j])         # gamma is the inverse of sigma
            rbf_svm.fit(X, y.ravel())
            
            # test it out on validation data
            predictions = rbf_svm.predict( Xval ).reshape(yval.shape)
            prediction_error = np.mean(predictions.astype(float) != yval)
            
            if prediction_error < error_t:
                C = train_set[i]
                sigma = train_set[j] 
    
    return C, sigma, train_set

def part1():
    print(' Part 1: Loading and Visualizing Data '.center(80, '='))
    print('Loading and Visualizing Data ...')
    
    # Load from ex6data1
    data1 = loadmat('ex6data1.mat')
    X1 = data1['X']
    y1 = data1['y']
    
    # Plot training data
    plotData(X1, y1)
    plt.show()
    
    return X1, y1

def part2():
    print(' Part 2: Training Linear SVM '.center(80, '='))
    # Load from ex6data1
    data1 = loadmat('ex6data1.mat')
    X1 = data1['X']
    y1 = data1['y']
    
    print('Training Linear SVM ...')
    
    # You should try to change the C value below and see how the decision
    # boundary varies (e.g., try C = 1000)
    C = 1
    model = svm.SVC(C= C, kernel= 'linear')
    model.fit(X1, y1.ravel())
    visualizeBoundaryLinear(X1, y1, model)
    
    C = 1000
    model.set_params(C= C)
    model.fit(X1, y1.ravel())
    visualizeBoundaryLinear(X1, y1, model)

def part3():
    print(' Part 3: Implementing Gaussian Kernel '.center(80, '='))
    print('Evaluating the Gaussian Kernel ...')
    
    x1 = np.array([1., 2., 1.])
    x2 = np.array([0., 4., -1.])
    sigma = 2.0
    
    sim = gaussianKernel(x1, x2, sigma)
    print('Gaussian Kernel between x1 = [1 2 1], x2 = [0 4 -1], sigma = %.1f : %f' %(sigma, sim))
    print('(for sigma = 2, this value should be about 0.324652)')

def part4():
    print(' Part 4: Visualizing Dataset 2 '.center(80, '='))
    print('Loading and Visualizing Data ...')
    
    # Load from ex6data2
    data1 = loadmat('ex6data2.mat')
    X2 = data1['X']
    y2 = data1['y']
    
    # Plot training data
    plotData(X2, y2)
    plt.show()
    
    return X2, y2

def part5():
    print(' Part 5: Training SVM with RBF Kernel (Dataset 2) '.center(80, '='))
    print('Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...')
    # Load from ex6data2
    data1 = loadmat('ex6data2.mat')
    X2 = data1['X']
    y2 = data1['y']
    
    # SVM Parameters
    C = 1.0
    sigma = 0.1
    
    rbf_svm = svm.SVC(C= C, kernel= 'rbf', gamma= 1.0/sigma)         # gamma is the inverse of sigma
    rbf_svm.fit(X2, y2.ravel())
    visualizeBoundaryLinear(X2, y2, rbf_svm, pad= 0.03)

def part6():
    print(' Part 6: Visualizing Dataset 3 '.center(80, '='))
    print('Loading and Visualizing Data ...')
    
    # Load from ex6data3
    data1 = loadmat('ex6data3.mat')
    X3 = data1['X']
    y3 = data1['y']
    
    # Plot training data
    plotData(X3, y3)
    plt.show()
    
    return X3, y3

def part7():
    print(' Part 7: Training SVM with RBF Kernel (Dataset 3) '.center(80, '='))
    # Load from ex6data3
    data1 = loadmat('ex6data3.mat')
    X3, y3, Xval, yval = data1['X'], data1['y'], data1['Xval'], data1['yval']
    
    # Try different SVM Parameters here
    C, sigma, train_set = dataset3Params(X3, y3, Xval, yval)
    
    # Train the SVM
    rbf_svm = svm.SVC(C= C, kernel= 'rbf', gamma= 1.0/sigma)         # gamma is the inverse of sigma
    rbf_svm.fit(X3, y3.ravel())
    visualizeBoundaryLinear(X3, y3, rbf_svm, pad= 0.03)
    print('Optimal C =', C, ' , Optimal sigma =', sigma)
    # print('Form train_set =', train_set)
    
    # Testing
    print('Test accuracy = {0}%'.format(np.round(rbf_svm.score(Xval, yval) * 100, 2)))

def main():
    X1, y1 = part1()
    part2()
    part3()
    X2, y2 = part4()
    part5()
    X3, y3 = part6()
    part7()

if __name__ == '__main__' :
    main()