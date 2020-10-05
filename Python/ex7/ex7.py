# -*- coding: utf-8 -*-
"""
Creator: Abdalla H.
Created on: 30 Sep 2020
"""

# Import Libraries
import sys
import scipy.io
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

def hsv(n=63):
    # return colors.hsv_to_rgb( np.column_stack([ np.array(range(n+1)).T / float(n), np.ones( ((n+1), 2) ) ]) )
    return colors.hsv_to_rgb( np.column_stack([ np.linspace(0, 1, n+1)            , np.ones( ((n+1), 2) ) ]) )

def findClosestCentroids(X, centroids):
    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly.
    idx = np.zeros((X.shape[0], 1))
    # set m = # of training examples
    m = X.shape[0]
    
    min_distance = None
    # for every training example
    for i in range(m):
        # for every centroid
        for j in range(K):
            # compute the euclidean distance between the example and the centroid
            difference = X[i,:] - centroids[j,:]
            distance = np.power(np.sqrt( difference.dot(difference.T) ), 2)

            # if this is the first centroid, initialize the min_distance and min_centroid
            # OR 
            # if distance < min_distance, reassign min_distance=distance and min_centroid to current j
            if j == 0 or distance < min_distance:
              min_distance = distance
              min_centroid = j


        # assign centroid for this example to one corresponding to the min_distance 
        idx[i]= min_centroid

    return idx

def computeCentroids(X, idx, K):
    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly.
    centroids = np.zeros((K, n))
    
    for j in range(K):
        
        #   two-array output
    	centroid_examples = np.nonzero(idx == j)[0]

    	# compute mean over all such training examples and reassign centroid
    	centroids[j,:] = np.mean( X[centroid_examples,:], axis=0 )

    return centroids

def drawLine(p1, p2, **kargs):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], **kargs)

def plotDataPoints(X, idx, K):
    # Create palette (see hsv.py)
    palette = hsv( K )
    colors = np.array([palette[int(i)] for i in idx])

    # Plot the data
    plt.scatter(X[:,0], X[:,1], s=75, facecolors='none', edgecolors= colors, linewidth= 1)

    return

def plotProgresskMeans(X, centroids, previous, idx, K, i):
    # Plot the examples
    plotDataPoints(X, idx, K)
    
    # Plot the centroids as black x's
    plt.scatter(previous[:,0], previous[:,1], marker='x', s=100, c='gray', linewidth=3)
    plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=100, c='k', linewidth=3)

    # Plot the history of the centroids with lines
    for j in range(centroids.shape[0]):
        drawLine(centroids[j, :], previous[j, :], c='black', linewidth= 4)
        drawLine(centroids[j, :], previous[j, :], c='magenta', linewidth= 2)
        
    # Title
    plt.title('Iteration number {:d}'.format(i+1))

    return

def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))

    # if plotting, set up the space for interactive graphs
    if plot_progress:
        plt.close()
        plt.ion()

    # Run K-Means
    for i in range(max_iters):
        
        # Output progress
        sys.stdout.write('\rK-Means iteration {:d}/{:d}...'.format(i+1, max_iters))
        sys.stdout.flush()
        
        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)
        
        # Optionally, plot progress here
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            plt.show()
        
        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)

    # Hold off if we are plotting progress
    print('\n')

    return centroids, idx

def kMeansInitCentroids(X, K):
    # You should return this values correctly
    centroids = np.zeros((K, X.shape[1]))
    
    # Initialize the centroids to be random examples
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]
    
    return centroids

## ================= Part 1: Find Closest Centroids ====================
#  To help you implement K-Means, we have divided the learning algorithm 
#  into two functions -- findClosestCentroids and computeCentroids. In this
#  part, you shoudl complete the code in the findClosestCentroids function. 
#
print('Finding closest centroids.\n')

# Load an example dataset that we will be using
mat = scipy.io.loadmat('ex7data2.mat')
X = mat["X"]

# Select an initial set of centroids
K = 3 # 3 Centroids
initial_centroids = np.array( [[3, 3], [6, 2], [8, 5]] )

# Find the closest centroids for the examples using the
# initial_centroids
idx = findClosestCentroids(X, initial_centroids)

print('Closest centroids for the first 3 examples: \n')
print(' {:s}, {:s}, {:s}'.format( str(float(idx[0])), str(float(idx[1])), str(float(idx[2])) ))
# adjusted next string for python's 0-indexing
print('\n(the closest centroids should be 0, 2, 1 respectively)\n')

print('=======================================================================')

## ===================== Part 2: Compute Means =========================
#  After implementing the closest centroids function, you should now
#  complete the computeCentroids function.
#
print('\nComputing centroids means.\n')

#  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids: \n')
print(' {:s} '.format(str(centroids)))
print('\n(the centroids should be\n')
print('   [ 2.428301 3.157924 ]')
print('   [ 5.813503 2.633656 ]')
print('   [ 7.119387 3.616684 ]\n')

print('=======================================================================')

## =================== Part 3: K-Means Clustering ======================
#  After you have completed the two functions computeCentroids and
#  findClosestCentroids, you have all the necessary pieces to run the
#  kMeans algorithm. In this part, you will run the K-Means algorithm on
#  the example dataset we have provided. 
#
print('\nRunning K-Means clustering on example dataset.\n\n')

# Load an example dataset
mat = scipy.io.loadmat('ex7data2.mat')
X = mat["X"]

# Settings for running K-Means
K = 3
max_iters = 10

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples (as can be seen in
# kMeansInitCentroids).
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters, True)
print('\nK-Means Done.\n')

print('=======================================================================')

## ============= Part 4: K-Means Clustering on Pixels ===============
#  In this exercise, you will use K-Means to compress an image. To do this,
#  you will first run K-Means on the colors of the pixels in the image and
#  then you will map each pixel on to it's closest centroid.
#  
#  You should now complete the code in kMeansInitCentroids.m
#

print('\nRunning K-Means clustering on pixels from an image.\n\n')

#  Load an image of a bird
mat = scipy.io.loadmat('bird_small.mat')
A = mat["A"]

A = A / 255.0 # Divide by 255 so that all values are in the range 0 - 1

# Size of the image
img_size = A.shape

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X = A.reshape(img_size[0] * img_size[1], 3, order='F').copy()

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16 
max_iters = 10

# When using K-Means, it is important the initialize the centroids
# randomly. 
# You should complete the code in kMeansInitCentroids.m before proceeding
initial_centroids = kMeansInitCentroids(X, K)

# Run K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters)

print('=======================================================================')


## ================= Part 5: Image Compression ======================
#  In this part of the exercise, you will use the clusters of K-Means to
#  compress an image. To do this, we first find the closest clusters for
#  each example. After that, we 

print('\nApplying K-Means to compress an image.\n')

# Find closest cluster members
idx = findClosestCentroids(X, centroids)

# Essentially, now we have represented the image X as in terms of the
# indices in idx. 

# We can now recover the image from the indices (idx) by mapping each pixel
# (specified by it's index in idx) to the centroid value
# X_recovered = centroids[idx,:]
m = np.shape( X )[0]
X_recovered = np.zeros( np.shape(X) )
for i in range( 0, m ):
		k 				= int(idx[i])
		X_recovered[i] 	= centroids[k]

# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3, order='F')

# Display the original image
plt.close()
plt.subplots(figsize=(8, 16))
plt.subplot(1, 2, 1)
plt.imshow(A) 
plt.title('Original')

# Display compressed image side by side
plt.subplot(1, 2, 2)
plt.imshow(X_recovered)
plt.title( 'Compressed, with {:d} colors.'.format(K) )
plt.show(block=False)

print('=======================================================================')