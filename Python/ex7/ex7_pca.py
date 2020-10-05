# -*- coding: utf-8 -*-
"""
Creator: Abdalla H.
Created on: 30 Sep 2020
"""

# Import Libraries
import sys
import scipy.io
import numpy as np
import scipy.linalg as linalg
import matplotlib.colors
import matplotlib.pyplot as plt

def hsv(n=63):
    # return colors.hsv_to_rgb( np.column_stack([ np.array(range(n+1)).T / float(n), np.ones( ((n+1), 2) ) ]) )
    return matplotlib.colors.hsv_to_rgb( np.column_stack([ np.linspace(0, 1, n+1) , np.ones( ((n+1), 2) ) ]) )

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

def pca(X):
    # Useful values
    m, n = X.shape

    # You need to return the following variables correctly.
    U = np.zeros(n)
    S = np.zeros(n)
    
    # compute the covariance matrix
    sigma = (1.0/m) * (X.T).dot(X)

    # compute the eigenvectors (U) and S
    U, S, Vh = np.linalg.svd(sigma)
    S = linalg.diagsvd(S, len(S), len(S))

    return U, S

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm = X_norm/sigma

    return X_norm, mu, sigma

def projectData(X, U, K):
    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))
    
    # get U_reduce for only the desired K
    U_reduce = U[:,:K]
    
    Z = X.dot(U_reduce)
    
    return Z

def recoverData(Z, U, K):
    # You need to return the following variables correctly.
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    
    # get U_reduce for only the desired K
    U_reduce = U[:,:K]

    # recover data
    X_rec = Z.dot(U_reduce.T)
    
    return X_rec

def displayData(X, example_width=None):
    # turns 1D X array into 2D
    if X.ndim == 1:
        X = np.reshape(X, (-1,X.shape[0]))

    # Set example_width automatically if not passed in
    if not example_width or not 'example_width' in locals():
        example_width = int(round(np.sqrt(X.shape[1])))

    # Gray Image
    plt.set_cmap("gray")

    # Compute rows, cols
    m, n = X.shape
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = -np.ones((pad + display_rows * (example_height + pad),  pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 1
    for j in range(1,display_rows+1):
        for i in range (1,display_cols+1):
            if curr_ex > m:
                break
        
            # Copy the patch
            
            # Get the max value of the patch to normalize all examples
            max_val = max(abs(X[curr_ex-1, :]))
            rows = pad + (j - 1) * (example_height + pad) + np.array(range(example_height))
            cols = pad + (i - 1) * (example_width  + pad) + np.array(range(example_width ))
            
            display_array[rows[0]:rows[-1]+1 , cols[0]:cols[-1]+1] = np.reshape(X[curr_ex-1, :], (example_height, example_width), order="F") / max_val
            curr_ex += 1
    
        if curr_ex > m:
            break

    # Display Image
    h = plt.imshow(display_array, vmin=-1, vmax=1)

    # Do not show axis
    plt.axis('off')

    plt.show(block=False)

    return h, display_array

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize
#
print('Visualizing example dataset for PCA.\n');

#  The following command loads the dataset. You should now have the 
#  variable X in your environment
mat = scipy.io.loadmat('ex7data1.mat')
X = np.array(mat["X"])

# interactive graphs
plt.ion()

#  Visualize the example dataset
plt.close()

# kept the scatter() (vs. the plot()) version 
#  because scatter() makes properly circular markers
# plt.plot(X[:, 0], X[:, 1], 'o', markersize=9, markeredgewidth=1, markeredgecolor='b', markerfacecolor='None')
plt.subplots(figsize=(8,8))
plt.scatter(X[:,0], X[:,1], s=75, facecolors='none', edgecolors='b')
plt.axis([0.5, 6.5, 2, 8])
plt.gca().set_aspect('equal', adjustable='box')
plt.show(block=False)


print('=======================================================================')


## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
print('Running PCA on example dataset.\n');

#  Before running PCA, it is important to first normalize X
X_norm, mu, _ = featureNormalize(X)

#  Run PCA
U, S = pca(X_norm)

#  Compute mu, the mean of the each feature

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
plt.subplots(figsize=(8,8))
plt.scatter(X[:,0], X[:,1], s=75, facecolors='none', edgecolors='b')
plt.axis([0.5, 6.5, 2, 8])
plt.gca().set_aspect('equal', adjustable='box')
drawLine(mu, mu + 1.5 * S[0,0] * U[:,0].T, c='k', linewidth=2)
drawLine(mu, mu + 1.5 * S[1,1] * U[:,1].T, c='k', linewidth=2)
plt.show()

print('Top eigenvector: \n')
print(' U(:,1) = {:f} {:f} \n'.format(U[0,0], U[1,0]))
print('(you should expect to see -0.707107 -0.707107)')


print('=======================================================================')


## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the 
#  first k eigenvectors. The code will then plot the data in this reduced 
#  dimensional space.  This will show you what the data looks like when 
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m
#
print('Dimension reduction on example dataset.\n');

#  Plot the normalized dataset (returned from pca)
plt.close()
plt.subplots(figsize=(8,8))
plt.scatter(X_norm[:,0], X_norm[:,1], s=75, facecolors='none', edgecolors='b')
plt.axis([-4, 3, -4, 3])
plt.gca().set_aspect('equal', adjustable='box')
# plt.show(block=False)

#  Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example: {:s}\n'.format(str(Z[0])))
print('(this value should be about 1.481274)\n')

X_rec  = recoverData(Z, U, K)
print('Approximation of the first example: {:f} {:f}\n'.format(X_rec[0, 0], X_rec[0, 1]))
print('(this value should be about  -1.047419 -1.047419)\n')

#  Draw lines connecting the projected points to the original points
# plt.hold(True)
plt.scatter(X_rec[:, 0], X_rec[:, 1], s=75, facecolors='none', edgecolors='r')
for i in range(X_norm.shape[0]):
    drawLine(X_norm[i,:], X_rec[i,:], linestyle='--', color='k', linewidth=1)

plt.show()
# plt.hold(False)


print('=======================================================================')


## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print('Loading face dataset.\n');

#  Load Face dataset
mat = scipy.io.loadmat('ex7faces.mat')
X = np.array(mat["X"])

#  Display the first 100 faces in the dataset
plt.subplots(figsize=(8,8))
displayData(X[:100, :])
plt.show()

print('=======================================================================')

## =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#
print('Running PCA on face dataset.\n(this mght take a minute or two ...)\n')

#  Before running PCA, it is important to first normalize X by subtracting 
#  the mean value from each feature
X_norm, _, _ = featureNormalize(X)

#  Run PCA
U, S = pca(X_norm)

#  Visualize the top 36 eigenvectors found
plt.subplots(figsize=(8,8))
displayData(U[:, :36].T)
plt.show()

print('=======================================================================')


## ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors 
#  If you are applying a machine learning algorithm 
print('Dimension reduction for face dataset.\n');

K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of: ')
print('{:d} {:d}'.format(Z.shape[0], Z.shape[1]))


print('=======================================================================')


## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and 
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

print('Visualizing the projected (reduced dimension) faces.\n');

K = 100;
X_rec  = recoverData(Z, U, K)

# Display normalized data
plt.close()
plt.subplots(figsize=(8,8))
displayData(X_norm[:100,:])
plt.title('Original faces')
plt.gca().set_aspect('equal', adjustable='box')

# Display reconstructed data from only k eigenfaces
plt.close()
plt.subplots(figsize=(8,8))
displayData(X_rec[:100,:])
plt.title('Recovered faces')
plt.gca().set_aspect('equal', adjustable='box')


print('=======================================================================')


## === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  One useful application of PCA is to use it to visualize high-dimensional
#  data. In the last K-Means exercise you ran K-Means on 3-dimensional 
#  pixel colors of an image. We first visualize this output in 3D, and then
#  apply PCA to obtain a visualization in 2D.

plt.close()

# Re-load the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first

# A = double(imread('bird_small.png'));
mat = scipy.io.loadmat('bird_small.mat')
A = mat["A"]

# from ex7.py, part 4
A = A / 255.0
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], 3, order='F').copy()
K = 16 
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
#  use flatten(). otherwise, Z[sel, :] yields array w shape [1000,1,2]
sel = np.floor(np.random.rand(1000, 1) * X.shape[0]).astype(int).flatten()

#  Setup Color Palette
palette = hsv(K)
colors = np.array([palette[int(i)] for i in idx[sel]])

#  Visualize the data and centroid memberships in 3D
fig1 = plt.figure(1, figsize=(8,8))
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], s=50, c=colors)
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
plt.show(block=False)

print('=======================================================================')

## === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
X_norm, _, _ = featureNormalize(X)

# PCA and project the data to 2D
U, S = pca(X_norm)
Z = projectData(X_norm, U, 2)

# Plot in 2D
fig2 = plt.figure(2, figsize=(8,8))
plotDataPoints(Z[sel, :], idx[sel], K)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');
plt.show(block=False)

print('=======================================================================')