# -*- coding: utf-8 -*-
"""
Creator: Abdalla H.
Created on: 26 Sep. 2020
"""

# Import Libraries
import re
import csv
import nltk
import pickle
import numpy as np
from sklearn import svm
from scipy.io import loadmat

# Ignore overflow and divide by zero of np.log() and np.exp()
# np.seterr(divide = 'ignore')
# np.seterr(over = 'ignore') 

def processEmail( email_contents ):
    vocab_list = getVocabList()
    
    word_indices = []
    
    email_contents = email_contents.lower()
    email_contents = re.sub( '<[^<>]+>', ' ', email_contents )
    email_contents = re.sub( '[0-9]+', 'number', email_contents )
    email_contents = re.sub( '(http|https)://[^\s]*', 'httpaddr', email_contents )
    email_contents = re.sub( '[^\s]+@[^\s]+', 'emailaddr', email_contents )
    email_contents = re.sub( '[$]+', 'dollar', email_contents )
    
    stemmer = nltk.stem.porter.PorterStemmer()
    tokens = re.split( '[ ' + re.escape("@$/#.-:&*+=[]?!(){},'\">_<;%") + ']' , email_contents )
    
    for token in tokens:
        token = re.sub( '[^a-zA-Z0-9]', '', token )
        token = stemmer.stem( token.strip() )

        if len(token) == 0:
            continue

        if token in vocab_list:
            word_indices.append( vocab_list[token] )
            
    return word_indices

def getVocabList():
    vocab_list = {}
    reader = csv.reader(open('vocab.txt', 'r'), delimiter= '\t')
    for row in reader:
        vocab_list[row[1]] = int(row[0])
    
    return vocab_list

def emailFeatures(word_indices):
    features = np.zeros((1899, 1))
    
    for index in word_indices:
        features[index] = 1
        
    return features

def part1():
    print(' Part 1 '.center(80, '='))
    email_contents = ''
    email_contents = open( 'emailSample1.txt', 'r' ).read()
    # print(email_contents)
    
    word_indices = processEmail( email_contents )
    print('Done!')

def part2():
    print(' Part 2 '.center(80, '='))
    email_contents = ''
    email_contents = open( 'emailSample1.txt', 'r' ).read()
    # print(email_contents)
    
    word_indices = processEmail( email_contents )
    features      = emailFeatures( word_indices )
    print('Done!')

def part3():
    print(' Part 3 '.center(80, '='))
    
    spamTrain = loadmat('spamTrain.mat')
    X, y = spamTrain['X'], spamTrain['y']
    
    # linear_svm = pickle.load( open("linear_svm.svm", "rb") )

    linear_svm = svm.SVC(C=0.1, kernel='linear')
    linear_svm.fit( X, y.ravel() )
    pickle.dump( linear_svm, open("linear_svm.svm", "wb") )

    predictions = linear_svm.predict( X )
    predictions = predictions.reshape( np.shape(predictions)[0], 1 )
    print('Train Accuracy = %', ( predictions == y ).mean() * 100.0)

    mat = loadmat('spamTest.mat')
    X_test, y_test = mat['Xtest'], mat['ytest']

    predictions = linear_svm.predict( X_test )
    predictions = predictions.reshape( np.shape(predictions)[0], 1 )
    print('Test Accuracy = %', ( predictions == y_test ).mean() * 100.0)

    vocab_list = getVocabList()
    reversed_vocab_list = dict( (v, k) for k, v in vocab_list.items() )
    sorted_indices = np.argsort( linear_svm.coef_, axis=None )
    
    print()
    for i in sorted_indices[0:15]:
        print(reversed_vocab_list[i])
    
    print('\nDone!')

def part4():
    print(' Part 4 '.center(80, '='))
    data = loadmat('spamTrain.mat')
    X, y = data['X'], data['y']

    # linear_svm = pickle.load( open("linear_svm.svm", "rb") )
    linear_svm = svm.SVC(C= 0.1, kernel= 'linear')
    linear_svm.fit( X, y.ravel() )
    # pickle.dump( linear_svm, open("linear_svm.svm", "wb") )

    email_contents = open('spamSample2.txt', 'r').read()

    word_indices = processEmail( email_contents )
    features     = emailFeatures( word_indices ).transpose()

    print('Prediction of spamSample2.txt = ', linear_svm.predict( features )[0], '       (1 for spam, 0 for not a spam)')
    print('Done!')

def main():
    part1()
    part2()
    part3()
    part4()

if __name__ == '__main__':
    main()