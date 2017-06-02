from __future__ import print_function

import numpy as np
import tensorflow as tf
import sklearn.metrics
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

def readData(filename):
        with open(filename, 'r') as f:
                string = [line.strip().split('\t') for line in f.readlines()]
                X = [map(float, line[:-1]) for line in string]
                Y = [int(line[-1]) for line in string]
        return np.array(X), np.array(Y)


def one_hot(Y,length):
    NewY=[]
    for i in range(len(Y)):
        content=[]
        num=Y[i]
        for i in range(num):
            content.append(0)
        content.append(1)
        for i in range(num+1,length):
            content.append(0)
        NewY.append(content)
    return np.array(NewY)


def init(X, Y):
    assert X.shape[0] == Y.shape[0], 'shape not match'
    num_all = X.shape[0]
    num_train = int(0.7 * num_all)
    num_test = num_all - num_train
    # shuffle
    mask = np.random.permutation(num_all)
    X = X[mask]
    Y = Y[mask]
    # training data
    mask_train = range(num_train)
    X_train = X[mask_train]
    Y_train = Y[mask_train]
    #testing data
    mask_test = range(num_train, num_all)
    X_test = X[mask_test]
    Y_test = Y[mask_test]
    print('All data shape: ', X.shape)
    print('Train data shape: ', X_train.shape)
    print('Train label shape: ', Y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test label shape: ', Y_test.shape)
    return X_train, Y_train, X_test, Y_test

X,Y=readData("../../data/finalData.txt")
Y=one_hot(Y,79)
X_train, Y_train, X_test, Y_test=init(X,Y)

xtr = tf.placeholder("float", [None, 453])
xte = tf.placeholder("float", [453])

distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
#distance = tf.reduce_sum(tf.sqrt(tf.square(tf.add(xtr, tf.negative(xte)))), reduction_indices=1)
pred = tf.arg_min(distance, 0)

accuracy = 0.
pred_class = []
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(X_test)):
        nn_index = sess.run(pred, feed_dict={xtr: X_train, xte: X_test[i, :]})
        print("Test", i, "Prediction:", np.argmax(Y_train[nn_index]), \
            "True Class:", np.argmax(Y_test[i]))
	pred_class.append(Y_train[nn_index])
        if np.argmax(Y_train[nn_index]) == np.argmax(Y_test[i]):
            accuracy += 1./len(X_test)
    print("Done!")
    print("Accuracy:", accuracy)
    print('F1 score: %f' % sklearn.metrics.f1_score(Y_test, np.array(pred_class), average='weighted'))
	
