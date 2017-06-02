from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import os
import pprint
import sys
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.utils import to_categorical
from keras import regularizers, initializers
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau 
from keras.layers.normalization import BatchNormalization

pp = pprint.PrettyPrinter()
flags = tf.app.flags
flags.DEFINE_string("gpu", "0", "GPU(s) to use. [0]")
flags.DEFINE_float("learning_rate", 2.5e-3, "Learning rate [2.5e-4]")
flags.DEFINE_integer("batch_size", 200, "The number of batch images [4]")
flags.DEFINE_integer("save_step", 500, "The interval of saveing checkpoints[500]")
flags.DEFINE_string("checkpoint", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log", "summary", "log [log]")
flags.DEFINE_integer("epoch", 100, "Epoch[10]")


FLAGS = flags.FLAGS
os.environ["CUDA_VISIABLE_DEVICES"] = FLAGS.gpu


def readData(filename):
        with open(filename, 'r') as f:
                string = [line.strip().split('\t') for line in f.readlines()]
                X = [map(float, line[:-1]) for line in string]
                Y = [int(line[-1]) for line in string]
        return np.array(X), np.array(Y)


def init(X, Y):
        assert X.shape[0] == Y.shape[0], 'shape not match'
        num_all = X.shape[0]
        num_train = int(0.8 * num_all)
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
	# Y_train, Y_test = np.expand_dims(Y_train, axis=1), np.expand_dims(Y_test, axis=1)
	print('All data shape: ', X.shape)
        print('Train data shape: ', X_train.shape)
        print('Train label shape: ', Y_train.shape)
        print('Test data shape: ', X_test.shape)
        print('Test label shape: ', Y_test.shape)
        return X_train, Y_train, X_test, Y_test


def add_initializer(model, kernel_initializer = initializers.random_normal(stddev=0.01), bias_initializer = initializers.Zeros()):
	for layer in model.layers:
		if hasattr(layer, "kernel_initializer"):
			layer.kernel_initializer = kernel_initializer
		if hasattr(layer, "bias_initializer"):
			layer.bias_initializer = bias_initializer


def add_regularizer(model, kernel_regularizer = regularizers.l2(), bias_regularizer = regularizers.l2()):
	for layer in model.layers:
		if hasattr(layer, "kernel_regularizer"):
			layer.kernel_regularizer = kernel_regularizer
		if hasattr(layer, "bias_regularizer"):
			layer.bias_regularizer = bias_regularizer

def genChipModel():
	inputs = Input(shape = (453, ))
	hidden1 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(256, activation = 'relu')(inputs)))
	hidden2 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(512, activation = 'relu')(hidden1)))
	#hidden3 = Dropout(0.5)(Dense(512, activation = 'relu')(hidden2))
	hidden3 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(512, activation = 'relu')(hidden2)))
	hidden4 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(256, activation = 'relu')(hidden3)))
	#hidden5 = Dropout(0.5)(Dense(128, activation = 'relu')(hidden4))
	predictions = Dense(79, activation = 'softmax')(hidden4)
	model = Model(inputs = inputs, outputs = predictions)
	add_regularizer(model)
	return model


def main(_):
	pp.pprint(flags.FLAGS.__flags)
	sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
	if not os.path.isdir(FLAGS.checkpoint):
		os.mkdir(FLAGS.checkpoint)
	if not os.path.isdir(FLAGS.log):
		os.mkdir(FLAGS.log)
	model = genChipModel()
	model.summary()
	
	opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])#'categorical_crossentropy', metrics=['accuracy'])
	
	filename = '../../data/finalData.txt'	
	x, y = readData(filename)
	x_train, y_train, x_test, y_test = init(x, y)
	
	y_train_labels = to_categorical(y_train, num_classes=79)
	y_test_labels = to_categorical(y_test, num_classes=79)	
	model_path = os.path.join(FLAGS.checkpoint, "weights.hdf5")
	callbacks = [
		ModelCheckpoint(filepath=model_path, monitor="val_acc", save_best_only=True, save_weights_only=True),
		TensorBoard(log_dir=FLAGS.log),
		ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2)
	]
	hist = model.fit(x_train, y_train_labels, epochs=FLAGS.epoch, batch_size=100, validation_data=(x_test, y_test_labels), callbacks=callbacks)

	loss, accuracy = model.evaluate(x_test, y_test_labels, batch_size=100, verbose=1)



if __name__ == '__main__':
	tf.app.run()
