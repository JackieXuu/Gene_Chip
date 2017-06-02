from __future__ import division, print_function
from softmax import Softmax
import numpy as np
import time
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
	print('All data shape: ', X.shape)
	print('Train data shape: ', X_train.shape)
	print('Train label shape: ', Y_train.shape)
	print('Test data shape: ', X_test.shape)
	print('Test label shape: ', Y_test.shape)
	return X_train, Y_train, X_test, Y_test


def train(X_train, Y_train, X_test, Y_test):
	learning_rates = [1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 1e-2, 3e-2, 5e-2, 8e-2]
	regularization_strengths = [1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 1e-3]
	results = {}
	loss = {}
	best_loss = None
	best_val = -1   
	best_train = -1
	best_lr = -1
	best_reg = -1
	best_softmax = None
	for i, rate in enumerate(learning_rates):
		for j, reg in enumerate(regularization_strengths):
			print('learning_rate: {} regularization: {}'.format(rate, reg))
			start = time.time()
			softmax = Softmax()
			loss_history, acc_history = softmax.train(X_train, Y_train, learning_rate=rate, reg=reg, num_iter=1500)
			Y_train_pred = softmax.test(X_train)
			Y_test_pred = softmax.test(X_test)
			learning_accuracy = np.mean(Y_train_pred == Y_train)	
			validation_accuracy = np.mean(Y_test_pred == Y_test)	
			if validation_accuracy > best_val:
				best_val = validation_accuracy
				best_y_pred = Y_test_pred
				best_train = learning_accuracy
				best_lr = rate
				best_reg = reg
				best_softmax = softmax
				best_loss = loss_history
			results[(rate, reg)] = (acc_history, learning_accuracy, validation_accuracy)
			loss[(rate, reg)] = loss_history
			print('Time: {}'.format(time.time() - start))
	return best_softmax, loss, best_train, best_val, best_lr, best_reg, best_y_pred, loss, results

def drawLossFig(loss, result, best_reg):
        import matplotlib.pyplot as plt
        learning_rate = []
        plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow', 'pink', 'magenta', 'black', 'cyan'])
        for (key, value) in loss.items():
                if best_reg in key:
                        lr = key[0]
                        learning_rate.append(lr)
                        loss_his = value
                        plt.plot(loss_his)
        plt.legend(['lr = %.3f'%learning_rate[0], 'lr = %.3f'%learning_rate[1], 'lr = %.3f'%learning_rate[2], 'lr = %.3f'%learning_rate[3], 'lr = %.3f'%learning_rate[4], 'lr = %.3f'%learning_rate[5], 'lr = %.3f'%learning_rate[6], 'lr = %.3f'%learning_rate[7]], loc='upper right')
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')
        #plt.show()
        plt.savefig('loss.png')
        plt.show()

def drawAccFig(result, best_reg):
        import matplotlib.pyplot as plt
        learning_rate = []
        plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow', 'pink', 'magenta', 'black', 'cyan'])
        for (key, value) in result.items():
                if best_reg in key:
                        lr = key[0]
                        learning_rate.append(lr)
                        acc_his = value[0]
                        plt.plot(acc_his)
        plt.legend(['lr = %.3f'%learning_rate[0], 'lr = %.3f'%learning_rate[1], 'lr = %.3f'%learning_rate[2], 'lr = %.3f'%learning_rate[3], 'lr = %.3f'%learning_rate[4], 'lr = %.3f'%learning_rate[5], 'lr = %.3f'%learning_rate[6], 'lr = %.3f'%learning_rate[7]], loc='lower right')
        plt.xlabel('Iteration number')
        plt.ylabel('Training Accuracy')
        plt.show()
        plt.savefig('acc.png')


if __name__ == '__main__':
	import sklearn.metrics
	filename = '../../data/finalData.txt'
	X, Y = readData(filename)
	X_train, Y_train, X_test, Y_test = init(X, Y)
	best_softmax, loss_history, best_train, best_val, best_lr, best_reg, best_y_pred, loss, result = train(X_train, Y_train, X_test, Y_test)
	print(best_y_pred)
	print('learning rate: %f regularization: %f' % (best_lr, best_reg))
	print('Training accuracy: %f' % best_train)
	print('Testing accuracy: %f'% best_val)
	print('F1 score: %f' % sklearn.metrics.f1_score(Y_test, best_y_pred, average='weighted'))
	drawLossFig(loss, result, best_reg)
	drawAccFig(result, best_reg)
