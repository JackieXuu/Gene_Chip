from sklearn import tree
import sklearn
import numpy as np
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
	print('All class: ', np.unique(Y))	
	print('All data shape: ', X.shape)
	print('Train data shape: ', X_train.shape)
	print('Train label shape: ', Y_train.shape)
	print('Test data shape: ', X_test.shape)
	print('Test label shape: ', Y_test.shape)
	return X_train, Y_train, X_test, Y_test
if __name__ == '__main__':
	filename = '../../data/finalData.txt'
	X, Y = readData(filename)
	X_train, Y_train, X_test, Y_test = init(X, Y)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, Y_train)
	Y_predict=clf.predict(X_test)
	total=0.
	right=0.
	for i in range(len(Y_predict)):
		if(Y_predict[i]==Y_test[i]):
			right+=1
		total+=1
	print right/total
	print('F1 score: %f' % sklearn.metrics.f1_score(Y_test, Y_predict, average='weighted'))
