from __future__ import print_function
import numpy as np

class Softmax():
	def __init__(self):
		self.W = None

	def train(self, x, y, learning_rate=1e-3, reg = 1e-5, num_iter=1500, batch_size=200):
		num_train, num_feature = x.shape
		num_classes = np.max(y) + 1
		if self.W == None:
			self.W = np.random.randn(num_feature, num_classes)
		
		loss_history = []
		acc_history = []
		for iter in range(num_iter):
			indices = np.random.choice(num_train, batch_size)
			x_batch = x[indices]
			y_batch = y[indices]
			loss, grad = self.loss(x_batch, y_batch, reg)
			acc = self.accuracy(x_batch, y_batch)
			loss_history.append(loss)
			acc_history.append(acc)
			self.W += -learning_rate * grad
			
			if np.mod(iter, 100) == 0:
				print("iteration {}/{} loss: {:.7f}".format(iter, num_iter, loss))
		return loss_history, acc_history

        def accuracy(self, x, y):
                y_pred = np.zeros(x.shape[0])
                scores = np.dot(x, self.W)
                y_pred = np.argmax(scores, axis = 1)
                return np.mean(y_pred == y)


	def test(self, x):
		y_pred = np.zeros(x.shape[0])
		scores = np.dot(x, self.W)
		y_pred = np.argmax(scores, axis = 1)
		return y_pred
	
	def loss(self, x, y, reg):
		loss = 0.0
		dW = np.zeros(self.W.shape)
		num_feature, num_class = self.W.shape
		num_train = x.shape[0]
		# compute loss
		scores = np.dot(x, self.W)
		scores = (scores.T - np.max(scores, axis = 1)).T
		correct_class = scores[np.arange(num_train), y]
		loss = -np.sum(correct_class) + np.sum(np.log(np.sum(np.exp(scores), axis = 1)))
		loss = loss / num_train + 0.5 * reg * np.sum(self.W*self.W)
		# compute gradient
		scores_exp = np.exp(scores)
		sum = scores_exp / (np.sum(scores_exp, axis = 1).reshape(-1, 1))
		sum[np.arange(num_train), y] -= 1
		dW = (x.T).dot(sum) / num_train + reg * self.W
		return loss, dW

if __name__ == "__main__":
	softmax = Softmax()
	x = np.random.randn(200, 5000)
	y = np.arange(200)
	print(x.shape, y.shape)
	softmax.train(x, y)	
		
