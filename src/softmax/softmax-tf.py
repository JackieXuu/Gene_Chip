#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf
import numpy as np
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


def next_batch_for_train(X_train,Y_train,batch_size):
    num_all=len(X_train)
    mask = np.random.permutation(num_all)
    X_train = X_train[mask]
    Y_train = Y_train[mask]
    X_new_train=[]
    Y_new_train=[]
    for i in range(batch_size):
        X_new_train.append(X_train[i])
        Y_new_train.append(Y_train[i])
    return X_new_train,Y_new_train


# Parameters
X,Y=readData("../../data/finalData.txt")
Y=one_hot(Y,79)
X_train, Y_train, X_test, Y_test=init(X,Y)
learning_rate = 0.01
training_epochs = 25
batch_size = 50
display_step = 1
# tf Graph Input
x = tf.placeholder(tf.float32, [None, 453]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 79]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([453, 79]))
b = tf.Variable(tf.zeros([79]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X_train)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch_for_train(X_train,Y_train,batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
            # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: X_test, y: Y_test}))
