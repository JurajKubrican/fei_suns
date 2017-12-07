# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import os

import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pickle_file = './dataset/notMNIST_small-0.1.pickle'
pickle_file = './dataset/notMNIST_small-1.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_data']
    train_labels = save['train_labels']
    valid_dataset = save['valid_data']
    valid_labels = save['valid_labels']
    test_dataset = save['test_data']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


##################PRETRENOVANIE


batch_size = 128
hiden_layer = 1024
first_n_batches = 3

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    l2Regular = tf.placeholder(tf.float32)

    # in layer
    weightsIn = tf.Variable(
        tf.truncated_normal([image_size * image_size, hiden_layer]))
    biasesIn = tf.Variable(tf.zeros([hiden_layer]))
    trainIn = tf.nn.relu(tf.matmul(tf_train_dataset, weightsIn) + biasesIn)

    # out layer
    weightsOut = tf.Variable(
        tf.truncated_normal([hiden_layer, num_labels]))
    biasesOut = tf.Variable(tf.zeros([num_labels]))

    # logits?
    logits = tf.matmul(trainIn, weightsOut) + biasesOut

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    # + \
    # l2Regular * \
    # (tf.nn.l2_loss(weightsIn) + tf.nn.l2_loss(weightsIn))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)

    validIn = tf.nn.relu(tf.matmul(tf_valid_dataset, weightsIn) + biasesIn)
    valid_prediction = tf.nn.softmax(tf.matmul(validIn, weightsOut) + biasesOut)

    testIn = tf.nn.relu(tf.matmul(tf_test_dataset, weightsIn) + biasesIn)
    test_prediction = tf.nn.softmax(tf.matmul(testIn, weightsOut) + biasesOut)

num_steps = 301

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = ((step % first_n_batches) * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 10 == 0):
            print("%.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
