# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = './dataset/notMNIST_small-0.1.pickle'
pickle_file = './dataset/notMNIST_small-1.pickle'
pickle_file = './dataset/notMNIST_large-0.3.pickle'

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


def reformat(data, labels):
    dataset = data.reshape(
        (-1, image_size, image_size, 1)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

train_subset = 5000


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


batch_size = 16
hiden_layer = 40
kernel_size = 5
depth = 12

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size, image_size, 1))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    l2Regular = tf.placeholder(tf.float32)

    # CONV Layer 1
    conv_weigths_1 = tf.Variable(tf.truncated_normal(
        [kernel_size, kernel_size, 1, depth], stddev=0.1))
    conv_biases_1 = tf.Variable(tf.zeros([depth]))

    # # CONV Layer 2
    # conv_weigths_2 = tf.Variable(tf.truncated_normal(
    #     [kernel_size, kernel_size, depth, depth], stddev=0.1))
    # conv_biases_2 = tf.Variable(tf.constant(1.0, shape=[depth]))

    # CONV Layer 3
    # size = 28 * 28 * 12
    size = 8 * 8 * 12
    conv_weigths_3 = tf.Variable(tf.truncated_normal(
        [size, size, depth, depth], stddev=0.1))
    conv_biases_3 = tf.Variable(tf.constant(1.0, shape=[depth]))

    padding = 1
    conv_size = 2
    stride = 2
    kernel_size = 2

    # size = image_size // (conv_size * conv_size)
    # size = size * image_size // (conv_size * conv_size)

    size = 4 * 4 * 12

    # FC Layer 1
    fc_weights_1 = tf.Variable(tf.truncated_normal(
        [size, hiden_layer], stddev=0.1))
    fc_biases_1 = tf.Variable(tf.constant(1.0, shape=[hiden_layer]))

    # # FC Layer 2
    # fc_weights_2 = tf.Variable(tf.truncated_normal(
    #     [hiden_layer, hiden_layer], stddev=0.1))
    # fc_biases_2 = tf.Variable(tf.constant(1.0, shape=[hiden_layer]))

    # FC Layer 3
    fc_weights_3 = tf.Variable(tf.truncated_normal(
        [hiden_layer, num_labels], stddev=0.1))
    fc_biases_3 = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    # LOGITS
    def model(data):
        layer = tf.nn.relu(tf.nn.conv2d(data, conv_weigths_1, [1, 1, 1, 1], padding='VALID') + conv_biases_1)
        print(layer.get_shape().as_list())
        layer = tf.nn.avg_pool(layer, [1, 3, 3, 1], [1, 3, 3, 1], padding='SAME')
        print(layer.get_shape().as_list())
        # layer = tf.nn.max_pool(layer, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        # layer = tf.nn.max_pool(layer, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        # layer = tf.nn.relu(tf.nn.conv2d(layer, conv_weigths_2, [1, 1, 1, 1], padding='SAME') + conv_biases_2)
        layer = tf.nn.relu(tf.nn.conv2d(layer, conv_weigths_3, [1, 1, 1, 1], padding='SAME') + conv_biases_3)
        print(layer.get_shape().as_list())
        layer = tf.nn.avg_pool(layer, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        print(layer.get_shape().as_list())
        shape = layer.get_shape().as_list()
        reshape = tf.reshape(layer, [shape[0], shape[1] * shape[2] * shape[3]])

        layer = tf.nn.relu(tf.matmul(reshape, fc_weights_1) + fc_biases_1)
        # layer = tf.matmul(layer, fc_weights_2) + fc_biases_2
        return tf.matmul(layer, fc_weights_3) + fc_biases_3


    train_prediction = tf.nn.softmax(model(tf_train_dataset))
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=model(tf_train_dataset)))
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

num_steps = 2001
reluNum = 0.005

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    print(train_subset)
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
