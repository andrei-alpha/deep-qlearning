import numpy as np
import tensorflow as tf

class Brain(object):
  def __init__(self, width, height, learning_rate=0.01, decay=0.9):
    self.width = width
    self.height = height
    self.output_size = 1
    # tf Graph input
    self.X = tf.placeholder(tf.float32, [None, self.height, self.width])
    self.Y = tf.placeholder(tf.float32, [None, self.output_size])
    self.init_weights()
    self.pred = self.build_network(self.X)
    # Define loss and optimizer
    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.Y))
    self.train_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay).minimize(self.loss)
    self.sess = tf.Session()
    self.sess.run(tf.initialize_all_variables())

  def init_weights(self):
    # Store layers weight & bias
    self.weights = {
      # 5x5 conv, 1 input, 32 outputs
      'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32])), 
      # 5x5 conv, 32 inputs, 64 outputs
      'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])), 
      # fully connected, 7*7*64 inputs, 1024 outputs
      'wd1': tf.Variable(tf.random_normal([self.height * self.width * 64, 1024])), 
      # 1024 inputs, 10 outputs (class prediction)
      'out': tf.Variable(tf.random_normal([1024, self.output_size])) 
    }

    self.biases = {
      'bc1': tf.Variable(tf.random_normal([32])),
      'bc2': tf.Variable(tf.random_normal([64])),
      'bd1': tf.Variable(tf.random_normal([1024])),
      'out': tf.Variable(tf.random_normal([self.output_size]))
    }

  def conv2d(self, name, lhs, w, b):
    conv = tf.nn.conv2d(lhs, w, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(tf.nn.bias_add(conv, b), name=name)

  def max_pool(self, lhs, k=2):
    return tf.nn.max_pool(lhs, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

  def build_network(self, X):
    # Reshape input data
    X = tf.reshape(X, [-1, self.height, self.width, 1])

    # Convolution Layer
    conv1 = self.conv2d("conv1", X, self.weights['wc1'], self.biases['bc1'])
    # Max Pooling (down-sampling)
    # conv1 = max_pool(conv1, k=2)

    # Convolution Layer
    conv2 = self.conv2d("conv2", conv1, self.weights['wc2'], self.biases['bc2'])
    # Max Pooling (down-sampling)
    # conv2 = max_pool(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit dense layer input
    dense1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]]) 
    # Relu activation
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, self.weights['wd1']), self.biases['bd1']))

    # Output, value prediction
    out = tf.add(tf.matmul(dense1, self.weights['out']), self.biases['out'])
    return out

  def train(self, (observations, rewards)):
    rewards = [[x] for x in rewards]
    observations = [[list(x) for x in obs] for obs in observations]
    self.sess.run(self.train_opt, feed_dict={self.X: observations, self.Y: rewards})

  def eval(self, observation):
    observation = [[list(x) for x in observation]]
    return self.sess.run(self.pred, feed_dict={self.X: observation})[0]
