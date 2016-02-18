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
    self.dropout = tf.placeholder(tf.float32)
    self.init_weights()
    self.pred = self.build_network(self.X, self.dropout)
    # Define loss and optimizer
    self.loss = tf.reduce_mean(tf.square(self.Y - self.pred))
    self.train_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay).minimize(self.loss)
    self.sess = tf.Session()
    self.sess.run(tf.initialize_all_variables())

  def init_weights(self):
    # Store layers weight & bias
    self.weights = {
      # 4x4 conv, 1 input, 32 outputs
      'wc1': tf.Variable(tf.random_normal([4, 4, 1, 32], stddev=0.1)),
      # 4x4 conv, 32 inputs, 64 outputs
      'wc2': tf.Variable(tf.random_normal([4, 4, 32, 64], stddev=0.1)),
      # fully connected, 7*7*64 inputs, 1024 outputs
      'wd1': tf.Variable(tf.random_normal([self.height * self.width * 64, 1024], stddev=0.1)),
      # 1024 inputs, 10 outputs (class prediction)
      'out': tf.Variable(tf.random_normal([1024, self.output_size], stddev=0.1))
    }

    self.biases = {
      'bc1': tf.Variable(tf.zeros([32])),
      'bc2': tf.Variable(tf.zeros([64])),
      'bd1': tf.Variable(tf.zeros([1024])),
      'out': tf.Variable(tf.zeros([self.output_size]))
    }

  def conv2d(self, name, lhs, w, b):
    conv = tf.nn.conv2d(lhs, w, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(tf.nn.bias_add(conv, b), name=name)

  def norm(self, name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

  def max_pool(self, name, lhs, k=2):
    return tf.nn.max_pool(lhs, ksize=[1, k, k, 1], strides=[1, k, k, 1],
        padding='SAME', name=name)

  def build_network(self, X, dropout):
    # Reshape input data
    X = tf.reshape(X, [-1, self.height, self.width, 1])

    # Convolution Layer
    conv1 = self.conv2d("conv1", X, self.weights['wc1'], self.biases['bc1'])
     # Max Pooling (down-sampling)
    # pool1 = self.max_pool('pool1', conv1, k=2)
    # Apply Normalization
    norm1 = self.norm('norm1', conv1, lsize=4)
    # Apply Dropout
    drop1 = tf.nn.dropout(norm1, dropout)

    # Convolution Layer
    conv2 = self.conv2d("conv2", drop1, self.weights['wc2'], self.biases['bc2'])
    # Max Pooling (down-sampling)
    # pool2 = self.max_pool('pool2', conv2, k=2)
    # Apply Normalization
    norm2 = self.norm('norm2', conv2, lsize=4)
    # Apply Dropout
    drop2 = tf.nn.dropout(norm2, dropout)

    # Fully connected layer
    # Reshape conv2 output to fit dense layer input
    dense1 = tf.reshape(drop2, [-1, self.weights['wd1'].get_shape().as_list()[0]]) 
    # Relu activation
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, self.weights['wd1']), self.biases['bd1']))

    # Output, value prediction
    out = tf.add(tf.matmul(dense1, self.weights['out']), self.biases['out'])
    return out

  def train(self, (observations, rewards)):
    rewards = [[x] for x in rewards]
    observations = [[list(x) for x in obs] for obs in observations]
    self.sess.run(self.train_opt, feed_dict={self.X: observations,
        self.Y: rewards, self.dropout: 0.8})

  def eval(self, observation):
    observation = [list(x) for x in observation]
    res = self.sess.run(self.pred, feed_dict={self.X: observation, self.dropout: 1.0})
    return res[0];
