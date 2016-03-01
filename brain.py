import os
import time

import numpy as np
import tensorflow as tf

NUM_CORES = 8
TENSORFLOW_CONFIG = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, 
  intra_op_parallelism_threads=NUM_CORES)

class Brain(object):
  def __init__(self, width=None, height=None, output_size=None,
               learning_rate=0.01, decay=0.9, load_path=None):
    self.width = width
    self.height = height
    self.input_channels = 4
    self.output_size = output_size

    # Hack to fix this problem: https://github.com/tensorflow/tensorflow/commit/430a054d6134f00e5188906bc4080fb7c5035ad5
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph, config=TENSORFLOW_CONFIG)

    with self.graph.as_default():
      # Initialize the weights
      self.init_weights()

      # tf Graph input
      self.X = tf.placeholder(tf.float32, [None, self.height, self.width, self.input_channels], name="X")
      self.Y = tf.placeholder(tf.float32, [None, 1], name="Y")
      self.dropout = tf.placeholder(tf.float32, name="dropout")

      # Softmax computes normalized scores for each action set
      self.actions_score = self.build_network(self.X, self.dropout)
      # We know the reward for only for one action in each state of the training set
      self.actions_mask = tf.placeholder(tf.float32, [None, self.output_size], name="actions_mask")
      self.masked_actions_score = tf.mul(self.actions_score, self.actions_mask, name="masked_actions_score")
      self.predicted_action = tf.argmax(self.masked_actions_score, dimension=1, name="predicted_actions")

      # Define loss and optimizer
      self.loss = tf.reduce_mean(tf.square(self.Y - self.masked_actions_score))
      self.train_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, name="train_opt").minimize(self.loss)

      # Initialize all variables
      self.sess.run(tf.initialize_all_variables())

      if load_path:
        self.eval_only = True
        self.load_model(load_path)
      else:
        self.eval_only = False

  def load_model(self, save_path):
    with self.graph.as_default():
      # Load saved weights from checkpoint
      saver = tf.train.Saver(tf.all_variables())
      saver.restore(self.sess, os.path.join(save_path, 'checkpoint.data'))

  def init_weights(self):
    # Store layers weight & bias
    self.weights = {
      # 4x4 conv, 3 input, 32 outputs
      'wc1': tf.Variable(tf.truncated_normal([8, 8, self.input_channels, 32], stddev=0.01)),
      # 4x4 conv, 32 inputs, 64 outputs
      'wc2': tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01)),
      # fully connected, 7*7*64 inputs, 1024 outputs
      'wc3': tf.Variable(tf.truncated_normal([2, 2, 64, 64], stddev=0.01)),
      # fully connected, 7*7*64 inputs, 1024 outputs
      'wd1': tf.Variable(tf.truncated_normal([self.width * self.height / 4, 512], stddev=0.01)),
      # 1024 inputs, 10 outputs (class prediction)
      'out': tf.Variable(tf.truncated_normal([512, self.output_size], stddev=0.01))
    }

    self.biases = {
      'bc1': tf.Variable(tf.constant(0.01, shape=[32])),
      'bc2': tf.Variable(tf.constant(0.01, shape=[64])),
      'bc3': tf.Variable(tf.constant(0.01, shape=[64])),
      'bd1': tf.Variable(tf.constant(0.01, shape=[512])),
      'out': tf.Variable(tf.constant(0.01, shape=[self.output_size]))
    }

  def conv2d(self, name, lhs, w, b, stride):
    conv = tf.nn.conv2d(lhs, w, strides=[1, stride, stride, 1], padding='SAME')
    return tf.nn.relu(tf.nn.bias_add(conv, b), name=name)

  def max_pool(self, name, lhs, k=2):
    return tf.nn.max_pool(lhs, ksize=[1, k, k, 1], strides=[1, k, k, 1],
        padding='SAME', name=name)

  def build_network(self, X, dropout):
    # Reshape input data
    X = tf.reshape(X, [-1, self.height, self.width, self.input_channels])

    # Convolution Layer
    conv1 = self.conv2d("conv1", X, self.weights['wc1'], self.biases['bc1'], 4)
     # Max Pooling (down-sampling)
    pool1 = self.max_pool('pool1', conv1, k=2)

    # Convolution Layer
    conv2 = self.conv2d("conv2", pool1, self.weights['wc2'], self.biases['bc2'], 2)
    # Max Pooling (down-sampling)
    # pool2 = self.max_pool('pool2', conv2, k=2)

    # Convolution Layer
    conv3 = self.conv2d("conv3", conv2, self.weights['wc3'], self.biases['bc3'], 1)
    # Max Pooling (down-sampling)
    # pool3 = self.max_pool('pool3', conv3, k=2)

    # Fully connected layers
    # Reshape conv2 output to fit dense layer input
    dense1 = tf.reshape(conv3, [-1, self.weights['wd1'].get_shape().as_list()[0]])
    # Relu activation
    dense1 = tf.nn.tanh(tf.add(tf.matmul(dense1, self.weights['wd1']), self.biases['bd1']))

    # Return the Q-value prediction for each action
    return tf.add(tf.matmul(dense1, self.weights['out']), self.biases['out'])

  def save(self, path):
    """Save the current model as a serialized proto for later use."""
    with  self.graph.as_default():
      folder = "model_" + str(int(time.time() * 1000) - 1456170641124)
      full_path = os.path.join(path, folder)
      tf.train.write_graph(self.sess.graph_def, full_path, "model.pb", False)
      saver = tf.train.Saver(tf.all_variables())
      saver.save(self.sess, os.path.join(full_path, "checkpoint.data"))
      return full_path

  def transform(self, state):
    new_state = []
    for line in state:
      new_line = []
      for x in line:
        #if x == 0:
        #  new_line.append([0])
        #elif x == 1:
        #  new_line.append([1])
        #else:
        #  new_line.append([2])
        if x == 0:
          new_line.append([1, 0, 0, 0])
        elif x == 1:
          new_line.append([0, 1, 0, 0])
        elif x == 2:
          new_line.append([0, 0, 1, 0])
        else:
          new_line.append([0, 0, 0, 1])

        # if x == 0:
        #  new_line.append([1, 0, 0])
        # elif x == 1:
        #  new_line.append([0, 1, 0])
        # else:
        #  new_line.append([0, 0, 1])
      new_state.append(new_line)
    return new_state

  def train(self, (observations, actions_mask, rewards)):
    if self.eval_only: # Don't train loaded models
      return

    with self.graph.as_default():
      rewards = [[x] for x in rewards]
      observations = [self.transform(obs) for obs in observations]
      actions_mask = [list(x) for x in actions_mask]
      #mask = []
      #for x, reward in enumerate(rewards):
      #  mask.append([reward if y else 0 for y in actions_mask[x]])
      #rewards = mask

      #print ''
      #for line in observations[0]:
      #  print line
      #print actions_mask[0]
      #print rewards[0]

      self.sess.run(self.train_opt, feed_dict={self.X: observations,
          self.Y: rewards, self.dropout: 0.8, self.actions_mask: actions_mask})

  def eval(self, observation, actions_mask):
    with self.graph.as_default():
      observation = [self.transform(observation)]
      actions_mask = [actions_mask]

      ret = self.sess.run(self.masked_actions_score, feed_dict={self.X: observation, self.dropout: 1.0, self.actions_mask: actions_mask})

      if False:
        print 'scores:'
        for idx, x in enumerate(ret[0]):
          if idx and idx % 4 == 0:
            print '' 
          print "%.3f" % x,
        print ''
        for line in observation[0]:
          print line
        print '\n'
      return ret[0]
