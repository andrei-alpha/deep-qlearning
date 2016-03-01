import os
from random import randint

class DummyGame(object):

  def __init__(self, width=32, height=32):
    self.width = width
    self.height = height
    self.len = 4
    self.actions_count = self.len
    self.score = 0
    self.color = randint(0, self.len)

  def get_actions(self):
    return range(self.len)

  def get_actions_mask_from_action(self, action):
    actions = [0] * self.len
    actions[action] = 1
    return actions

  def get_action_from_index(self, index):
    return index

  def get_actions_mask(self):
    return tuple([1] * self.len)

  def perform_action(self, action):
    if self.color == action:
      self.color = randint(0, self.len)
      self.score += 1
      return 1
    else:
      self.color = randint(0, self.len)
      self.score -= 1
      return -1

  def get_state(self):
    self.state = [[self.color for _ in xrange(self.width)] for _ in xrange(self.height)]
    return tuple([tuple(x) for x in self.state])

  def get_score(self):
    return self.score

  def display(self):
    os.system('clear')
    print 'Score:', self.score