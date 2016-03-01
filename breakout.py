import os
import sys

import numpy as np
from random import randint
from recordclass import recordclass
from getch import getch, pause

class Breakout(object):
  """The classic breakout game."""

  def __init__(self, width=32, height=32):
    self.width = width
    self.height = height
    self.actions_count = 3
    self.bar_len = self.width / 4
    self.reset()

  def read_action(self, step=0):
    # Map keys to actions ids
    actions_map = {'a': 0, 's': 1, 'd': 2}
    try:
      if step > 1:
        exit(0)
      ch = getch()
      x = actions_map.get(ch, -1)
      if not x in self.get_actions():
        raise ValueError
      return x
    except ValueError:
      print 'Invalid move! Allowed actions: %s', self.get_actions()
      return self.read_action(step=step+1)

  def get_actions(self):
    if self.final:
      return None
    return (0, 1, 2)

  def get_actions_mask_from_action(self, action):
    actions = [0, 0, 0]
    actions[action] = 1
    return actions

  def get_action_from_index(self, index):
    return index

  def get_actions_mask(self):
    if self.final:
      return (0, 0, 0)
    return (1, 1, 1)

  def _colide_ball(self, (x, y), (x1, y1, x2, y2)):
    if x1 <= x and x <= x2 and y1 <= y and y <= y2:
      self.ball_dir[1] *= -1
      return 1
    return 0

  def collect_reward(self):
    new_pos = (self.ball_pos[0] + self.ball_dir[0],
      self.ball_pos[1] + self.ball_dir[1])
    # Reverse on the x-axis
    reward = 0
    if new_pos[0] < 0 or new_pos[0] == self.width:
      self.ball_dir[0] *= -1
    # Reverse on the y-axis
    if new_pos[1] == 0 and self.bar_pos <= new_pos[0] and new_pos[0] < self.bar_pos + self.bar_len:
      self.ball_dir[1] *= -1
      reward = 1
    elif new_pos[1] < 0:
      self.reset_ball()
      reward = -5
      self.lives -= 1
    elif new_pos[1] == self.height:
      self.ball_dir[1] *= -1

    # Try to colide with the bars
    if not reward:
      for i, bar in enumerate(self.bars):
        reward = self._colide_ball(new_pos, bar)
        if reward:
          self.score += reward
          self.bars.pop(i)
          break

    self.ball_pos = (self.ball_pos[0] + self.ball_dir[0],
      self.ball_pos[1] + self.ball_dir[1])
    # Game has ended
    if not self.lives or len(self.bars) == 0:
      self.final = True
    return reward

  def perform_action(self, action):
    if action == 0:
      if self.bar_pos > 0:
        self.bar_pos -= 1
    elif action == 2:
      if self.bar_pos + self.bar_len + 1 < self.width:
        self.bar_pos += 1
    return self.collect_reward()

  def reverse_action(self, action):
    assert False, "Not suported"

  def reset_ball(self):
    self.ball_pos = [self.width / 2, self.height / 2]
    self.ball_dir = [1, np.random.choice([-1, 1])]

  def reset(self):
    self.final = False
    self.bars = []
    start_bars_y = int(self.height * 0.75)
    for x in xrange(0, self.width, 2):
      for y in xrange(start_bars_y, start_bars_y + 4):
        self.bars.append((x, y, x + 1, y))
    self.reset_ball()
    self.bar_pos = self.width / 2 - self.bar_len / 2
    self.score = 0
    self.lives = 3

  def get_state(self):
    self.state = [[0 for _ in xrange(self.width)] for _ in xrange(self.height)]
    for (x1, y1, x2, y2) in self.bars:
      self.state[y1][x1] = 1
      self.state[y2][x2] = 1
    x, y = self.ball_pos
    self.state[y][x] = 2
    for x in xrange(self.bar_len):
      self.state[0][self.bar_pos + x] = 3
    return tuple([tuple(x) for x in self.state])

  def get_score(self):
    return self.score

  def display(self):
    os.system('clear')
    print '-' * (self.width + 2)
    state = self.get_state()
    display_map = {0: ' ', 1: '@', 2: 'o', 3: '#'}
    for line in reversed(self.state):
      sys.stdout.write('|')
      for elem in line:
        sys.stdout.write(display_map[elem])
      print '|'
    print '-' * (self.width + 2)
    print 'Score: %d Lives: %d' % (self.score, self.lives)

def play_game():
  """Simple class usage example."""
  sim = Breakout()
  while True:
    sim.display()
    action = sim.read_action()
    sim.perform_action(action)
    if not sim.get_actions():
      print 'Game over!'
      break

if __name__ == "__main__":
  play_game()
