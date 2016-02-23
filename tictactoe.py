import numpy as np
from board_game import LineBoardGame

class TicTacToe(LineBoardGame):
  def __init__(self, width=3, height=3, win_length=3):
    self.actions_count = width * height
    LineBoardGame.__init__(self, width, height, win_length)

  def read_action(self, player_name):
    x,y = raw_input("%s: " % player_name).split()
    return int(x) - 1, int(y) - 1

  def get_actions(self):
    if self.final:
      return None
    actions = []
    for x in xrange(self.height):
      for y in xrange(self.width):
        if self.state[x][y] == 0:
          actions.append((x, y))
    return actions

  def get_actions_mask_from_action(self, action):
    x, y = action
    actions = [0] * self.width * self.height
    actions[x * self.height + y] = 1
    return actions

  def get_action_from_index(self, index):
    return (index / self.height, index % self.width)

  def get_actions_mask(self):
    if self.final:
      return [0] * self.width * self.height
    actions = []
    for x in xrange(self.height):
      for y in xrange(self.width):
        if self.state[x][y] == 0: # allowed
          actions.append(1)
        else:
          actions.append(0)
    return actions

  def perform_action(self, action):
    (x, y), value = action
    assert self._check_range([x], [y])
    assert value in self.values
    assert self.state[x][y] == 0
    self.state[x][y] = value
    self.turn = self.next_turn()
    return self.collect_reward()

  #@staticmethod
  #def new_state(state, action):
  #  (x, y), value = action
  #  ret = (state[x][:y] + (value,) + state[x][y+1:],)
  #  return tuple(state[:x] + ret + state[x+1:])

  def reverse_action(self, action):
    (x, y), value = action
    assert self._check_range([x], [y])
    assert value in self.values
    assert self.state[x][y] == value
    self.state[x][y] = 0
    self.final = False
    self.next_turn()

if __name__ == "__main__":
  sim = TicTacToe()
  player = 1
  while True:
    sim.display()
    x,y = sim.read_action('Player %d' % player)
    reward = sim.perform_action(((x, y), player))
    if reward:
      print 'Player %d wins!' % player
      sim.display()
      break
    player = (2 if player == 1 else 1)
