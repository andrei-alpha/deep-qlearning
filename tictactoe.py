import numpy as np
from board_game import LineBoardGame

class TicTacToe(LineBoardGame):
  def __init__(self, width=3, height=3, win_length=3):
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

  def perform_action(self, action):
    (x, y), value = action
    assert self._check_range([x], [y])
    assert value in self.values
    assert self.state[x][y] == 0
    self.state[x][y] = value
    self.turn = self.next_turn()
    return self.collect_reward()

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
