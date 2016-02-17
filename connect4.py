import numpy as np
from board_game import LineBoardGame

class Connect4(LineBoardGame):
  def __init__(self, board_width=7, board_height=6, win_length=4):
    LineBoardGame.__init__(self, board_width, board_height, win_length)

  def read_action(self, player_name):
    return int(raw_input("%s: " % player_name)) - 1

  def get_actions(self):
    if self.final:
      return None
    actions = []
    for col in xrange(self.width):
      # Check if top column is empty, then we can move there
      if self.state[0][col] == 0:
          actions.append(col)
    return actions

  def perform_action(self, action):
    col, value = action
    assert col >= 0 and col < self.width
    assert value in self.values
    assert not self.state[0][col]
    for row in reversed(xrange(self.height)):
      if not self.state[row][col]:
        self.state[row][col] = value
        break
    return self.collect_reward()

  def reverse_action(self, action):
    col, value = action
    assert col >= 0 and col < self.width
    assert value in self.values
    for row in xrange(self.height):
      if self.state[row][col] == value:
        self.state[row][col] = 0
        self.final = False
        return
    assert False

if __name__ == "__main__":
  sim = Connect4()
  player = 1
  while True:
    sim.display()
    col = sim.read_action('Player %d' %  1)
    reward = sim.perform_action((col, player))
    if reward:
      print 'Player %d wins!' % player
      sim.display()
      break
    player = (2 if player == 1 else 1)
