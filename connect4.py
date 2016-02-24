from board_game import LineBoardGame

class Connect4(LineBoardGame):
  """The classic connect4 game."""

  def __init__(self, board_width=7, board_height=6, win_length=4):
    self.actions_count = board_width
    LineBoardGame.__init__(self, board_width, board_height, win_length)

  def read_action(self, player_name):
    try:
      x = int(raw_input("%s: " % player_name)) - 1
      if not x in self.get_actions():
        raise ValueError
      return x
    except ValueError:
      print 'Invalid move! Allowed actions: %s', self.get_actions()
      return self.read_action(player_name)

  def get_actions(self):
    if self.final:
      return None
    actions = []
    for col in xrange(self.width):
      # Check if top column is empty, then we can move there
      if self.state[0][col] == 0:
        actions.append(col)
    return actions

  def get_actions_mask_from_action(self, action):
    actions = [0] * self.width
    actions[action] = 1
    return actions

  def get_action_from_index(self, index):
    return index

  def get_actions_mask(self):
    if self.final:
      return [0] * self.width
    actions = []
    for y in xrange(self.width):
      if self.state[0][y] == 0:
        actions.append(1)
      else:
        actions.append(0)
    return actions

  def perform_action(self, action):
    col, value = action
    assert col >= 0 and col < self.width
    assert value in self.values
    assert not self.state[0][col]
    for row in reversed(xrange(self.height)):
      if not self.state[row][col]:
        self.state[row][col] = value
        reward = self.collect_reward()
        break
    self.turn = self.next_turn()
    return reward

  def reverse_action(self, action):
    col, value = action
    assert col >= 0 and col < self.width
    assert value in self.values
    for row in xrange(self.height):
      if self.state[row][col] == value:
        self.state[row][col] = 0
        self.final = False
        self.turn = self.next_turn()
        return
    assert False

def play_game():
  """Simple class usage example."""
  sim = Connect4()
  player = 1
  while True:
    sim.display()
    col = sim.read_action("Player %d" %  1)
    reward = sim.perform_action((col, player))
    if reward:
      print "Player %d wins!" % player
      sim.display()
      break
    player = (2 if player == 1 else 1)

if __name__ == "__main__":
  play_game()
