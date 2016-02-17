class LineBoardGame(object):
  def __init__(self, width, height, win_length):
    self.width = width
    self.height = height
    self.win_length = win_length
    # We represent the board using a vector of 6x7 integers
    self.final = False
    self.state = [[0 for _ in xrange(self.width)] for _ in xrange(self.height)]
    # Two player game, the board can have either 0, 1 or 2
    self.values = [1, 2]
    self.turn = 1
    self.encoding = {0: ' ', 1: 'X', 2: '0'}

  def _check_range(self, xxs, yys):
    return min(min(xxs), min(yys)) >= 0 and max(xxs) < self.height and max(yys) < self.width

  def _check_line(self, x, y, xx, yy):
    xxs = [x + xx * k for k in xrange(0, self.win_length)]
    yys = [y + yy * k for k in xrange(0, self.win_length)]
    if not self._check_range(xxs, yys):
      return 0
    return self.state[x][y] != 0 and all([self.state[xx][yy] == self.state[x][y] for (xx,yy) in zip(xxs, yys)])

  def collect_reward(self):
    # Check for any line of <win_line> values of the same type
    xx = [-1, -1, 0, 1, 1, 1, 0, -1]
    yy = [0, 1, 1, 1, 0, -1, -1, -1]
    for x in xrange(self.height):
      for y in xrange(self.width):
        if self.state[x][y] == 0:
          continue
        # It's enough to check only 4 directions, the rest are just reversed
        for k in xrange(4):
          if self._check_line(x, y, xx[k], yy[k]):
            self.final = True
            return 100
    return 0

  def next_turn(self):
    return (1 if self.turn == 2 else 2)

  def reset(self):
    # Reset the board
    self.final = False
    self.turn = 1
    self.state = [[0 for _ in xrange(self.width)] for _ in xrange(self.height)]

  def get_state(self):
    return tuple([tuple(x) for x in self.state])

  def read_action(self):
    assert not 'You should override this method'

  def get_actions(self):
    assert not 'You should override this method'

  def perform_action(self):
    assert not 'You should override this method'

  def reverse_action(self):
    assert not 'You should override this method'

  @staticmethod
  def new_state(state, action):
    assert not 'You should override this method'

  def display(self):
    print ' ' * 3 + ' '.join([' %d ' % x for x in xrange(1, self.width + 1)]) 
    print ' ' * 2 + '-' * (len(self.state[0]) * 4 + 1)
    for idx, line in enumerate(self.state):
      print idx + 1,
      for item in line:
        print '|', self.encoding[item],
      print '|'
    print '  ' + '-' * (len(self.state[0]) * 4 + 1)
    print ''