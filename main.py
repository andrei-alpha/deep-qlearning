import random

from tictactoe import TicTacToe
from connect4 import Connect4
from deepq import QLearn
from board_game_ai import MinMax

train_steps = 10000
human_player = False
stats_frequency = 10

class Scorer():
  def __init__(self):
    self.results = []
    self.games = 0

  def count(self, val, n):
    return len(filter(lambda x: x == val, self.results[len(self.results)-n:]))

  def record_result(self, score):
    self.results.append(score)
    self.games += 1
    if self.games % stats_frequency == 0:
      p1_wins = self.count(1, stats_frequency) * 100 / stats_frequency
      p2_wins = self.count(2, stats_frequency) * 100 / stats_frequency
      print 'Games: %d QLearn wins: %d%s Random play wins: %d%s' % (self.games, p1_wins, '%', p2_wins, '%')

def game_ended(ai, sim, player, reward, reward_draw=0):
  global human_player
  if not sim.get_actions():
    if reward:
      if player == 1: # AI won this game
        if human_player:
          print 'AI won!'
        ai.learn(state, action, reward)
      else:
        ai.learn(state, action, -reward)
      scorer.record_result(player)
      if human_player:
        print 'You have won!'
      if step > train_steps:
        human_player = True
    else:
      # This is a draw
      if human_player:
        print 'This was a draw!'
      ai.learn(state, action, reward_draw)
      scorer.record_result(0)
    sim.reset()
    return True
  return False

if __name__ == "__main__":
  sim = Connect4() # TicTacToe()
  ai = QLearn()
  minmax = MinMax()

  scorer = Scorer()
  actions, state, new_actions, new_state = None, None, None, None
  for step in xrange(train_steps * 2):
    # Get available actions and current state from emulator
    new_state, new_actions = sim.get_state(), sim.get_actions()

    actions = new_actions
    state = new_state
    action = ai.choose_action(state, actions)
    reward = sim.perform_action((action, 1))

    if human_player:
      sim.display()

    if game_ended(ai, sim, 1, reward, reward*0.5):
      continue
    else:
      # Just pick a random move
      if not human_player:
        # new_action = random.choice(sim.get_actions())
        new_action = minmax.choose_action(sim)
      else:
        new_action = sim.read_action('Player')

      reward = sim.perform_action((new_action, 2))
      if game_ended(ai, sim, 2, reward, reward*0.5):
        continue
      else:
        ai.learn(state, action, 0, sim.get_state(), sim.get_actions())
