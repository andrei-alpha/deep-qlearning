import os
import itertools
import random
import numpy as np

from tictactoe import TicTacToe
from connect4 import Connect4
from deepq import QLearn
from brain import Brain
from agents import MinMax, RandomPlayer, HumanPlayer

new_models = 20
train_games = 3000
stats_frequency = 100
models_root = "data"

class Scorer():
  def __init__(self, frequency):
    self.results = []
    self.frequency = frequency
    self.games = 0

  def count(self, val, n):
    return len(filter(lambda x: x == val, self.results[len(self.results)-n:]))

  def record_result(self, players, result):
    self.results.append(result)
    self.games += 1
    if self.games % stats_frequency == 0:
      p1_wins = self.count(1, self.frequency) * 100 / self.frequency
      p2_wins = self.count(2, self.frequency) * 100 / self.frequency
      print 'Games: %d %s wins: %d%s %s wins: %d%s' % (self.games,
        players[0].name, p1_wins, '%', players[1].name, p2_wins, '%')

  def get_statistics(self, last_games):
    p1_wins = self.count(1, last_games) * 100 / last_games
    p2_wins = self.count(2, last_games) * 100 / last_games
    return p1_wins, p2_wins

def play_game(players, sim, scorer, display=False):
  """Clasic turn based game loop."""
  idx = 0
  prev_state, prev_action = None, None
  while True:
    # Get available actions and current state from emulator
    state, actions = sim.get_state(), sim.get_actions()
    action = players[idx].choose_action(sim)
    turn = sim.turn
    reward = sim.perform_action((action, turn))

    if not sim.get_actions(): # Game has ended
      players[idx].learn(state, action, reward)
      players[idx ^ 1].learn(prev_state, prev_action, -reward)
      if reward: # Current player won the game
        players[idx].notify(sim, "win")
        players[idx ^ 1].notify(sim, "lose")
        scorer.record_result(players, turn)
      else: # This is a draw
        players[idx].notify(sim, "draw")
        players[idx ^ 1].notify(sim, "draw")
        scorer.record_result(players, 0)
      break
    elif prev_action and prev_state:
      new_actions_mask = sim.get_actions_mask()
      players[idx ^ 1].learn(prev_state, prev_action, 0, sim.get_state(), new_actions_mask)
    prev_state, prev_action = state, action
    idx = idx ^ 1
  sim.reset()

if __name__ == "__main__":
  sim = Connect4()  
  opponents = [RandomPlayer("Rand")]
  for idx, model_path in enumerate(os.listdir(models_root)):
    full_path = os.path.join(models_root, model_path)
    prev_brain = Brain(sim.width, sim.height, sim.actions_count, load_path=full_path)
    opponents.append(QLearn(sim, prev_brain, "v" + str(idx)))

  scorer = Scorer(stats_frequency)
  for step in xrange(new_models):
    brain = Brain(sim.width, sim.height, sim.actions_count)
    player = QLearn(sim, brain, "AI")

    p = [0.1 * x for x in xrange(1, len(opponents) + 1)]
    p = [x / sum(p) for x in p]

    for games in xrange(1, 100000):
      if games % 300 == 0:
        # If the new model wins more than 90% of the games of the last 300
        win_statistics = scorer.get_statistics(300)[0]
        if win_statistics > 95:
          # Save the model to disk and load it as an inference only model
          model_path = brain.save(models_root)
          prev_brain = Brain(sim.width, sim.height, sim.actions_count, load_path=model_path)
          opponents.append(QLearn(sim, prev_brain, "vnew" + str(step)))
          print 'The new model wins %d%s against previous models' % (win_statistics, '%')
          break

      opponent = np.random.choice(opponents, 1, p)[0]
      players = [player, opponent]
      play_game(players, sim, scorer)
  
  human_player = HumanPlayer("Player")
  while True:
    play_game([opponents[-1], human_player], sim, scorer, display=True)
