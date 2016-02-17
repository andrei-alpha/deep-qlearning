import itertools
import random

from tictactoe import TicTacToe
from connect4 import Connect4
from deepq import QLearn
from brain import Brain
from agents import MinMax, RandomPlayer, HumanPlayer

train_games = 2000
stats_frequency = 100

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
      players[idx].learn( sim.new_state(state, (action, turn)), reward)
      players[idx ^ 1].learn( sim.new_state(prev_state, (prev_action, sim.turn)), -reward)
      if reward: # Current player won the game
        players[idx].notify(sim, "win")
        players[idx ^ 1].notify(sim, "lose")
        scorer.record_result(players, turn)
      else: # This is a draw
        players[idx].notify(sim, "draw")
        players[idx ^ 1].notify(sim, "draw")
        scorer.record_result(players, 0)
      break
    else:
      players[idx ^ 1].learn( sim.new_state(state, (action, turn)), 0)
    prev_state, prev_action = state, action
    idx = idx ^ 1
  sim.reset()

if __name__ == "__main__":
  sim = Connect4()
  brain = Brain(sim.width, sim.height)
  players = QLearn(brain, "Vasile"), RandomPlayer("Gigel")

  scorer = Scorer(stats_frequency)
  for step in xrange(train_games):
    play_game(players, sim, scorer)
  human_player = HumanPlayer("Player")
  while True:
    play_game([players[0], human_player], sim, scorer, display=True)
