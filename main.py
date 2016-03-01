import itertools
import os
import time

import numpy as np

from agents import (
    HumanPlayer,
    MinMax,
    RandomPlayer,
)
from brain import Brain
from connect4 import Connect4
from breakout import Breakout
from dummy_game import DummyGame
from deepq import QLearn
from tictactoe import TicTacToe

new_models = 20
win_threshold_percentage = 50
win_threshold_games = 300
train_games = 3000
stats_frequency = 5
minmax_level = 4
models_root = "data"
run_mode = "train"  # Can be either train or eval
eval_mode = "human" # Can be either human, minmax or random
eval_model = "data/model_95940496"
train_mode = "single" # Can be either mixt, minmax, random, single

class Scorer(object):
  """Simple class to keep track of score, game duration and avg moves."""

  def __init__(self, frequency):
    self.results = []
    self.stats = []
    self.frequency = frequency
    self.games = 0

  def count(self, val, n):
    return self.results[len(self.results) - n:].count(val)

  def average(self, sum_type, n):
    if sum_type == "score":
      return sum([x for x in self.results[len(self.results) - n:]]) / n
    index = (0 if sum_type == "time" else 1)
    return sum([x[index] for x in self.stats[len(self.stats) - n:]]) / n

  def record_result(self, players, result, start_time, no_moves):
    game_duration = int((time.time() - start_time) * 1000)
    self.results.append(result)
    self.stats.append((game_duration, no_moves))
    self.games += 1
    if self.games % stats_frequency == 0:
      p1_wins = self.count(1, self.frequency) * 100 / self.frequency
      p2_wins = self.count(2, self.frequency) * 100 / self.frequency
      avg_games_per_min = (60 * 1000.0) / self.average("time", self.frequency)
      avg_moves = self.average("moves", self.frequency)
      print("Games: %d %s wins: %d%s %s wins: %d%s games/min: %d "
            "avg_moves: %d") % (self.games, players[0].name, p1_wins, "%",
                                players[1].name, p2_wins, "%",
                                avg_games_per_min, avg_moves)

  def record_result_single(self, player, score, start_time, no_moves):
    game_duration = int((time.time() - start_time) * 1000)
    self.results.append(score)
    self.stats.append((game_duration, no_moves))
    self.games += 1
    if self.games % stats_frequency == 0:
      avg_score = self.average("score", self.frequency)
      avg_games_per_min = (60 * 1000.0) / self.average("time", self.frequency)
      avg_moves = self.average("moves", self.frequency)
      print("Games: %d %s avg_score: %d games/min: %d avg_moves: %d"
        ) % (self.games, player.name, avg_score, avg_games_per_min, avg_moves)

  def get_statistics(self, last_games):
    p1_wins = self.count(1, last_games) * 100 / last_games
    p2_wins = self.count(2, last_games) * 100 / last_games
    return p1_wins, p2_wins


def play_game_with_opponent(players, sim, scorer, display_board=False):
  """Clasic turn based game loop."""
  start_time = time.time()
  idx, prev_state, prev_action = 0, None, None
  for moves in itertools.count(start=1):
    # Get current state from game emulator
    state = sim.get_state()
    action = players[idx].choose_action(sim)
    turn = sim.turn
    reward = sim.perform_action((action, turn))
    if display_board:
      sim.display()

    if not sim.get_actions():  # Game has ended
      players[idx].learn(state, action, reward)
      players[idx ^ 1].learn(prev_state, prev_action, -reward)
      if reward:  # Current player won the game
        players[idx].notify(sim, "win")
        players[idx ^ 1].notify(sim, "lose")
        scorer.record_result(players, turn, start_time, moves)
      else:  # This is a draw
        players[idx].notify(sim, "draw")
        players[idx ^ 1].notify(sim, "draw")
        scorer.record_result(players, 0, start_time, moves)
      break
    elif prev_action and prev_state:
      new_actions_mask = sim.get_actions_mask()
      players[idx ^ 1].learn(prev_state, prev_action, 0, sim.get_state(),
                             new_actions_mask)
    prev_state, prev_action = state, action
    idx ^= 1
  sim.reset()

def play_single_player_game(player, sim, scorer, display_board=False):
  """Single player game loop."""
  start_time = time.time()
  for moves in itertools.count(start=1):
    # Get current state from game emulator
    state = sim.get_state()
    action = player.choose_action(sim)
    reward = sim.perform_action(action)
    if display_board:
      sim.display()

    player.learn(state, action, reward)

    if not sim.get_actions(): # Game has ended
      score = sim.get_score()
      scorer.record_result_single(player, score, start_time, moves)
      break
  sim.reset()

def test():
  sim = Connect4()
  brain = Brain(sim.width, sim.height, sim.actions_count, load_path=eval_model)
  opponent = QLearn(sim, brain, "AI", exploration_period=0,
                    discount_factor=0.9)

  display = False
  scorer = Scorer(stats_frequency)
  if eval_mode == "human":
    player = HumanPlayer("Player")
    display = True
  elif eval_mode == "random":
    player = RandomPlayer()
  elif eval_mode == "minmax":
    player = MinMax(max_level=minmax_level)
  else:
    raise ValueError, ("Invalid eval_mode. Got %s expected "
      "(human|random|minmax)") % eval_mode

  while True:
    play_game_with_opponent([opponent, player], sim, scorer, display_board=display)

def train_with_opponent():
  sim = Connect4()
  if train_mode == "mixt":
    opponents = [RandomPlayer("Rand")]
    for idx, model_path in enumerate(os.listdir(models_root)):
      full_path = os.path.join(models_root, model_path)
      prev_brain = Brain(sim.width,
                         sim.height,
                         sim.actions_count,
                         load_path=full_path)
      opponents.append(QLearn(sim, prev_brain, "v" + str(idx)))
  elif train_mode == "random":
    opponents = [RandomPlayer("Rand")]
  elif train_mode == "minmax":
    opponents = [MinMax(max_level=minmax_level)]
  else:
    raise ValueError, ("Invalid train_mode. Got %s expected "
      "(mixt|random|minmax)") % train_mode

  scorer = Scorer(stats_frequency)
  for step in xrange(new_models):
    brain = Brain(sim.width, sim.height, sim.actions_count)
    player = QLearn(sim, brain, "AI")

    w = [0.1 * i for i in xrange(1, len(opponents) + 1)]
    p = [wi / sum(w) for wi in w]

    for games in xrange(1, 100000):
      if games % win_threshold_games == 0:
        # If the new model wins more than 90% of the games of the last 300
        win_statistics = scorer.get_statistics(win_threshold_games)[0]
        if win_statistics > win_threshold_percentage:
          # Save the model to disk and load it as an inference only model
          model_path = brain.save(models_root)
          prev_brain = Brain(sim.width,
                             sim.height,
                             sim.actions_count,
                             load_path=model_path)
          opponents.append(QLearn(sim, prev_brain, "V" + str(step),
                                  exploration_period=0, discount_factor=0.9))
          print "-" * 70
          print("New model wins %d%s against previous models after "
                "%d games") % (win_statistics, "%", games)
          print "-" * 70
          print ''
          break

      opponent = np.random.choice(opponents, 1, p)[0]
      players = [player, opponent]
      play_game_with_opponent(players, sim, scorer)

  human_player = HumanPlayer("Player")
  while True:
    play_game_with_opponent([opponents[-1], human_player], sim, scorer)

def train_single_player():
  sim = DummyGame() # Breakout()
  brain = Brain(sim.width, sim.height, sim.actions_count)
  player = QLearn(sim, brain, "AI")
  scorer = Scorer(stats_frequency)

  for games in xrange(train_games):
    play_single_player_game(player, sim, scorer, display_board=True)
  # Save the model to disk and load it as an inference only model
  model_path = brain.save(models_root)
  print 'Saved trained model to', model_path

if __name__ == "__main__":
  if run_mode == "train":
    if train_mode == "single":
      train_single_player()
    else:
      train_with_opponent()
  else:
    test()
