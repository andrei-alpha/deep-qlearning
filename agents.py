import numpy as np
from random import choice


class Agent(object):

  def __init__(self, name="agent"):
    self.name = name

  def notify(self, sim, status):
    pass

  def learn(self, *args):
    pass


class MinMax(Agent):

  def __init__(self, name="MinMax", max_level=4):
    self.max_level = max_level
    self.discount_factor = 0.95
    Agent.__init__(self, name)

  def choose_action(self, sim, level=0, alpha=-999999, beta=999999):
    player = (2 if level % 2 else 1)
    maximizing_player = (True if player == 1 else False)
    actions = sim.get_actions()
    if level == self.max_level or not actions:
      score = sim.collect_reward()
      # If the current player can't move -> he lost
      if maximizing_player:
        return (-score, level)
      else:
        return (score, level)

    scores, comb_scores = [], []
    np.random.shuffle(actions)
    for action in actions:
      sim.perform_action((action, player))
      val, lvl = self.choose_action(sim, level + 1, alpha, beta)
      score = val * (self.discount_factor ** lvl)
      comb_scores.append((val, lvl))
      scores.append(score)
      sim.reverse_action((action, player))
      if maximizing_player:
        alpha = max(alpha, score)
      else:
        beta = min(beta, score)
      if beta <= alpha:
        break
    best = (max(scores) if maximizing_player else min(scores))
    count = scores.count(best)
    idx = scores.index(best)
    if level == 0:
      return actions[idx]
    return comb_scores[idx]


class RandomPlayer(Agent):

  def __init__(self, name="RandomPlayer"):
    Agent.__init__(self, name)

  def choose_action(self, sim):
    return choice(sim.get_actions())


class HumanPlayer(Agent):

  def __init__(self, name):
    Agent.__init__(self, name)

  def choose_action(self, sim):
    return sim.read_action(self.name)

  def notify(self, sim, status):
    if status == "win":
      print "You have won!"
    elif status == "lose":
      print "AI has won!"
    else:
      print "Draw"
