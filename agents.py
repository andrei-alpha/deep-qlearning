from random import choice
from copy import deepcopy

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
    Agent.__init__(self, name)

  def choose_action(self, sim, level=0):
    player = (2 if level % 2 else 1)
    actions = sim.get_actions()
    if level == self.max_level or not actions:
      score = sim.collect_reward()
      # If the current player can't move -> he lost
      return (-score if player == 1 else score)

    scores = []
    for action in actions:
      sim.perform_action((action, player))
      scores.append(self.choose_action(sim, level + 1))
      sim.reverse_action((action, player))
    best = (max(scores) if player == 1 else min(scores))
    count = scores.count(best)
    if count > 1:
      best_idxs = [i for i in range(len(actions)) if scores[i] == best]
      idx = choice(best_idxs)
    else:
      idx = scores.index(best)
    if level == 0:
      return actions[idx]
    return scores[idx]

class RandomPlayer(Agent):
  def __init__(self, name="RandomPlayer"):
    Agent.__init__(self, name)
  
  def choose_action(self, sim):
    return choice(sim.get_actions())

class HumanPlayer(Agent):
  def __init__(self, name):
    Agent.__init__(self, name)

  def choose_action(self, sim):
    sim.display()
    return sim.read_action(self.name)

  def notify(self, sim, status):
    sim.display()
    if status == 'win':
      print 'You have won!'
    elif status == 'lose':
      print 'AI has won!'
    else:
      print 'Draw'