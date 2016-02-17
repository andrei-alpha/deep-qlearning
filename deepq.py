import random

from agents import Agent

class QLearn(Agent):
  def __init__(self, name="QLearn", epsilon=0.005, alpha=0.2, gamma=0.9):
    Agent.__init__(self, name)
    self.epsilon = epsilon
    self.gamma = gamma
    self.alpha = alpha
    self.training_steps = 1000
    self.discount_factor = 0.6
    self.memory = {}

  def getQ(self, state, action):
    return self.memory.get((state, action), 0)

  def learnQ(self, state, action, reward, value):
    oldv = self.memory.get((state, action), None)
    if oldv is None:
      self.memory[(state, action)] = reward
    else:
      self.memory[(state, action)] = oldv + self.alpha * (value - oldv)

  def learn(self, state1, action1, reward, new_state=None, new_actions=None):
    if new_actions:
      maxqnew = max([self.getQ(new_state, a) for a in new_actions])
    else:
      maxqnew = reward
    self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

  def choose_action(self, sim):
    state, actions = sim.get_state(), sim.get_actions()
    action = None
    if random.random() < self.epsilon:
      action = random.choice(actions)
    else:
      q = [self.getQ(state, a) for a in actions]
      maxQ = max(q)
      count = q.count(maxQ)
      if count > 1:
        best = [i for i in range(len(actions)) if q[i] == maxQ]
        i = random.choice(best)
      else:
        i = q.index(maxQ)
      action = actions[i]
    return action
