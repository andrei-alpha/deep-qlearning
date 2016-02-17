import random

from agents import Agent

class QLearn(Agent):
  def __init__(self, name="QLearn", epsilon_final=0.005, alpha=0.2, gamma=0.9):
    Agent.__init__(self, name)
    self.epsilon_final = epsilon_final
    self.gamma = gamma
    self.alpha = alpha
    self.discount_factor = 0.6
    self.exploration_period = 1000
    self.actions_executed = 0
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

  def linear_annealing(self, p_initial, p_final, n, total):
    if n > total:
      return p_final
    return p_initial - (n * (p_initial - p_final)) / total

  def choose_action(self, sim):
    state, actions = sim.get_state(), sim.get_actions()
    action = None
    self.actions_executed += 1
    
    epsilon_current = self.linear_annealing(1.0, self.epsilon_final,
        self.actions_executed, self.exploration_period)
    if random.random() < epsilon_current:
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
