import random

class QLearn():
  def __init__(self, epsilon=0.01, alpha=0.2, gamma=0.9):
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

  def choose_action(self, state, actions, display=False):
    action = None
    if random.random() < self.epsilon:
      if display:
        print 'Pick randomly :)'
      action = random.choice(actions)
    else:
      if display:
        for a in actions:
          print a, self.getQ(state, a)

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
