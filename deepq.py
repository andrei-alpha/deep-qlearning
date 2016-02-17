import random
import numpy as np

from agents import Agent

class QLearn(Agent):
  def __init__(self, brain, name="QLearn",
      train_every_nth=5, train_batch_size=30,
      max_experience=20000, exploration_period=1000,
      epsilon_final=0.005, alpha=0.2, gamma=0.9):
    Agent.__init__(self, name)
    self.brain = brain
    self.train_every_nth = train_every_nth
    self.train_batch_size = train_batch_size
    self.epsilon_final = epsilon_final
    self.gamma = gamma
    self.alpha = alpha
    self.max_experience = max_experience
    self.exploration_period = exploration_period
    self.actions_executed = 0
    self.memory = []

  def getQ(self, state):
    """Evaluate the value learned for the current state."""
    return self.brain.eval(state)

  def learn(self, state, reward, *args):
    """Learn that reaching the current state yields this reward. The game can
    be stocastics and this should be modeled well by the neural network."""
    self.memory.append((state, reward))
    if len(self.memory) > self.max_experience:
      self.memory.pop_left()

  def linear_annealing(self, p_initial, p_final, n, total):
    if n > total:
      return p_final
    return p_initial - (n * (p_initial - p_final)) / total

  def choose_action(self, sim):
    state, actions = sim.get_state(), sim.get_actions()
    action = None
    self.actions_executed += 1
    self.training_step()
    
    epsilon_current = self.linear_annealing(1.0, self.epsilon_final,
        self.actions_executed, self.exploration_period)
    if random.random() < epsilon_current:
      action = random.choice(actions)
    else:
      q = [self.getQ(sim.new_state(state, (a, sim.turn))) for a in actions]
      maxQ = max(q)
      count = q.count(maxQ)
      if count > 1:
        best = [i for i in range(len(actions)) if q[i] == maxQ]
        i = random.choice(best)
      else:
        i = q.index(maxQ)
      action = actions[i]
    return action

  def training_step(self):
    """Pick a self.train_batch of experiences from memory and backpropagate
    the observed reward."""
    if self.actions_executed % self.train_every_nth:
      return
    if len(self.memory) < self.train_batch_size:
      return
    choices = np.random.randint(len(self.memory), size=self.train_batch_size)
    train_data = [self.memory[x] for x in choices]
    self.brain.train(zip(*train_data))
