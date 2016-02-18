import random
import numpy as np

from agents import Agent

class QLearn(Agent):
  def __init__(self, brain, name="QLearn",
      train_every_nth=5, train_batch_size=30,
      max_experience=100000, exploration_period=10000,
      epsilon_final=0.01, discount_factor=0.9):
    Agent.__init__(self, name)
    self.brain = brain
    self.train_every_nth = train_every_nth
    self.train_batch_size = train_batch_size
    self.epsilon_final = epsilon_final
    self.discount_factor = discount_factor
    self.max_experience = max_experience
    self.exploration_period = exploration_period
    self.actions_executed = 0
    self.memory = []

  def getQ(self, states):
    """Evaluate the value learned for the currents states."""
    return self.brain.eval(states).tolist()

  def learnQ(self, state, reward):
    """Get previous score for this state and update it."""
    self.memory.append((state, reward))
    if len(self.memory) > self.max_experience:
      self.memory.pop(0)

  def learn(self, state, reward, new_states=None):
    """Learn that reaching the current state yields this reward. The game can
    be stocastics and this should be modeled well by the neural network."""
    if new_states:
      maxqnew = max(self.getQ(new_states))
      reward += self.discount_factor * maxqnew
    self.learnQ(state, reward)

  def linear_annealing(self, p_initial, p_final, n, total):
    """Perform linear annealing to compute the probability to pick and random
    action, depending on the progress of the exploration period."""
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
      new_states = [sim.new_state(state, (a, sim.turn)) for a in actions]
      q = self.getQ(new_states)
      maxQ = max(q)
      if q.count(maxQ) > 1:
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
