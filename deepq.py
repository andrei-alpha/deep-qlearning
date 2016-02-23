import random
import numpy as np

from agents import Agent

class QLearn(Agent):
  def __init__(self, sim, brain, name="QLearn",
      train_every_nth=5, train_batch_size=30,
      max_experience=100000, exploration_period=10000,
      epsilon_final=0.01, discount_factor=0.5):
    Agent.__init__(self, name)
    self.sim = sim
    self.brain = brain
    self.train_every_nth = train_every_nth
    self.train_batch_size = train_batch_size
    self.epsilon_final = epsilon_final
    self.discount_factor = discount_factor
    self.max_experience = max_experience
    self.exploration_period = exploration_period
    self.actions_executed = 0
    self.memory = []

  def getQ(self, states, actions_mask):
    """Evaluate the value learned for the currents states."""
    return self.brain.eval(states, actions_mask).tolist()

  def learnQ(self, state, action, reward):
    """Get previous score for this state and update it."""
    self.memory.append((state, action, reward))
    if len(self.memory) > self.max_experience:
      self.memory.pop(0)

  def learn(self, state, action, reward, new_state=None, new_actions_mask=None):
    """Learn that reaching the current state yields this reward. The game can
    be stocastics and this should be modeled well by the neural network."""
    if new_state:
      maxqnew = max(self.getQ(new_state, new_actions_mask))
      reward += self.discount_factor * maxqnew
    self.learnQ(state, action, reward)

  def linear_annealing(self, p_initial, p_final, n, total):
    """Perform linear annealing to compute the probability to pick and random
    action, depending on the progress of the exploration period."""
    if n > total:
      return p_final
    return p_initial - (n * (p_initial - p_final)) / total

  def choose_action(self, sim):
    state, actions, actions_mask = sim.get_state(), sim.get_actions(), sim.get_actions_mask()
    action = None
    self.actions_executed += 1
    self.training_step()
    
    epsilon_current = self.linear_annealing(1.0, self.epsilon_final,
        self.actions_executed, self.exploration_period)
    if random.random() < epsilon_current:
      action = random.choice(actions)
    else:
      q = self.getQ(state, actions_mask)
      maxQ = max(q)
      if len(q) == 0:
        print 'Empty predictions ....'
        return random.choice(actions)
      if q.count(maxQ) > 1:
        best = [i for i in xrange(len(actions_mask)) if q[i] == maxQ]
        i = random.choice(best)
      else:
        i = q.index(maxQ)
      action = sim.get_action_from_index(i)
      if not action in actions:
        print 'Picked wrong action!'
        return random.choice(actions)
    return action

  def training_step(self):
    """Pick a self.train_batch of experiences from memory and backpropagate
    the observed reward."""
    if self.actions_executed % self.train_every_nth:
      return
    if len(self.memory) < self.train_batch_size:
      return
    choices = np.random.randint(len(self.memory), size=self.train_batch_size)
    train_data = []
    for x in choices:
      state, action, reward = self.memory[x]
      actions_mask = self.sim.get_actions_mask_from_action(action)
      train_data.append((state, actions_mask, reward))
    self.brain.train(zip(*train_data))
