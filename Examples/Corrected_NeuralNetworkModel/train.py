import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np
import math

import model

LEARNING_RATE = 0.001
GAMMA = 0.95

class Trainer:

	def __init__(self, state_dim, action_dim, gridWidth, gridHeight):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.gridWidth = gridWidth
		self.gridHeight = gridHeight

		self.actor = model.Actor(self.state_dim, self.action_dim)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)

		self.critic = model.Critic(self.state_dim)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)

		self.normal_distribution = None

	def getMean(self, val):
		return torch.tensor([ (self.gridWidth*1.0)/(1.0 + math.e**-val[0]), (self.gridHeight*1.0)/(1.0 + math.e**-val[1]) ])

	def getAction(self, state):
		state = Variable(torch.from_numpy(state))
		temp = self.actor.forward(state).detach()

		self.normal_distribution = MultivariateNormal(self.getMean(temp), torch.eye(2))
		action = self.normal_distribution.sample()
		action = action.data.numpy()

		while (action[0] < 0 or action[0] > self.gridWidth) or \
				(action[1] < 0 or action[1] > self.gridHeight):
			action = self.normal_distribution.sample()
			action = action.data.numpy()

		return action
		

	def update(self, s1, a1, r1, s2):
		s1 = Variable(torch.from_numpy(s1))
		a1 = Variable(torch.from_numpy(a1))
		r1 = Variable(torch.from_numpy(r1))
		s2 = Variable(torch.from_numpy(s2))

		# ---------------------- optimize critic ----------------------
		loss = nn.MSELoss()
		td_target = r1 + GAMMA * self.critic.forward(s2)
		value_estimate = self.critic.forward(s1)
		critic_loss = loss(td_target, value_estimate)
		self.critic_optimizer.zero_grad()
		critic_loss.backward(retain_graph=True)
		self.critic_optimizer.step()

		# ---------------------- optimize actor ----------------------
		td_error = td_target - value_estimate
		actor_loss = -1 * self.normal_distribution.log_prob(a1) * td_error
		self.actor_optimizer.zero_grad()
		actor_loss.backward(retain_graph=True)
		self.actor_optimizer.step()


	def save_models(self, episode_count):
		torch.save(self.actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
		torch.save(self.critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
		print 'Models saved successfully'

	def load_models(self, episode):
		self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
		self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
		print 'Models loaded succesfully'