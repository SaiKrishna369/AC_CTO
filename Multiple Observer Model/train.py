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

	def __init__(self, state_dim, action_dim, gridDimensions):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.gridDimensions = gridDimensions

		self.actor = model.Actor(self.state_dim, self.action_dim, gridDimensions)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)

		self.critic = model.Critic(self.state_dim)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)

		self.actor.train()
		self.critic.train()

		self.normal_distribution = None
		self.action = None

	# def getMean(self, val):
	# 	mean = torch.zeros(2)
	# 	mean[0] = (self.gridWidth*1.0)/(1.0 + math.e**(-val[0]))
	# 	mean[1] = (self.gridHeight*1.0)/(1.0 + math.e**(-val[1]))
	# 	return mean

	def getVariance(self):
		var1 = 1.0/(1 + torch.exp( torch.clamp(-self.actor.variance1, min=-10, max=10) ) ) + 1e-3
		var2 = 1.0/(1 + torch.exp( torch.clamp(-self.actor.variance2, min=-10, max=10) ) ) + 1e-3
		
		variance = torch.zeros(2, 2)
		variance[0][0] = var1**2
		variance[1][1] = var2**2
		return variance

	def getAction(self, state):
		state = Variable(torch.from_numpy(state))
		self.temp = self.actor.forward(state)

		self.normal_distribution = MultivariateNormal(loc=self.temp, covariance_matrix=self.getVariance())
		self.action = self.normal_distribution.sample()

		while (self.action[0] < 0 or self.action[0] > self.gridDimensions[0]) or \
				(self.action[1] < 0 or self.action[1] > self.gridDimensions[1]):
			self.action = self.normal_distribution.sample()

		return self.action
		

	def update(self, s1, r1, s2):
		s1 = Variable(torch.from_numpy(s1))
		#a1 = Variable(torch.from_numpy(a1))
		r1 = Variable(torch.from_numpy(r1))
		s2 = Variable(torch.from_numpy(s2))

		# ---------------------- optimize critic ----------------------
		loss = nn.MSELoss()
		td_target = r1 + GAMMA * self.critic.forward(s2)
		value_estimate = self.critic.forward(s1)
		critic_loss = loss(td_target, value_estimate)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# ---------------------- optimize actor ----------------------
		td_error = td_target.detach().data - value_estimate.detach().data
		actor_loss = -td_error * self.normal_distribution.log_prob(self.action)
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		# print "\n\n"
		# print "Actor loss: ", actor_loss
		# print "Mean grads: ", self.actor.hidden1.weight.grad, self.actor.output.weight.grad
		# print "Vaiance grads: ", self.actor.variance1.grad, self.actor.variance2.grad
		# print "\nCritic loss: ", critic_loss
		# print "Critic grads: ", self.critic.hidden1.weight.grad, self.critic.output.weight.grad
		# print "\n\n"		
		# print "action: ", self.action
		# print self.temp, self.temp.grad, "\n\n"
		self.actor_optimizer.step()


	def save_models(self, agent, episode_count):
		torch.save(self.actor.state_dict(), './Models/' + str(agent) + '/' + str(episode_count) + '_' + str(agent) + '_actor.pt')
		torch.save(self.critic.state_dict(), './Models/' + str(agent) + '/' + str(episode_count) + '_' + str(agent) + '_critic.pt')
		print 'Successfully saved models for agent', agent

	def load_models(self, agent, episode):
		self.actor.load_state_dict(torch.load('./Models/' + str(agent) + '/' + str(episode) + '_' + str(agent) + '_actor.pt'))
		self.critic.load_state_dict(torch.load('./Models/' + str(agent) + '/' + str(episode) + '_' + str(agent) + '_critic.pt'))
		print 'Succesfully loaded models of agent', agent