import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import MultivariateNormal

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

                self.actor.train()
                self.critic.train()

		self.normal_distribution = None
                self.action = None

	def getMean(self, val):
                res = torch.zeros(2)
                res[0] = (self.gridWidth*1.0)/(1.0 + math.e**-val[0])
                res[1] = (self.gridHeight*1.0)/(1.0 + math.e**-val[1])
		return res#Variable(torch.tensor([ (self.gridWidth*1.0)/(1.0 + math.e**-val[0]), (self.gridHeight*1.0)/(1.0 + math.e**-val[1]) ]), requires_grad=True)

	def getVariance(self):
		var1 = 1.0/(1 + math.e**(-self.actor.variance1)) + 1e-3
		var2 = 1.0/(1 + math.e**(-self.actor.variance2)) + 1e-3
                #print var1
                #exit(-1)
                res = torch.zeros(2, 2)
                res[0][0] = var1**2
                res[1][1] = var2**2
		return res#Variable(torch.tensor([[var1**2, 0.0],[0.0, var2**2]]), requires_grad=True)

	def getAction(self, state):
		state = Variable(torch.from_numpy(state))
		temp = self.actor.forward(state)
                
                #print self.getMean(temp)
                #print self.getVariance()
                #exit(0)
		self.normal_distribution = MultivariateNormal(loc=self.getMean(temp), covariance_matrix=self.getVariance())
                print "self.normal_dis: ", self.normal_distribution
		action = self.normal_distribution.rsample()
		#action = action.data.numpy()

		while (action[0] < 0 or action[0] > self.gridWidth) or \
				(action[1] < 0 or action[1] > self.gridHeight):
			action = self.normal_distribution.rsample()
			#action = action.data.numpy()
                self.action = action
                #print ">", self.action
                #exit(0)
		return action
		

	def update(self, s1, a1, r1, s2):
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
		critic_loss.backward(retain_graph=True)
                #print self.critic.fc1.weight.grad
		self.critic_optimizer.step()

		# ---------------------- optimize actor ----------------------
		td_error = td_target - value_estimate
		actor_loss = -1 * self.normal_distribution.log_prob(self.action) * td_error
                from torchviz import make_dot
                dot = make_dot(actor_loss)
		dot.format = 'svg'
		dot.render()
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
                #print self.normal_distribution.log_prob(self.action)
                #print td_error
                print self.actor.fc1.weight.grad
                print "self.action:", self.action
                #print actor_loss
                exit(0)
		self.actor_optimizer.step()


	def save_models(self, episode_count):
		torch.save(self.actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
		torch.save(self.critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
		print 'Models saved successfully'

	def load_models(self, episode):
		self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
		self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
		print 'Models loaded succesfully'
