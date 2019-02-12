import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np
import math

import model

BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.95
TAU = 0.001

class Trainer:

	def __init__(self, state_dim, action_dim, replay_buffer, range):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.replay_buffer = replay_buffer
		self.sensor_range = range

		self.actor = model.Actor(self.state_dim, self.action_dim)
		self.target_actor = model.Actor(self.state_dim, self.action_dim)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)

		self.critic = model.Critic(self.state_dim, self.action_dim)
		self.target_critic = model.Critic(self.state_dim, self.action_dim)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)

		hard_update(self.target_actor, self.actor)
		hard_update(self.target_critic, self.critic)

	def getAction(self, state, position):
		state = Variable(torch.from_numpy(state))
		temp = self.target_actor.forward(state).detach()
		temp = temp.data

		gaussian = MultivariateNormal(temp, torch.eye(2))
		action = gaussian.sample()
		action = action.data.numpy()

		return (action*self.sensor_range + position)

	def optimize(self):
		s1,a1,r1,s2 = self.replay_buffer.sample(BATCH_SIZE)

		s1 = Variable(torch.from_numpy(s1))
		a1 = Variable(torch.from_numpy(a1))
		r1 = Variable(torch.from_numpy(r1))
		s2 = Variable(torch.from_numpy(s2))

		# ---------------------- optimize critic ----------------------
		# Use target actor exploitation policy here for loss evaluation
		a2 = self.target_actor.forward(s2).detach()
		next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
		# y_exp = r + gamma*Q'( s2, pi'(s2))
		y_expected = r1 + GAMMA*next_val
		# y_pred = Q( s1, a1)
		y_predicted = torch.squeeze(self.critic.forward(s1, a1))
		# compute critic loss, and update the critic
		loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		self.critic_optimizer.zero_grad()
		loss_critic.backward()
		self.critic_optimizer.step()

		# ---------------------- optimize actor ----------------------
		pred_a1 = self.actor.forward(s1)
		loss_actor = -1*torch.sum(self.critic.forward(s1, pred_a1))
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()

		soft_update(self.target_actor, self.actor, TAU)
		soft_update(self.target_critic, self.critic, TAU)

	def save_models(self, episode_count):
		torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
		torch.save(self.target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
		print 'Models saved successfully'

	def load_models(self, episode):
		self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
		self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
		hard_update(self.target_actor, self.actor)
		hard_update(self.target_critic, self.critic)
		print 'Models loaded succesfully'


def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)


def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)