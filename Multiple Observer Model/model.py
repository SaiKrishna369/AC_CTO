import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):

	def __init__(self, state_dim):
		super(Critic, self).__init__()

		self.state_dim = state_dim

		self.hidden1 = nn.Linear(state_dim, 128, bias=False)
		self.hidden1.weight.data = fanin_init(self.hidden1.weight.data.size())

		self.output = nn.Linear(128, 1, bias=False)
		self.output.weight.data = fanin_init(self.output.weight.data.size())

	def forward(self, state):
		h1 = torch.sigmoid(self.hidden1(state))
		o = self.output(h1)

		return o


class Actor(nn.Module):

	def mean_activation(self, x):
		return (self.gridDimension)/(1.0 + torch.exp( torch.clamp(-x, min=-64, max=64) ))

	def __init__(self, state_dim, action_dim, gridDimension):
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.hidden1 = nn.Linear(state_dim, 128, bias=False)
		self.hidden1.weight.data = fanin_init(self.hidden1.weight.data.size())

		self.output = nn.Linear(128, action_dim, bias=False)
		self.output.weight.data = fanin_init(self.output.weight.data.size())

		self.variance1 = nn.Parameter(torch.tensor(np.random.random()))
		self.variance2 = nn.Parameter(torch.tensor(np.random.random()))

		self.gridDimension = torch.tensor(gridDimension)

	def forward(self, state):
		h1 = torch.sigmoid(self.hidden1(state))
		o = self.mean_activation(self.output(h1))

		return o