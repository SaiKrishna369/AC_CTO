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

		self.fc1 = nn.Linear(state_dim,1, bias=False)
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

	def forward(self, state):
		x = self.fc1(state)

		return x


class Actor(nn.Module):

	def __init__(self, state_dim, action_dim):
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fc1 = nn.Linear(state_dim, action_dim, bias=False)
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

		self.variance1 = nn.Parameter(torch.tensor(np.random.random()))
		self.variance2 = nn.Parameter(torch.tensor(np.random.random()))

	def forward(self, state):
		x = self.fc1(state)

		return x