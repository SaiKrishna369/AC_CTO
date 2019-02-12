import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

eps = 0.025

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):

	def __init__(self, state_dim):
		super(Critic, self).__init__()

		self.state_dim = state_dim

		#First fully connected layer
		self.fc1 = nn.Linear(state_dim,64)
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

		#Second fully connected layer
		self.fc2 = nn.Linear(64,128)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		#Final output layer
		self.fc3 = nn.Linear(128,1)
		self.fc3.weight.data.uniform_(-eps,eps)

	def forward(self, state):
		s1 = F.relu(self.fc1(state))
		s2 = F.relu(self.fc2(s1))
		x = self.fc3(s2)

		return x


class Actor(nn.Module):

	def __init__(self, state_dim, action_dim):
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fc1 = nn.Linear(state_dim,64)
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

		self.fc2 = nn.Linear(64,128)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(128,64)
		self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

		self.fc4 = nn.Linear(64,action_dim)
		self.fc4.weight.data.uniform_(-eps,eps)

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)

		return x