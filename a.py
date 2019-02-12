import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


mean_head = nn.Linear(3, 2)
logvar_head = nn.Linear(3, 2)

x = torch.randn(1, 3)

mean = mean_head(x)
logvar = logvar_head(x)
logvar = torch.mm(logvar.transpose(0, 1), logvar)
print logvar
std = torch.exp(0.5*logvar)

d = MultivariateNormal(mean, std)
a = d.rsample()
logprob = d.log_prob(a)

L = -logprob

mean_head.zero_grad()
logvar_head.zero_grad()

L.backward()

print a
print d
print logprob
