import gym, gym_cto
import numpy as np
import torch
from torch.autograd import Variable
import os

#custom modules
import train

env = gym.make('CTO-v0')

MAX_EPISODES = 50000

NUM_TARGETS = 10
SENSOR_RANGE = 15
GRID_WIDTH = 150.0
GRID_HEIGHT = 150.0

S_DIM = 2*NUM_TARGETS
A_DIM = 2

trainer = train.Trainer(S_DIM, A_DIM, GRID_WIDTH, GRID_HEIGHT)

for _ep in range(MAX_EPISODES):
	#Reset environment
	env.initialize(targets=NUM_TARGETS, sensorRange=SENSOR_RANGE, gridWidth=GRID_WIDTH, gridHeight=GRID_HEIGHT)
	
	observation = env.reset()
	state = np.float32(observation).flatten()

	print 'EPISODE :- ' +  str(_ep)
	while True:
		#env.render()
				
		action = trainer.getAction(state)
		new_state, reward, done, info = env.step(action)
		
		new_state = np.float32(new_state).flatten()
		rew = np.array(reward, dtype=np.float32)
		if reward != 0:
			trainer.update(state, action, rew, new_state)
		state = new_state

		if done:
			break

	if _ep%100 == 0:
		trainer.save_models(_ep)


print 'Completed episodes\n'