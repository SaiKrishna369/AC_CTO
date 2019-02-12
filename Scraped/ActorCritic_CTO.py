import gym, gym_cto
import numpy as np
import torch
from torch.autograd import Variable
import os
import gc

#custom modules
import train
import buffer

env = gym.make('CTO-v0')

MAX_EPISODES = 5000
MAX_STEPS = 1000
MAX_BUFFER = 100000

NUM_TARGETS = 10
SENSOR_RANGE = 15
GRID_WIDTH = 150.0
GRID_HEIGHT = 150.0

S_DIM = 2*NUM_TARGETS
A_DIM = 2

replay_memory = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, replay_memory, SENSOR_RANGE)
trainer.load_models(1000)

for _ep in range(1001,MAX_EPISODES):
	#Reset environment
	env.initialize(targets=NUM_TARGETS, sensorRange=SENSOR_RANGE, gridWidth=GRID_WIDTH, gridHeight=GRID_HEIGHT)
	observation = env.reset()

	print 'EPISODE :- ' +  str(_ep) + "\r"
	while True:
		env.render()

		state = np.float32(observation).flatten()
		position = env.getAgentPosition()
		action = trainer.getAction(state, position)

		new_observation, reward, done, info = env.step(action)

		if done:
			new_state = None
		else:
			new_state = np.float32(new_observation).flatten()
			replay_memory.add(state, action, reward, new_state)

		observation = new_observation

		trainer.optimize()
		if done:
			break

	if _ep%100 == 0:
		trainer.save_models(_ep)


print 'Completed episodes\n'