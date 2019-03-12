import gym, gym_cto_pytorch
import numpy as np
import torch
from torch.autograd import Variable
import os, time, sys

#custom modules
import train

env = gym.make('TCTO-v0')

MAX_EPISODES = 10#75000

NUM_TARGETS = 1
SENSOR_RANGE = 15
GRID_DIMENSION = [150.0, 150.0]

S_DIM = 2*(NUM_TARGETS + 1)
A_DIM = 2

trainer = train.Trainer(S_DIM, A_DIM, GRID_DIMENSION)

start_time = time.time()

for _ep in range(MAX_EPISODES):
	#Reset environment
	env.initialize(targets=NUM_TARGETS, sensorRange=SENSOR_RANGE, gridWidth=GRID_DIMENSION[0], gridHeight=GRID_DIMENSION[1])
	
	observation = env.reset()
	state = observation.view(-1) #flatten

	print 'EPISODE :- ', _ep

	done = False
	while not done:
		#env.render()
				
		action = trainer.getAction(state).clone()
		new_state, reward, done, info = env.step(action)
		
		new_state = new_state.view(-1) #flatten
		rew = np.array(reward, dtype=np.float32)

		trainer.update(rew, new_state)
		state = new_state

	if _ep%100 == 0:
		trainer.save_models(_ep)


print 'Completed episodes\n'
end_time = time.time()
print 'Total time elapsed: ', (end_time - start_time)/60
