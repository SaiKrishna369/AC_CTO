import gym, gym_cto
import numpy as np
import torch
from torch.autograd import Variable
import os, time

#custom modules
import train

env = gym.make('CTO-v1')

MAX_EPISODES = 5#75001

NUM_TARGETS = 10
NUM_AGENTS = 10
SENSOR_RANGE = 15
GRID_DIMENSION = [150.0, 150.0]

S_DIM = 3*(NUM_TARGETS + NUM_AGENTS)
A_DIM = 2

def getState(observation, scaleDown=False):
	state = []
	for i in xrange(NUM_AGENTS):
			temp = observation[i]
			if scaleDown:
				for j in xrange(NUM_TARGETS + NUM_AGENTS):
					temp[j][0] /= GRID_DIMENSION[0]
					temp[j][1] /= GRID_DIMENSION[1]

			temp = np.float32(temp).flatten()
			state.append(temp)

	return state

if __name__ == "__main__":
	start_time = time.time()

	trainer = []
	for _ in xrange(NUM_AGENTS):
		trainer.append(train.Trainer(S_DIM, A_DIM, GRID_DIMENSION))

	for _ep in range(MAX_EPISODES):
		#Reset environment
		env.initialize(agents=NUM_AGENTS, targets=NUM_TARGETS,\
						sensorRange=SENSOR_RANGE, gridWidth=GRID_DIMENSION[0],\
						gridHeight=GRID_DIMENSION[1], mark=True)
		
		observation = env.reset()

		state = getState(observation)

		done = False
		print 'EPISODE :- ', _ep

		while not done:
			env.render()

			global_action = []

			for i in xrange(NUM_AGENTS):
				local_state = state[i]
				action = trainer[i].getAction(local_state).clone()
				global_action.append(action.data.numpy())

			# print "\n\n"
			# print _ep, global_action
			observation, reward, done, info = env.step(global_action)
			rew = np.array([ [i] for i in reward ], dtype=np.float32)
			
			new_state = getState(observation)

			for i in xrange(NUM_AGENTS):
				trainer[i].update(state[i], rew[i], new_state[i])

		if _ep%100 == 0:
			for i in xrange(NUM_AGENTS):			
				trainer[i].save_models(i, _ep)


	print 'Completed episodes\n'
	end_time = time.time()
	print 'Total time elapsed: ', (end_time - start_time)/60