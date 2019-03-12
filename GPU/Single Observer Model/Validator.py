import gym, gym_cto
import numpy as np
import torch
from torch.autograd import Variable
import os, time

#custom modules
import train

env = gym.make('CTO-v0')

NUM_TARGETS = 10
SENSOR_RANGE = 107
GRID_DIMENSION = [150.0, 150.0]

S_DIM = 2*NUM_TARGETS
A_DIM = 2

SIM = 50

start_time = time.time()

trainer = train.Trainer(S_DIM, A_DIM, GRID_DIMENSION)
f=open("Output.txt", "a+")

for episodes in xrange(0, 75001, 100):
	trainer.load_models(episodes)

	average_score = 0.0
	print 'EPISODE :- ', episodes
	for _ep in range(SIM):
		#Reset environment
		env.initialize(targets=NUM_TARGETS, sensorRange=SENSOR_RANGE, gridWidth=GRID_DIMENSION[0], gridHeight=GRID_DIMENSION[1])
		
		observation = env.reset()
		state = np.float32(observation/GRID_DIMENSION).flatten()

		#print 'EPISODE :- ', _ep
		total_reward = 0.0
		while True:
			#env.render()

			action = trainer.getAction(state).clone()
			new_state, reward, done, info = env.step(action.data.numpy())
			total_reward += reward
			
			new_state = np.float32(new_state/GRID_DIMENSION).flatten()
			rew = np.array(reward, dtype=np.float32)
			#if reward != 0 and state.sum() != 0:
				#trainer.update(state, rew, new_state)
			state = new_state

			if done:
				average_score += total_reward
				#print total_reward
				break

		if _ep%100 == 0:
			pass#trainer.save_models(_ep)
	average_score /= SIM
	f.write("Episode: %d\n" % (episodes))
	f.write("Score: %d\n\n" % (average_score))
	#print "Episode : ", episodes, "Score: ", average_score

f.close()
print 'Completed\n'
end_time = time.time()
print 'Total time elapsed: ', (end_time - start_time)/60