import train


trainer = train.Trainer(20, 2, [150.0, 150.0])

# prev = 0
# trainer.load_models(prev)
# prev_var1 = trainer.actor.variance1.data
# prev_var2 = trainer.actor.variance2.data
# prev_critic = trainer.critic.fc1.weight.data
# prev_actor = trainer.actor.fc1.weight.data

for curr in range(99500, 100000,100):
	trainer.load_models(curr)
	
	curr_var1 = trainer.actor.variance1.data
	curr_var2 = trainer.actor.variance2.data
	curr_critic = trainer.critic.fc1.weight.data
	curr_actor = trainer.actor.fc1.weight.data

	print "\n\nEpisode: ", curr
	print "Variances: ", curr_var1, curr_var2
	print "Actor wts: ", curr_actor
	print "Critic wts: ", curr_critic