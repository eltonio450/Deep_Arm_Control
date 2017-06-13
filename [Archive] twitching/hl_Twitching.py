#%% Imports
# Derived from keras-rl
import opensim as osim
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam

import numpy as np

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import sys
sys.path = ["../"] + sys.path
from osim.env import *

from keras.optimizers import RMSprop

import argparse
import math


#%%
# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=10000)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
args = parser.parse_args()

# Load walking environment
# TODO ? minutes : Increase simulation duration time to M*T = 9*500
muscle_names = [
				"hamstrings_l",
				"bifemsh_l",
				"glut_max_l",
				"iliopsoas_l",
				"rect_fem_l",
				"vasti_l",
				"gastroc_l",
				"soleus_l",
				"tib_ant_l",
				"hamstrings_r",
				"bifemsh_r",
				"glut_max_r",
				"iliopsoas_r",
				"rect_fem_r",
				"vasti_r",
				"gastroc_r",
				"soleus_r",
				"tib_ant_r"
				]
env = BioInspiredHierarchicalLearning(args.visualize,muscle_names)


#%% End of env. genration
nb_actions = env.action_space.shape[0]

# Total number of steps in training
nallsteps = int(args.steps)

# Create networks for DDPG
# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(18))
actor.add(Activation('relu'))
actor.add(Dense(4))
actor.add(Activation('relu'))
actor.add(Dense(18))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = merge([action_input, flattened_observation], mode='concat')
x = Dense(18)(x)
x = Activation('relu')(x)
x = Dense(12)(x)
x = Activation('relu')(x)
x = Dense(18)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(input=[action_input, observation_input], output=x)
print(critic.summary())

# Set up the agent for training
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.noutput)
agent = DDPGAgent(nb_actions=nb_actions, 
				  #start_step_policy=5, // 
				  actor=actor, critic=critic, critic_action_input=action_input,
				  memory=memory, nb_steps_warmup_critic=50, nb_steps_warmup_actor=50,
				  random_process=random_process, 
				  gamma=.89, target_model_update=1e-3,
				  delta_clip=1.)
# agent = ContinuousDQNAgent(nb_actions=env.noutput, V_model=V_model, L_model=L_model, mu_model=mu_model,
#							memory=memory, nb_steps_warmup=1000, random_process=random_process,
#							gamma=.99, target_model_update=0.1)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])


# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

from shutil import copyfile
from os import remove

if args.train:
	try:
		copyfile(args.model+'_actor',args.model+'__actor')
		copyfile(args.model+'_critic',args.model+'__critic')
  		agent.load_weights(args.model+'_')
  		remove(args.model+'__critic')
  		remove(args.model+'__actor')
	except: 
  		pass
	agent.fit(env, 
		# nb_max_start_steps=20, # Samples from the action space a random number of steps. Maybe this is replaced by nb_steps_warmup
		action_repetition=1, 
		nb_steps=nallsteps, 
		visualize=True, 
		verbose=1, 
		#nb_max_episode_steps=env.timestep_limit, 
		nb_max_episode_steps=500, 
		log_interval=100)
	# After training is done, we save the final weights.
	agent.save_weights(args.model, overwrite=True)

if not args.train:
	agent.load_weights(args.model)
	# Finally, evaluate our algorithm for 1 episode.
	agent.test(env, nb_episodes=1, 
		action_repetition=1,
		nb_max_start_steps=20, # Samples from the action space a random number of steps.
		visualize=False, nb_max_episode_steps=10000)


