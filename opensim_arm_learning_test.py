#%% Importation
from osim.env import ArmEnv
from CustomArmEnv import CustomArmEnv
import os


#%% Specific Import
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

from osim.env import *

from keras.optimizers import RMSprop

import argparse
import math

#%% Initialisation
env = CustomArmEnv(visualize=False)

#%% First Observation
observation = env.reset()


#%% Learning

for i in range(5000):
    observation, reward, done, info = env.step(env.action_space.sample())

#%%

print("Test cell")


#%%
print("next cell")
print("second line next cell")