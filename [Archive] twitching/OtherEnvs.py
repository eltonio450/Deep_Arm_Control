import opensim
import math
import numpy as np
import sys
import os
from osim.env import OsimEnv
import os.path
from GaitEnv import GaitEnv

class StandEnv(GaitEnv):
    def compute_reward(self):
        y = self.osim_model.joints[0].getCoordinate(2).getValue(self.osim_model.state)
        x = self.osim_model.joints[0].getCoordinate(1).getValue(self.osim_model.state)

        pos = self.osim_model.model.calcMassCenterPosition(self.osim_model.state)
        vel = self.osim_model.model.calcMassCenterVelocity(self.osim_model.state)
        acc = self.osim_model.model.calcMassCenterAcceleration(self.osim_model.state)

        a = abs(acc[0])**2 + abs(acc[1])**2 + abs(acc[2])**2
        v = abs(vel[0])**2 + abs(vel[1])**2 + abs(vel[2])**2
        #rew = 50.0 - min(a,10.0) - min(v,40.0) + pos[1]
        #rew = 50.0 - min(a,10.0) - min(v,40.0) + pos[1]

        #return rew / 50.0
        return pos[1]
    
class HopEnv(GaitEnv):
    def __init__(self, visualize = True):
        self.model_path = 'hop8dof9musc.osim'
        super(HopEnv, self).__init__(visualize = visualize, noutput = 9)

    def compute_reward(self):
        y = self.osim_model.joints[0].getCoordinate(2).getValue(self.osim_model.state)
        return (y) ** 3

    def is_head_too_low(self):
        y = self.osim_model.joints[0].getCoordinate(2).getValue(self.osim_model.state)
        return (y < 0.4)

    def activate_muscles(self, action):
        for j in range(9):
            muscle = self.osim_model.muscleSet.get(j)
            muscle.setActivation(self.osim_model.state, action[j])
            muscle = self.osim_model.muscleSet.get(j + 9)
            muscle.setActivation(self.osim_model.state, action[j])

class CrouchEnv(HopEnv):
    def compute_reward(self):
        y = self.osim_model.joints[0].getCoordinate(2).getValue(self.osim_model.state)
        return 1.0 - (y-0.5) ** 3

    def is_head_too_low(self):
        y = self.osim_model.joints[0].getCoordinate(2).getValue(self.osim_model.state)
        return (y < 0.25)