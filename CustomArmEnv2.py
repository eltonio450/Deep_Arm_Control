import opensim
import math
import numpy as np
import os
import random
from osim.env import OsimEnv

# In that version of arm environment, the task is a point to point reaching executed twice,
# once from the initial position to the target, once comming back to the initial.
class CustomArmEnv2(OsimEnv):
    ninput = 14
    model_path = os.path.join(os.path.dirname(__file__), '../models/customArmEnv.osim')

    def __init__(self, visualize = False):
        self.iepisode = 0
        self.shoulder = 0.0
        self.elbow = 0.0
        # task is divided in two period, reaching the target (counter = 0) and comming back to initial position (counter = 1)
        self.task_counter = 0
        super(CustomArmEnv2, self).__init__(visualize = visualize)

    def configure(self):
        super(CustomArmEnv2, self).configure()
        self.osim_model.joints.append(opensim.CustomJoint.safeDownCast(self.osim_model.jointSet.get(0)))
        self.osim_model.joints.append(opensim.CustomJoint.safeDownCast(self.osim_model.jointSet.get(1)))

    def new_target(self):
        #saving initial positiion
        self.shoulder0 = self.shoulder
        self.elbow0 = self.elbow
        #new target position
        self.shoulder = random.uniform(-1.2,0.3)
        self.elbow = random.uniform(-1.0,0)
        

    def reset(self):
        self.new_target()
        self.task_counter = 0
        return super(CustomArmEnv2, self).reset()

    def is_done(self):
        if self.compute_reward < self.precision:
            self.task_counter += 1
            #the new target becomes the initial position
            self.shoulder = self.shoulder0
            self.elbow = self.elbow0
            
        if self.task_counter > 1:
            return True
        else:
            return False
    
    def compute_reward(self):
        obs = self.get_observation()
        pos = (self.angular_dist(obs[2],self.shoulder) + self.angular_dist(obs[3],self.elbow))
        speed = 0 #(obs[4]**2 + obs[5]**2) / 200.0
        return - pos - speed

    def get_observation(self):
        invars = np.array([0] * self.ninput, dtype='f')

        invars[0] = self.shoulder
        invars[1] = self.elbow
        
        invars[2] = self.osim_model.joints[0].getCoordinate(0).getValue(self.osim_model.state)
        invars[3] = self.osim_model.joints[1].getCoordinate(0).getValue(self.osim_model.state)

        invars[4] = self.osim_model.joints[0].getCoordinate(0).getSpeedValue(self.osim_model.state)
        invars[5] = self.osim_model.joints[1].getCoordinate(0).getSpeedValue(self.osim_model.state)

        invars[6] = self.sanitify(self.osim_model.joints[0].getCoordinate(0).getAccelerationValue(self.osim_model.state))
        invars[7] = self.sanitify(self.osim_model.joints[1].getCoordinate(0).getAccelerationValue(self.osim_model.state))

        pos = self.osim_model.model.calcMassCenterPosition(self.osim_model.state)
        vel = self.osim_model.model.calcMassCenterVelocity(self.osim_model.state)
        
        invars[8] = pos[0]
        invars[9] = pos[1]
        invars[10] = pos[2]

        invars[11] = vel[0]
        invars[12] = vel[1]
        invars[13] = vel[2]

        return invars


