import opensim
import math
import numpy as np
import os
import random
from osim.env import OsimEnv

class ArmEnv(OsimEnv):
    ninput = 14
    model_path = "./arm2dof6musc.osim"

    def __init__(self, visualize = False):
        self.iepisode = 0
        self.shoulder = 0.0
        self.elbow = 0.0
        super(ArmEnv, self).__init__(visualize = visualize)

    def configure(self):
        super(ArmEnv, self).configure()
        self.osim_model.joints.append(opensim.CustomJoint.safeDownCast(self.osim_model.jointSet.get(0)))
        self.osim_model.joints.append(opensim.CustomJoint.safeDownCast(self.osim_model.jointSet.get(1)))

    def new_target(self):
        #self.shoulder = -0.02275097#random.uniform(-1.2,0.3)
        #self.elbow = 0.2195816#random.uniform(-1.0,0)
        self.shoulder = 0.0#random.uniform(-1.2,0.3)
        self.elbow = 0.0#random.uniform(-1.0,0)
        #print(self.osim_model.joints[0].getCoordinate(0).getValue(self.osim_model.state))
        #print(self.osim_model.joints[0].getCoordinate(0).getValue(self.osim_model.state))
        #print(dir(self.osim_model.joints[0].set_coordinates))
        #print(self.osim_model.joints[0].set_coordinates.__doc__)
        #self.osim_model.joints[0].set_coordinates(0)
        #print(dir(self.osim_model.joints[0]))
        #self.osim_model.joints[1].set_coordinates(0,0.0)
        #action = [0,0,0,0,0,0]
        #muscle = self.osim_model.muscleSet.get(0)
        #muscle.setActivation(self.osim_model.state, float(action[0]))
        #muscleSet = self.osim_model.model.getMuscles()
        #muscle = muscleSet.get(0)
        #muscle.setActivation(self.osim_model.state, float(action[0]))
    

    def new_new(self):
        print(self.osim_model.joints[0].getCoordinate(0).getValue(self.osim_model.state))
        #self.osim_model.joints[0].getCoordinate(0).setValue(self.osim_model.state, -0.02275097)
        #self.osim_model.joints[1].getCoordinate(0).setValue(self.osim_model.state, 0.2195816)
        self.osim_model.joints[0].getCoordinate(0).setValue(self.osim_model.state, 0.0)
        self.osim_model.joints[1].getCoordinate(0).setValue(self.osim_model.state, 0.0)
        #muscle.setActivation(self.osim_model.state, float(action[0]))
        print(self.osim_model.joints[0].getCoordinate(0).getValue(self.osim_model.state))

    def reset(self):
        self.new_target()
        return super(ArmEnv, self).reset()

    def compute_reward(self):
        obs = self.get_observation()
        #pos = (self.angular_dist(obs[2],self.shoulder) + self.angular_dist(obs[3],self.elbow))
        #speed = 0 #(obs[4]**2 + obs[5]**2) / 200.0
        """
        if -0.1 < obs[2] and obs[2] < 0.1 and -0.1 < obs[3] and obs[3] < 0.1:
            return 1
        else:
            return 0
        #return - pos - speed
        #return 1.0
        """
        #costs = (obs[2]+0.3)**2 + (obs[3]-0.3)**2 + 0.1*(obs[4])**2 + (obs[5])**2
        costs = (obs[2]-0.4*np.cos(0.001*self.istep))**2 + (obs[3]-0.4*np.sin(0.001*self.istep))**2
        return -costs

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

