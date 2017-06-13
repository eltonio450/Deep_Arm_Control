import opensim
import math
import numpy as np
import sys
import os
from osim.env import OsimEnv
import os.path
from GaitEnv import GaitEnv

class BioInspiredHierarchicalLearning(GaitEnv):
    ninput = 31
    stageMax = []
    stageMin = []
    stageCurrent = []
    isStageOver = []
    stageNumber = 0;

    def __init__(self, visualize = False, musclesName = '', stepsize = 0.01):
        self.initialStage(2)
        self.model_path = 'gait9dof18musc.osim'
        self.musclesLength = [ 0 for i in range(len(musclesName))]
        folder = 'data'
        self.musclesName = musclesName
        #self.W = 0.1+0*np.eye(len(self.musclesLength),len(self.musclesLength)*2)
        self.X0 = np.zeros(len(self.musclesLength)*2)
        

        if os.path.isfile(folder + '/weights_antoine') :
            print("File exists")
            self.weightsFile = open(folder + '/weights_antoine', 'r')
            self.zeroLengthFile = open(folder + '/l0_antoine', 'r')
            self.X0 = np.loadtxt(self.zeroLengthFile)
            self.X0 = np.concatenate([self.X0,self.musclesLength])
            self.W = np.loadtxt(self.weightsFile)
        else:
            print("File does not exists")
            self.weightsFile = open(folder + '/weights_antoine', 'w')
            self.W = np.randomself.X0 = np.concatenate([self.X0,self.musclesLength]).rand(len(self.musclesLength),len(self.musclesLength)*2)-0.5; #- 5*np.eye(len(self.musclesLength),len(self.musclesLength)*2)
            self.W = self.W.round(3)
            #sparse matrix
            #self.W = np.power(self.W,5).round(2)

            np.savetxt(self.weightsFile,self.W)
        super(BioInspiredHierarchicalLearning, self).__init__(visualize = visualize, noutput = 36)


    def X_tw(self):
        musclesLength_new = [self.osim_model.model.getActuators().get(m).getStateVariableValues(self.osim_model.state).get(1) for m in self.musclesName]
        musclesdLength = [ (x-y)/self.stepsize for x,y in zip( musclesLength_new, self.musclesLength) ]
        actuatorsValues = [self.osim_model.model.getActuators().get(m).getStateVariableValues(self.osim_model.state).get(0) for m in self.musclesName]

        sensorsValues = musclesLength_new + musclesdLength 

        if len(musclesdLength) == 0:
            sensorsValues = sensorsValues + [ 0 for i in range(len(self.musclesName))]

        self.musclesLength = musclesLength_new
        return sensorsValues

    def _step(self, action):
        # Action contains the current output of the RL network.
        # We want to plug it into our twitching controller


        # Y = W(1+A)*(X-X0)

        # X : sensory input, [36,1]
        # W : twitching weight matrix, [length(Y),2*(length(Y))] = [18,36]
        # Y : muscle activity, [18,1]

        # Size of the action space = 18*36 = 2*length(Y)^2 = 18*36 = 648 output dimension ^^

        # OR 

        # Y = G*W*(X-X0) 
        self.last_action = action
        # multiplication
        
        X = np.array(self.X_tw())
        Y = self.W.dot(-X+self.X0+action)
        #self.activate_musclesTwitch(Y)
        self.activate_muscles(Y)

        self.osim_model.manager.setInitialTime(self.stepsize * self.istep)
        self.osim_model.manager.setFinalTime(self.stepsize * (self.istep + 1))


        try:
            self.osim_model.manager.integrate(self.osim_model.state)
        except Exception:
            print ("Exception raised")
            return self.get_observation(), -500, True, {}

        self.istep = self.istep + 1

        # self.randomRestart = True;
        # self.minimumStep = 10;


        # if(self.randomRestart):
        #     if(self.istep > self.minimumStep):
        #         self.randomRestart = False;
        #         if(random.randint(1,100) < 30):
        #             self.stopAt = self.istep = random.randint(20,100)

        # if(self.stopAt == self.istep):
        #     self.randomRestart = True;
        #     res = [ self.get_observation(), self.compute_reward(), True, {} ]
        # else:
        #     res = [ self.get_observation(), self.compute_reward(), self.is_done(), {} ]

        res = [ self.get_observation(), self.compute_reward(), self.is_done(), {} ]


        #if(self.istep == 10):
        #    self.X0 = np.zeros(len(self.musclesLength)*2)

        #print(action)

        return res


    def activate_musclesTwitch(self, action):
        for j in range(9):
            muscle = self.osim_model.muscleSet.get(j)
            muscle.setActivation(self.osim_model.state, action[j])
            muscle = self.osim_model.muscleSet.get(j + 9)
            muscle.setActivation(self.osim_model.state, action[j])

    def activate_muscles(self, action):
        for j in range(18):
            muscle = self.osim_model.muscleSet.get(j)
            muscle.setActivation(self.osim_model.state, float(action[j]))



    def initialStage(self,stageNumber):
        self.stageNumber = stageNumber;
        for i in range(stageNumber):
            self.stageMax.append(-300)
            self.stageMin.append(300)
            self.stageCurrent.append(0)
            self.isStageOver.append(False)

    def updateStage(self,stage, value, cond, mi = -300, ma = 300):
        if(value >= self.stageMax[stage]):
            self.stageMax[stage] = value
        if(value < self.stageMin[stage]):
            self.stageMin[stage] = value

        if(value >= ma):
            value = ma;
        if(value < mi):
            value = mi;


        if(self.stageMax[stage] != self.stageMin[stage]):
            self.stageCurrent[stage] = (value-self.stageMin[stage])/(self.stageMax[stage]-self.stageMin[stage]);
        else:  
            self.stageCurrent[stage] = 0;

        
        self.isStageOver[stage] = cond;
        

    def getCurrentStageReward(self):
        isLastOver = True
        currentStage = 0
        currentReward = 0;
        
        for i in range(self.stageNumber):
            if self.isStageOver[i] and isLastOver:
                currentStage = currentStage +1;
                isLastOver = True
            if not self.isStageOver[i]:
                isLastOver = False
        

        for i in range(currentStage):
            currentReward = currentReward + i + 1

        #print " |{}|".format(currentStage)
        
        currentReward += self.stageCurrent[currentStage-1]
        return currentReward


    def NormAcceleration(self):
        acc = self.osim_model.model.calcMassCenterAcceleration(self.osim_model.state)
        return abs(acc[0])**2 + abs(acc[1])**2 + abs(acc[2])**2

    def NormSpeed(self):
        vel = self.osim_model.model.calcMassCenterVelocity(self.osim_model.state)
        return abs(vel[0])**2 + abs(vel[1])**2 + abs(vel[2])**2

    def Dist(self):
        pos = self.osim_model.model.calcMassCenterPosition(self.osim_model.state)
        return pos[0]+0.1;

    def Speed(self):
        vel = self.osim_model.model.calcMassCenterVelocity(self.osim_model.state)
        return vel[0];

    def PelvisTilt(self):
        return -self.osim_model.joints[0].getCoordinate(0).getValue(self.osim_model.state)

    def GrfLeft(self):
        return self.currentL[0] + self.currentL[1]

    def GrfRight(self):
        return self.currentR[0] + self.currentR[1]

    def CoM(self):
        return self.osim_model.model.calcMassCenterPosition(self.osim_model.state)

    def compute_reward(self):
        #TODO : Add a samll force that helps staying upright
        #TODO : Reduce the help based on current reward. Higher the reward lower the help
        #TODO : Help should also change the reward so that, the same behavior with less help will have higher reward.
        #TODO : start simulation from previous simulation results.
        #return -sumForceUnderFeet*symmetryForceUnderFeet*(100*(-ankle_angle+1) + 10*(1-math.fabs(pos[1] - ymin))+10*(1-math.fabs(pos[0]) - xmin))

        #self.NormAcceleration()
        #self.NormSpeed()
        #self.Dist()
        #self.Speed()
        #self.PelvisTilt()
        #self.GrfLeft()
        #self.GrfRight()
        #self.CoM

        rewardDistance = self.Dist() if self.Dist() > 0 else 0;
        rewardVelocity = 1 if self.Speed() >= 0 else 0
        #stage1 = rewardGRF
        #stage1 = rewardVelocity    
        #stage0 = rewardDistance*rewardVelocity
        
        self.updateStage(0,self.CoM()[0],self.CoM()[0] > -0.05, 0)
        self.updateStage(1,rewardDistance*rewardVelocity,False)
        #print " {}=||{}||{}||".format( round(self.getCurrentStageReward(),2),round(self.stageCurrent[0],2),round(self.stageCurrent[1],2))

        return self.getCurrentStageReward()