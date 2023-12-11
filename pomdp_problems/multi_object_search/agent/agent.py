# Defines the agent. There's nothing special
# about the MOS agent in fact, except that
# it uses models defined in ..models, and
# makes use of the belief initialization
# functions in belief.py
import random

import pomdp_py
from .belief import *
from ..models.transition_model import *
from ..models.observation_model import *
from ..models.reward_model import *
from ..models.policy_model import *


class MosAgent(pomdp_py.Agent):
    """One agent is one robot."""

    def __init__(self,
                 robot_id,
                 init_robot_state,
                 # initial robot state (assuming robot state is observable perfectly)
                 object_ids,  # target object ids
                 dim,  # tuple (w,l) of the width (w) and length (l) of the gridworld search space.
                 sensor,  # Sensor equipped on the robot
                 sigma=0.01,  # parameter for observation model
                 epsilon=1,  # parameter for observation model
                 belief_rep="histogram",
                 # belief representation, either "histogram" or "particles".
                 prior={},  # prior belief, as defined in belief.py:initialize_belief
                 num_particles=100,  # used if the belief representation is particles
                 grid_map=None):  # GridMap used to avoid collision with obstacles (None if not provided)
        self.robot_id = robot_id
        self._object_ids = object_ids
        self.sensor = sensor

        # since the robot observes its own pose perfectly, it will have 100% prior
        # on this pose.
        prior[robot_id] = {init_robot_state.pose: 1.0}
        rth = init_robot_state.pose[2]

        # initialize belief
        init_belief = initialize_belief(dim,
                                        self.robot_id,
                                        self._object_ids,
                                        prior=prior,
                                        representation=belief_rep,
                                        robot_orientations={self.robot_id: rth},
                                        num_particles=num_particles)
        transition_model = MosTransitionModel(dim,
                                              {self.robot_id: self.sensor},
                                              self._object_ids)
        observation_model = MosObservationModel(dim,
                                                self.sensor,
                                                self._object_ids,
                                                sigma=sigma,
                                                epsilon=epsilon)
        reward_model = GoalRewardModel(self._object_ids, robot_id=self.robot_id)
        policy_model = PolicyModel(self.robot_id, grid_map=grid_map)
        super().__init__(init_belief, policy_model,
                         transition_model=transition_model,
                         observation_model=observation_model,
                         reward_model=reward_model)

    def clear_history(self):
        """Custum function; clear history"""
        self._history = None


class trashList():
    """This is needed to make sure the belief is sampling the right
    type of State for this problem."""

    def __init__(self):
        """
        robot_id (int): The id of the robot that has this belief.
        object_beliefs (objid -> GenerativeDistribution)
            (includes robot)
        """
        self.trashtrue = {}
        self.trashbel = {}
        self.trashbelKey = {}
        self.probbel = {}
        self.probtrue = {}
        self.time_since_last_visit = {}

    def addTrash(self, trash):
        self.trashbel = trash
        for key in self.trashbel:
            self.trashtrue[key] = random.randrange(1, 10)
        self.randProb()

    def getLocation(self):
        return self.trashbelKey[self.maxTrash()]

    def addTrashKey(self, trashKey):
        self.trashbelKey = trashKey

    def maxTrash(self):
        maxNode = 0
        maxTrash = 0
        for itrash in self.trashbel:
            if self.trashbel[itrash] > maxTrash:
                maxTrash = self.trashbel[itrash]
                maxNode = itrash
        return maxNode

    def clean(self):
        nodeToClean = self.maxTrash()
        print("Litterbug believes ", nodeToClean, " has ", self.trashbel[nodeToClean], "trash vs cleaned ", self.trashtrue[nodeToClean], "trash")
        print("Litterbug believes ", nodeToClean, " produces ", self.probbel[nodeToClean], "trash vs true", self.probtrue[nodeToClean], "trash produced")
        oldprobbel = self.probbel[nodeToClean]
        self.probbel[nodeToClean] = (float(self.probbel[nodeToClean]) +
                                     float(self.trashtrue[nodeToClean]) /
                                     float(self.time_since_last_visit[nodeToClean])) / 2
        print("Litterbug updates probability belief from ", oldprobbel, " to ", self.probbel[nodeToClean])
        self.makeTrash()
        self.time_since_last_visit[nodeToClean] = 0
        self.trashbel[nodeToClean] = 0
        self.trashtrue[nodeToClean] = 0


    def makeTrash(self):
        for key in self.trashbel:
            bonus = random.randrange(1,3) - 2
            self.trashbel[key] = self.trashbel[key] + self.probbel[key]
            self.trashtrue[key] = self.trashtrue[key] + self.probtrue[key] + bonus
            self.time_since_last_visit[key] = self.time_since_last_visit[key] + 1

    def randProb(self):
        for key in self.trashbel:
            print("Node: ", key)
            self.probbel[key] = random.randrange(10, 20)
            self.probtrue[key] = random.randrange(2, 30)
            print("Litterbug initial probbel: ", self.probbel[key], " vs probtrue ", self.probtrue[key])
            self.time_since_last_visit[key] = 0
        print("Trash made per cycle = probtrue + [-1,0,1]")
        self.makeTrash()


    def updateTrash(self, itrash, newvalue):
        self.trashbel[itrash] = newvalue

    def printTrash(self):
        print("Litterbug Beliefs versus Truth")
        for key in self.trashbel:
            print("Node ", key)
            print("Believed ", self.trashbel[key], " trash vs actual trash ", self.trashtrue[key])
            print("Believed ", self.probbel[key], " trash_Prob vs true_Prob ", self.probtrue[key])
        print("")

class ParkAgent(pomdp_py.Agent):
    """One agent is one robot."""

    def __init__(self,
                 robot_id,
                 init_robot_state,
                 # initial robot state (assuming robot state is observable perfectly)
                 object_ids,  # target object ids
                 dim,  # tuple (w,l) of the width (w) and length (l) of the gridworld search space.
                 sensor,  # Sensor equipped on the robot
                 sigma=0.01,  # parameter for observation model
                 epsilon=1,  # parameter for observation model
                 belief_rep="Park",
                 # belief representation, either "histogram" or "particles".
                 prior={},  # prior belief, as defined in belief.py:initialize_belief
                 num_particles=100,  # used if the belief representation is particles
                 grid_map=None,
                 # GridMap used to avoid collision with obstacles (None if not provided)
                 trashList=trashList()):
        self.robot_id = robot_id
        self._object_ids = object_ids
        self.sensor = sensor
        self.trashList = trashList

        # since the robot observes its own pose perfectly, it will have 100% prior
        # on this pose.
        prior[robot_id] = {init_robot_state.pose: 1.0}
        rth = init_robot_state.pose[2]

        # initialize belief
        init_belief = initialize_belief(dim,
                                        self.robot_id,
                                        self._object_ids,
                                        prior=prior,
                                        representation=belief_rep,
                                        robot_orientations={self.robot_id: rth},
                                        num_particles=num_particles)
        #print(init_belief.object_beliefs)
        # temptrash = copy.deepcopy(init_belief.object_beliefs)
        temptrash = {}
        temptrashkey = {}
        numtrash = 0
        for key in init_belief.object_beliefs:
            if key != -114:
                temptrash[key] = numtrash
                temptrashkey[key] = init_belief.object_beliefs[key]
                numtrash = numtrash + 1
        self.trashList.addTrash(temptrash)
        self.trashList.addTrashKey(temptrashkey)
        #initialize small beliefs
        #MAKE ONLY ONE GOAL
        #ADJUST GOAL IN PROBLEM
        self._object_ids = {0}
        init_belief = initialize_belief(dim,
                                        self.robot_id,
                                        self._object_ids,
                                        prior=prior,
                                        representation=belief_rep,
                                        robot_orientations={self.robot_id: rth},
                                        num_particles=num_particles)
        transition_model = MosTransitionModel(dim,
                                              {self.robot_id: self.sensor},
                                              self._object_ids)
        #self.trashList.setTransitionModel(transition_model)
        observation_model = MosObservationModel(dim,
                                                self.sensor,
                                                self._object_ids,
                                                sigma=sigma,
                                                epsilon=epsilon)
        reward_model = GoalRewardModel(self._object_ids, robot_id=self.robot_id)
        policy_model = PolicyModel(self.robot_id, grid_map=grid_map)

        print(init_belief)
        super().__init__(init_belief, policy_model,
                         transition_model=transition_model,
                         observation_model=observation_model,
                         reward_model=reward_model)

    def clear_history(self):
        """Custum function; clear history"""
        self._history = None
