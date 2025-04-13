# import sys
# print("python version: ", sys.version) # this project uses python 3.11.4
import numpy as np
from matplotlib import pyplot as plt
# import py_trees Behavior Tree library
from PyInfiniteTree.Core.TaskStackMachine import TaskStackMachine
from PyInfiniteTree.Core.TaskStackMachine import Status
# import blackboard singleton
from blackboard.blackboard import blackboard
from blackboard.data_tree import DataTree
# import misc behaviors
from naive_navigation.follow_waypoints import FollowWaypoints
# webots API
from controller import Supervisor

# INSTANTIATE ROBOT AND BLACKBOARD
robot = Supervisor()
blackboard.setup(robot)

# SETUP ROOT LEVEL TREES
dataTree = TaskStackMachine(blackboard)
tree = TaskStackMachine(blackboard)
dataTree.addBehavior(DataTree(blackboard)) # to keep consistent time dependent meta-data such as wheel velocity 
tree.addBehavior(FollowWaypoints(blackboard))

# step through webots robot and tick behavior tree
while robot.step(blackboard.timestep) != -1:
    dataTree.drive() # tick data tree
    message = tree.drive() # step/tick behavior tree

    if message in [Status.SUCCESS, Status.FAILURE]:
        print("Tree completed!")
        break