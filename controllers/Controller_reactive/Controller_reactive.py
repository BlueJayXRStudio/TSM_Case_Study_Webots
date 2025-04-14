# import sys
# print("python version: ", sys.version) # this project uses python 3.11.4
import numpy as np
from matplotlib import pyplot as plt
# import py_trees Behavior Tree library
import py_trees
# import blackboard singleton
from blackboard.blackboard import blackboard
from blackboard.data_tree import DataTree
# import robust waypoints follower
from naive_navigation.follow_waypoints import FollowWaypoints
# webots API
from controller import Supervisor

# INSTANTIATE ROBOT AND BLACKBOARD
robot = Supervisor()
blackboard.setup(robot)

# SETUP ROOT LEVEL TREES
dataTree = DataTree("Data tree") # to keep consistent time dependent meta-data such as wheel velocity 
tree = FollowWaypoints("")

# Invoke setup on all nodes before stepping through
dataTree.setup_with_descendants()
tree.setup_with_descendants()

# step through webots robot and tick behavior tree
while robot.step(blackboard.timestep) != -1:
    # tick data tree
    dataTree.tick_once()
    # step/tick behavior tree
    tree.tick_once()

    if tree.status in [py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE]:
        print("Tree completed!")
        break