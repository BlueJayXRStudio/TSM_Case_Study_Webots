# import sys
# print("python version: ", sys.version) # this project uses python 3.11.4
import numpy as np
from matplotlib import pyplot as plt
# import py_trees Behavior Tree library
import py_trees

# import blackboard singleton
from blackboard.blackboard import blackboard
from blackboard.data_tree import DataTree

# import misc behaviors
from naive_navigation.follow_waypoints import FollowWaypoints
from naive_navigation.test_behavior import TestBehavior

# webots API
from controller import Supervisor

# INSTANTIATE ROBOT AND BLACKBOARD
robot = Supervisor()
blackboard.setup(robot)

# SETUP ROOT LEVEL TREES
dataTree = DataTree("meta tree") # to keep consistent time dependent meta-data such as wheel velocity 
# tree = RotateClockwise("rotate clockwise", [], 2) # Main behavior tree
# tree = RotateCounterclockwise("rotate counter-clockwise", [], 2) # Main behavior tree
# tree = MoveBackwards("move backwards", [], 2, 0.05) # Main behavior tree
# tree = dRotate("", [], 90, 2)
# tree = dMove("", [], 0.0254 * 2, 4)
# tree = MoveToRL("", [], (1.09817, 0.3374))
tree = FollowWaypoints("")
# tree = TestBehavior("test behavior")

# tree = Selector("Main", children=[
#     RotateClockwise("rotate clockwise", [], 2), # Main behavior tree
#     RotateCounterclockwise("rotate counter-clockwise", [], 2), # Main behavior tree
#     MoveBackwards("move backwards", [], 2, 0.05) # Main behavior tree
# ], memory = True)


# MAIN BEHAVIOR TREE
# tree = Sequence("Main", children=[
#     # set arm to a safe position
#     ResetArm("reset arm to safe position", defaultPoses.default_arm_pos),
    
#     # map cspace. Load previously saved map if it exists
#     Selector("Does map exist?", children=[
#         DoesMapExist("Test for map"),
#         Parallel("Mapping", policy=py_trees.common.ParallelPolicy.SuccessOnOne(), children=[
#             Mapping("map the environment"),
#             Navigation("move around the table")
#         ])
#     ], memory=True),

# ], memory=True)

# tree = FollowWaypoints("")

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