# import sys
# print("python version: ", sys.version) # this project uses python 3.11.4

import numpy as np
from matplotlib import pyplot as plt
# import py_trees Behavior Tree library
import py_trees
from py_trees.composites import Sequence, Parallel, Selector
from py_trees.decorators import Retry

# import all custom mapping and navigation behaviors
from mapping_navigation.navigation import Navigation
from mapping_navigation.database import  DoesMapExist
from mapping_navigation.mapping import Mapping

# import blackboard singleton
from blackboard.blackboard import blackboard
from blackboard.DefaultPoses import defaultPoses

# import all IK related behaviors
from IK_behaviours.ResetArm import ResetArm

# webots API
from controller import Supervisor

# INSTANTIATE ROBOT AND BLACKBOARD
robot = Supervisor()
blackboard.setup(robot)

# MAIN BEHAVIOR TREE
tree = Sequence("Main", children=[
    # set arm to a safe position
    ResetArm("reset arm to safe position", defaultPoses.default_arm_pos),
    
    # map cspace. Load previously saved map if it exists
    Selector("Does map exist?", children=[
        DoesMapExist("Test for map"),
        Parallel("Mapping", policy=py_trees.common.ParallelPolicy.SuccessOnOne(), children=[
            Mapping("map the environment"),
            Navigation("move around the table")
        ])
    ], memory=True),

    ResetArm("reset arm to safe position", defaultPoses.default_arm_pos, True, 1),

], memory=True)


# Invoke setup on all nodes before stepping through
tree.setup_with_descendants()

# step through webots robot and tick behavior tree
while robot.step(blackboard.timestep) != -1:
    # step/tick behavior tree
    tree.tick_once()

    if tree.status in [py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE]:
        print("Tree completed!")
        break