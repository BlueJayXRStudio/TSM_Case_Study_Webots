# the big two
import numpy as np
from matplotlib import pyplot as plt
# import py_trees Behavior Tree library
import py_trees
from py_trees.composites import Sequence, Parallel, Selector
# import all custom behaviors and blackboard
from navigation import Navigation
from database import  DoesMapExist
from mapping import Mapping
from planning import Planning
from blackboard import Blackboard
# webots API
from controller import Supervisor


# INSTANTIATE ROBOT
robot = Supervisor()
# get timestep
timestep = int(robot.getBasicTimeStep())

# Manually define way points for mapping
WP = [(0.614, -0.19), (0.77, -0.94), (0.37, -3.04), (-1.41, -3.39), (-1.53, -3.39), (-1.8, -1.46), (-1.44, 0.38), (0.0, 0.0)]

# INITIALIZE robot components (GPS and Compass) before entry to avoid unforseen error
gps = robot.getDevice('gps')
gps.enable(timestep)
compass = robot.getDevice('compass')
compass.enable(timestep)
# INITIALIZE robot components (Wheels) before entry to avoid unforseen error
leftMotor = robot.getDevice('wheel_left_joint')
rightMotor = robot.getDevice('wheel_right_joint')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
# INITIALIZE robot components (LiDAR) before entry to avoid unforseen error
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# Instantiate blackboard
blackboard = Blackboard()
blackboard.write('robot', robot)
blackboard.write('waypoints', np.concatenate((WP, np.flip(WP, 0)), axis=0))

# Instantiate Behavior Tree. Set memory True for both Sequence and Selector. 
# This will ensure that only the current running node will continue to get executed in subsequent ticks.
tree = Sequence("Main", children=[
    Selector("Does map exist?", children=[
        DoesMapExist("Test for map", blackboard),
        Parallel("Mapping", policy=py_trees.common.ParallelPolicy.SuccessOnOne(), children=[
            Mapping("map the environment", blackboard),
            Navigation("move around the table", blackboard)
        ])
    ], memory=True),
    Planning("compute path to lower left corner", blackboard, (-1.46, -3.12)),
    Navigation("move to lower left corner", blackboard),
    Planning("compute path to sink", blackboard, (0.88, 0.09)),
    Navigation("move to sink", blackboard)
], memory=True)

# Invoke setup on all children
tree.setup_with_descendants()

# step through webots robot
while robot.step(timestep) != -1:
    # step/tick behavior tree
    tree.tick_once() 