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
from mapping_navigation.fine_navigation import FineNavigation
from mapping_navigation.look_at import LookAt
from mapping_navigation.database import  DoesMapExist
from mapping_navigation.mapping import Mapping
from mapping_navigation.reactive_mapping import ReactiveMapping
from mapping_navigation.planning import Planning

# import blackboard singleton
from blackboard.blackboard import blackboard
from blackboard.DefaultPoses import defaultPoses

# import all IK related behaviors
from IK_behaviours.ResetArm import ResetArm
from controllers.Controller_py_trees.IK_behaviours.GrabCan import GrabCan
from controllers.Controller_py_trees.IK_behaviours.ScanCans import ScanCans
from IK_behaviours.TrackNearestObject import TrackNearestObject

# import miscellaneous behaviors
from misc_behaviours.close_position_gap import ClosePositionGap
from misc_behaviours.pause import Pause

# webots API
from controller import Supervisor

# INSTANTIATE ROBOT
robot = Supervisor()
# get timestep
timestep = int(robot.getBasicTimeStep())

# (!) Store all devices and relevant global variables into blackboard and initialize in this entry point to avoid confusion and unforseen errors.
blackboard.timestep = timestep
blackboard.delta_t = timestep / 1000.0
blackboard.robot = robot
# COMMISSION robot components (GPS and Compass)
blackboard.gps = robot.getDevice('gps')
blackboard.gps.enable(blackboard.timestep)
blackboard.compass = robot.getDevice('compass')
blackboard.compass.enable(blackboard.timestep)
# COMMISSION robot components (Wheels)
blackboard.leftMotor = blackboard.robot.getDevice('wheel_left_joint')
blackboard.rightMotor = blackboard.robot.getDevice('wheel_right_joint')

blackboard.leftWheelSensor = robot.getDevice("wheel_left_joint_sensor")
blackboard.rightWheelSensor = robot.getDevice("wheel_right_joint_sensor")
blackboard.leftWheelSensor.enable(timestep)
blackboard.rightWheelSensor.enable(timestep)

blackboard.leftMotor.setPosition(float('inf'))
blackboard.rightMotor.setPosition(float('inf'))
blackboard.leftMotor.setVelocity(0.0)
blackboard.rightMotor.setVelocity(0.0)
# COMMISSION robot components (LiDAR)
blackboard.lidar = blackboard.robot.getDevice('Hokuyo URG-04LX-UG01')
blackboard.lidar.enable(blackboard.timestep)
blackboard.lidar.enablePointCloud()
# COMMISSION Camera for object recognition
blackboard.camera = blackboard.robot.getDevice("camera")
blackboard.camera.enable(blackboard.timestep)
blackboard.camera.recognitionEnable(blackboard.timestep)

# COMMISSION markers for visualization purpose
blackboard.visual_marker = blackboard.robot.getFromDef("marker").getField("translation")
blackboard.visual_marker_1 = blackboard.robot.getFromDef("marker_1").getField("translation")

# Manually define way points for mapping
WP = [(0.614, -0.19), (0.77, -0.94), (0.37, -3.04), (-1.41, -3.39), (-1.53, -3.39), (-1.8, -1.46), (-1.44, 0.38), (0.0, 0.0)]
blackboard.write('mapping_waypoints', np.concatenate((WP, np.flip(WP, 0)), axis=0))

# Invoke arm set up function
blackboard.setup_arm()

print("1. blackboard initialized: ", blackboard)


'''
can-to-island sequence construction wrapped in a function to allow fine tuning

can-to-island sequence is capable of dynamically scanning for a can, grabbing it, transporting to kitchen island, and finally releasing it onto the island. It uses a combination of default arm configurations and IK solution.

Movement in this sequence uses A* pathfinding for planning and quad-tree for obstacle detection and path collision. A* planning will reoccur if obstacles are detected in the path. This higher level dynamic planning and navigation is augmented by reactive behavior to slightly influence the steer of the robot with the immediate laser data.
'''
def build_can_to_island(kitchen_wp, corner_wp, look_wp_1, look_wp_2):
    can_to_island = Sequence("Can to Island Sequence", children=[
        # wrap path planning and navigation under retry decorations to allow dynamic path planning
        # failure will occur when obstacles are detected in the current path
        Retry("Allow retries",
            child=Parallel("simultaneous mapping and navigation", policy=py_trees.common.ParallelPolicy.SuccessOnOne(), children=[
            ReactiveMapping("live scanning the environment"),
            Sequence("planning and navigating", children=[
                Planning("planning path to given destination", kitchen_wp),
                FineNavigation("fine navigation with PID control", False),
            ], memory = True)]
        ), num_failures=1000000),

        # manually set arm to favorable positions
        ResetArm("reset arm to grab position", blackboard.default_grab_pose, False),
        
        # SCAN for available cans
        ScanCans("scanning for cans"),
        
        # GRAB found cans
        Retry("Allow retries", child=GrabCan("grabbing a can"), num_failures=1000000),
        LookAt("looking towards a destination", corner_wp),
        ResetArm("reset arm to hold pose 2", blackboard.default_hold_pose_2, True),
        ResetArm("reset arm to hold pose 3", blackboard.default_hold_pose_3, True),
        
        Retry("Allow retries",
            child=Parallel("simultaneous mapping and navigation", policy=py_trees.common.ParallelPolicy.SuccessOnOne(), children=[
            ReactiveMapping("live scanning the environment"),
            Sequence("planning and navigating", children=[
                Planning("planning path to given destination", corner_wp),
                FineNavigation("fine navigation with PID control", False),
            ], memory = True)]
        ), num_failures=1000000),
        
        # RELEASE sequence
        LookAt("looking towards a destination", look_wp_1),
        ResetArm("reset arm to release pose", blackboard.default_release_pose_1, True, 200),
        ResetArm("reset arm to release pose", blackboard.default_release_pose_2, True, 200),
        
        # track nearest object and reposition relative to that object for finer jar placement
        TrackNearestObject("tracking nearest object"),
        
        # manual release sequence. Adjust parameters to optimize speed
        ResetArm("repositioning", defaultPoses.jar_release_pose_intermediate, True, 200),
        Pause("pausing", 1.0),
        ResetArm("repositioning", defaultPoses.jar_release_pose_up, True, 200),
        Pause("pausing", 1.0),
        ResetArm("setting down", defaultPoses.jar_release_pose_down, True, 200),
        Pause("pausing", 1.0),
        ResetArm("actually releasing", defaultPoses.jar_release_pose_down, False, 200),
        Pause("pausing", 1.0),
        ResetArm("moving arm up", defaultPoses.jar_release_pose_up, False, 200),
        Pause("pausing", 1.0),
        
        # FINISH up
        ResetArm("reset arm to hold pose 1", blackboard.default_hold_pose_1, False),
        LookAt("looking towards a destination", look_wp_2),
        ResetArm("reset arm to hold pose 2", blackboard.default_hold_pose_2, False),
        ResetArm("reset arm to hold pose 3", blackboard.default_hold_pose_3, False),        
    ], memory=True)

    return can_to_island

# MAIN BEHAVIOR TREE
tree = Sequence("Main", children=[
    # set arm to a safe position
    ResetArm("reset arm to safe position", blackboard.default_arm_pos),
    
    # map cspace. Load previously saved map if it exists
    Selector("Does map exist?", children=[
        DoesMapExist("Test for map"),
        Parallel("Mapping", policy=py_trees.common.ParallelPolicy.SuccessOnOne(), children=[
            Mapping("map the environment"),
            Navigation("move around the table")
        ])
    ], memory=True),

    # Soda can to kitchen island sequence for 3 jars
    LookAt("looking at kitchen island", (0.75, 0.20)),
    build_can_to_island((0.75, 0.20), (-1.43, -2.85), (-0.35, -2.0), ( -1.06793, -3.19647)),
    build_can_to_island((0.75, 0.20), (-1.30, -2.85), (-0.50, -2.0), ( -1.06793, -3.19647)),
    build_can_to_island((0.75, 0.20), (-1.30, -2.85), (-0.50, -2.0), ( -1.06793, -3.19647)),

    ResetArm("reset arm to safe position", blackboard.default_arm_pos, True, 1),

], memory=True)


# Invoke setup on all nodes before stepping through
tree.setup_with_descendants()

# step through webots robot and tick behavior tree
while robot.step(timestep) != -1:
    # step/tick behavior tree
    tree.tick_once()

    if tree.status in [py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE]:
        print("Tree completed!")
        break