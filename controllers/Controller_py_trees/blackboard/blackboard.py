import numpy as np
from collections import deque
from helpers.misc_helpers import *

from .DefaultPoses import defaultPoses

'''
ORIGINAL ARM CONFIG
            |  min  |  max  |
arm_1_joint |  0.07 |  2.68 |
arm_2_joint |  -1.5 |  1.02 |
arm_3_joint | -3.46 |  1.5  |
arm_4_joint | -0.32 |  2.29 |
arm_5_joint | -2.07 |  2.07 |
arm_6_joint | -1.39 |  1.39 |
arm_7_joint | -2.07 |  2.07 |
'''

# Custom blackboard as a singleton and a Webots API adapter    
class Blackboard:
    # ensure that only one instance can exist
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):

        self.bounds = {
            'torso_lift_joint': (  0.0 ,  0.35 ),
            'arm_1_joint': (  0.07 ,  2.68 ),
            'arm_2_joint': (  -1.5 ,  1.02 ),
            'arm_3_joint': ( -3.46 ,  1.5  ),
            'arm_4_joint': ( -0.32 ,  2.29 ),
            'arm_5_joint': ( -2.07 ,  2.07 ),
            'arm_6_joint': ( -1.39 ,  1.39 ),
            'arm_7_joint': ( -2.07 ,  2.07 ),
            'gripper_left_finger_joint': (  0.0 ,  0.045 ),
            'gripper_right_finger_joint': (  0.0 ,  0.045 ),
            'head_1_joint': ( -1.24, 1.24 ),
            'head_2_joint': ( -0.98, 0.79 )
        }
        
        self.data = {}
        self.robot  = None
        self.gps = None
        self.compass = None
        self.lidar = None
        self.camera = None
        self.display = None
        self.leftMotor = None
        self.rightMotor = None
        self.leftWheelSensor = None
        self.rightWheelSensor = None
        self.visual_marker = None
        self.visual_marker_1 = None
        self.marker = None

        self.angles = np.linspace(4.19 / 2 + np.pi/2, -4.19 / 2 + np.pi/2, 667)
        self.angles = self.angles[80:len(self.angles)-80]

        self.MAXSPEED = 10.0

        self.joints = {}
        self.encoders = {}
        self.camera_joint_encoders = {}
        self.camera_encoder_names = ["torso_lift_joint", "head_1_joint", "head_2_joint"]
        self.timestep = 0
        self.delta_t = 0

        self.positionSteps = 10 # how many frames we want to separate between p1 and p0
        self.wheelPositionsL = deque(maxlen=self.positionSteps)
        self.wheelPositionsR = deque(maxlen=self.positionSteps)

        self.headingSteps = 10 # how many frames we want to separate between vec1 and vec0
        self.robotHeadings = deque(maxlen=self.headingSteps)

        self.coordSteps = 10 # how many frames we want to separate between pos1 and pos0
        self.robotCoords = deque(maxlen=self.coordSteps)
        
    def setup(self, robot):
        # get timestep
        timestep = int(robot.getBasicTimeStep())

        # (!) Store all devices and relevant global variables into blackboard and initialize in this entry point to avoid confusion and unforseen errors.
        self.timestep = timestep
        self.delta_t = timestep / 1000.0
        self.robot = robot
        # COMMISSION robot components (GPS and Compass)
        self.gps = robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.compass = robot.getDevice('compass')
        self.compass.enable(self.timestep)
        # COMMISSION robot components (Wheels)
        self.leftMotor = self.robot.getDevice('wheel_left_joint')
        self.rightMotor = self.robot.getDevice('wheel_right_joint')

        self.leftWheelSensor = robot.getDevice("wheel_left_joint_sensor")
        self.rightWheelSensor = robot.getDevice("wheel_right_joint_sensor")
        self.leftWheelSensor.enable(timestep)
        self.rightWheelSensor.enable(timestep)
        self.prevPosL = self.leftWheelSensor.getValue()
        self.prevPosR = self.rightWheelSensor.getValue()

        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)
        # COMMISSION robot components (LiDAR)
        self.lidar = self.robot.getDevice('Hokuyo URG-04LX-UG01')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        # COMMISSION Camera for object recognition
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)

        self.display = self.robot.getDevice('display')

        # COMMISSION markers for visualization purpose
        self.visual_marker = self.robot.getFromDef("marker").getField("translation")
        self.visual_marker_1 = self.robot.getFromDef("marker_1").getField("translation")
        self.marker = blackboard.robot.getFromDef("marker").getField("translation")

        # Manually define way points for mapping
        WP = [(1.09817, 0.3374), (3.18446, 0.92647), (3.18551, 2.41334), (3.18551, 2.41334), (3.02884, 4.19974), (2.96884, 5.76974), (2.06884, 6.76974), (0.978838, 6.43974), (0.978838, 6.43974), (2.06884, 6.76974), (2.96884, 5.76974), (3.02884, 4.19974), (3.18551, 2.41334), (3.18551, 2.41334), (4.58, 3.03), (4.58, 3.03), (5.02, 4.28), (5.02, 4.28), (5.01, 5.54), (5.01, 5.54), (5.02, 4.28), (5.02, 4.28), (7.85, 4.45), (7.85, 4.45), (5.02, 4.28)]
        self.write('mapping_waypoints', np.concatenate((WP, np.flip(WP, 0)), axis=0))

        # Invoke arm set up function
        self.setup_arm()

        print("1. blackboard initialized: ", self)
    
    def write(self, key, value):
        self.data[key] = value
    def read(self, key):
        return self.data.get(key)
    def get_keys(self):
        return self.data.keys()
    
    # set up joints and encoders
    def setup_arm(self):
        # set up actuators
        for name in defaultPoses.joint_names:
            device = self.robot.getDevice(name)
            if device:
                self.joints[name] = device
                self.joints[name].setPosition(defaultPoses.default_arm_pos[name])
            else:
                print(f"Device, {name}, does not exist.")

        self.joints['gripper_left_finger_joint'].enableForceFeedback(self.timestep)
        self.joints['gripper_right_finger_joint'].enableForceFeedback(self.timestep)

        # set up sensors
        self.encoders['gripper_left_finger_joint'] = self.robot.getDevice('gripper_left_sensor_finger_joint')
        self.encoders['gripper_right_finger_joint'] = self.robot.getDevice('gripper_right_sensor_finger_joint')

        self.encoders['gripper_left_finger_joint'].enable(self.timestep)
        self.encoders['gripper_right_finger_joint'].enable(self.timestep)

        for name in defaultPoses.joint_names:
            sensor = self.robot.getDevice(f'{name}_sensor')
            if sensor:
                self.encoders[name] = sensor
                self.encoders[name].enable(self.timestep)
            else:
                print(f"Sensor, {name}_sensor, does not exist.")

        for key in self.camera_encoder_names:
            sensor = self.robot.getDevice(f'{key}_sensor')
            if sensor:
                self.camera_joint_encoders[key] = sensor
                self.camera_joint_encoders[key].enable(self.timestep)
            else:
                print(f"Sensor, {key}_sensor, does not exist.")

    def get_coord(self):
        """
        Gets robot's world coordinates

        Returns:
            np.array: np.array((x, y)) world coordinates as floats.
        """
            
        xw = self.gps.getValues()[0]
        yw = self.gps.getValues()[1]
        return np.array((xw, yw))

    def get_heading(self):
        """
        Gets robot's world heading as unit vector

        Returns:
            np.array: np.array((x, y))
        """

        a = self.compass.getValues()[1]
        b = self.compass.getValues()[0]
        return np.array((a, b))

    # get unitless difference between current measured pose and given target pose
    def get_joint_diff(self, pose, ignore_fingers=False):
        diff_sum = 0
        for name in defaultPoses.joint_names:
            if ignore_fingers and (name == 'gripper_left_finger_joint' or name == 'gripper_right_finger_joint'):
                continue
            diff_sum += abs(pose[name] - self.encoders[name].getValue())
        return diff_sum
    
    # get body world pose
    def get_world_pose(self):
        xw = self.gps.getValues()[0]
        yw = self.gps.getValues()[1]
        theta = np.arctan2(self.compass.getValues()[0], self.compass.getValues()[1])
        return np.array((xw, yw, theta)) 
    
    def get_angle_from_to(self, current_heading, initial_heading):
        """
        get angle of initial_heading in the robot's coordinate frame

        Args:
            wp ((x, y)): world coordinate 2D tuple 

        Returns:
            (rad, deg): the angle in radian and degree
        """
        a = np.array((current_heading[1], current_heading[0], 0), dtype=float)
        b = np.array((initial_heading[1], initial_heading[0], 0), dtype=float)
        
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)

        dot = np.dot(b, a)
        cross = np.cross(b, a)  

        signed_angle_rad = np.arctan2(cross[-1], dot)

        omega_rad = signed_angle_rad
        omega_deg = np.degrees(signed_angle_rad)

        return omega_rad, omega_deg
    
    def get_angle_to(self, wp):
        """
        get angle of wp vector in the robot's coordinate frame

        Args:
            wp ((x, y)): world coordinate 2D tuple 

        Returns:
            (rad, deg): the angle in radian and degree
        """

        if len(self.robotHeadings) < 2:
            return 0

        pose = self.get_world_pose()

        a = np.array((wp[0] - pose[0], wp[1] - pose[1], 0), dtype=float)
        b = np.array((self.compass.getValues()[1], self.compass.getValues()[0], 0), dtype=float)
        
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)

        dot = np.dot(b, a)
        cross = np.cross(b, a)  

        signed_angle_rad = np.arctan2(cross[-1], dot)

        omega_rad = signed_angle_rad
        omega_deg = np.degrees(signed_angle_rad)

        return omega_rad, omega_deg
    
    # get current encoder values as a pose dictionary (for 6DoF arm)
    def get_pose(self):
        curr_pose = {}
        for name in defaultPoses.joint_names:
            curr_pose[name] = min(self.bounds[name][1], max(self.bounds[name][0], self.encoders[name].getValue()))
        return curr_pose
    
    # set arm pose
    def set_pose(self, pose, ignore_fingers=False):
        for name in defaultPoses.joint_names:
            if ignore_fingers and (name == 'gripper_left_finger_joint' or name == 'gripper_right_finger_joint'):
                continue
            self.joints[name].setPosition(min(self.bounds[name][1], max(self.bounds[name][0], pose[name])))

    # we will resort to linear interpolation of joint angles to smooth out movements, since we'll primarily rely on setting position rather than torque.
    def lerp(self, startPos, endPos, curr_step, max_steps=10, ignore_fingers=False):
        lerped_pose = {}
        for name in list(startPos.keys()):
            if ignore_fingers and (name == 'gripper_left_finger_joint' or name == 'gripper_right_finger_joint'):
                lerped_pose[name] = self.read('FINGER_POSITION')
                continue
                
            delta = endPos[name]-startPos[name]
            step = delta*curr_step/max_steps
            # print("confirming final step: ", curr_step == max_steps)
            lerped_pose[name] = startPos[name] + step
            # print(name, lerped_pose[name])
            
        return lerped_pose
    
    def pause_arm(self):
        for name in defaultPoses.joint_names:
            self.joints[name].setPosition(self.encoders[name].getValue())
    
    def update_velocity(self):
        self.wheelPositionsL.append(self.leftWheelSensor.getValue())
        self.wheelPositionsR.append(self.rightWheelSensor.getValue())

    def update_true_angular_velocity(self):
        self.robotHeadings.append((self.compass.getValues()[1], self.compass.getValues()[0], 0))

    def update_true_velocity(self):
        xw = blackboard.gps.getValues()[0]
        yw = blackboard.gps.getValues()[1]
        self.robotCoords.append((xw, yw))

    # get left wheel velocity
    def getLWV(self):
        if len(self.wheelPositionsL) < 2:
            return 0
        return (self.wheelPositionsL[-1] - self.wheelPositionsL[0]) / (self.delta_t * self.positionSteps)

    # get right wheel velocity
    def getRWV(self):
        if len(self.wheelPositionsR) < 2:
            return 0
        return (self.wheelPositionsR[-1] - self.wheelPositionsR[0]) / (self.delta_t * self.positionSteps)

    # get true angular velocity
    def getTrueAngularVelocity(self):
        if len(self.robotHeadings) < 2:
            return 0

        a = np.array(self.robotHeadings[0], dtype=float)
        b = np.array(self.robotHeadings[-1], dtype=float)

        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)

        dot = np.dot(b, a)
        cross = np.cross(b, a)  

        signed_angle_rad = np.arctan2(cross[-1], dot)

        omega_rad = signed_angle_rad / (self.delta_t * self.positionSteps)
        omega_deg = np.degrees(signed_angle_rad) / (self.delta_t * self.positionSteps)

        return omega_rad, omega_deg
    
    # get true positional velocity
    def getTrueVelocity(self):
        if len(self.robotCoords) < 2:
            return 0

        a = np.array(self.robotCoords[0], dtype=float)
        b = np.array(self.robotCoords[-1], dtype=float)

        return np.linalg.norm(b - a) / (self.delta_t * self.positionSteps)

    def isLookingAt(self, WP, threshold):
        xw = blackboard.gps.getValues()[0]
        yw = blackboard.gps.getValues()[1]
        
        if compute_dot_product_error((xw, yw, 0), (self.compass.getValues()[1], self.compass.getValues()[0], 0), (WP[0], WP[1], 0)) > threshold:
            return True
        return False

blackboard = Blackboard()
