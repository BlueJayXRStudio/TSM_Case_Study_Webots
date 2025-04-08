import numpy as np
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

# Custom blackboard as a singleton     
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
        self.leftMotor = None
        self.rightMotor = None
        self.leftWheelSensor = None
        self.rightWheelSensor = None
        self.visual_marker = None
        self.visual_marker_1 = None

        self.angles = np.linspace(4.19 / 2 + np.pi/2, -4.19 / 2 + np.pi/2, 667)
        self.angles = self.angles[80:len(self.angles)-80]

        self.MAXSPEED = 6.0

        self.joints = {}
        self.encoders = {}
        self.camera_joint_encoders = {}
        self.camera_encoder_names = ["torso_lift_joint", "head_1_joint", "head_2_joint"]
        self.timestep = 0
        self.delta_t = 0
        
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

        # COMMISSION markers for visualization purpose
        self.visual_marker = self.robot.getFromDef("marker").getField("translation")
        self.visual_marker_1 = self.robot.getFromDef("marker_1").getField("translation")

        # Manually define way points for mapping
        WP = [(1.08817, 0.567395), (3.18446, 0.866467), (3.25551, 2.30334), (3.25551, 2.30334), (3.02884, 4.19974), (3.02884, 5.76974), (2.06884, 6.76974), (0.978838, 6.43974), (0.978838, 6.43974), (2.06884, 6.76974), (3.02884, 5.76974), (3.02884, 4.19974), (3.25551, 2.30334), (3.25551, 2.30334), (4.58, 3.03), (4.58, 3.03), (5.02, 4.28), (5.02, 4.28), (5.01, 5.54), (5.01, 5.54), (5.02, 4.28), (5.02, 4.28), (7.85, 4.45), (7.85, 4.45), (5.02, 4.28)]
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

    # get unitless difference between current measured pose and given target pose
    def get_joint_diff(self, pose, ignore_fingers=False):
        diff_sum = 0
        for name in defaultPoses.joint_names:
            if ignore_fingers and (name == 'gripper_left_finger_joint' or name == 'gripper_right_finger_joint'):
                continue
            diff_sum += abs(pose[name] - self.encoders[name].getValue())
        return diff_sum
    
    # get current encoder values as a pose dictionary 
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
    
blackboard = Blackboard()
