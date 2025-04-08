import numpy as np

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

        '''
        Hardcoded joint configs below. Please note this project does indeed use IK, but in conjunction with manual arm configurations. IK alone struggles to converge on ideal arm paths. A more sophisticated approach would be to use A* or RRT on the arm configurations itself, as strongly hinted by the lectures. However, I will reserve this for future personal projects as our current sensor setup is not ideal for creating 3D point cloud data, and things like octtrees and non-3D multidimensional path finding is a bit out of the scope for this course.
        '''
        # !! Note: Both sensor and actuator look up tables use the same names. Should simplify things hopefully.
        self.lift_height = 0.35
        self.camera_tilt = -0.5

        self.joint_names = [
            'torso_lift_joint',
            'arm_1_joint',
            'arm_2_joint',
            'arm_3_joint',
            'arm_4_joint',
            'arm_5_joint',
            'arm_6_joint',
            'arm_7_joint',
            'gripper_left_finger_joint',
            'gripper_right_finger_joint',
            'head_1_joint',
            'head_2_joint' ]
        
        self.default_arm_pos = {
            'torso_lift_joint' : 0.0,
            'arm_1_joint' : 1.5708,
            'arm_2_joint' : 0.9,
            'arm_3_joint' : -3.14,
            'arm_4_joint' : 0.99,
            'arm_5_joint' : -1.5708,
            'arm_6_joint' : 1.0,
            'arm_7_joint' : 0,
            'gripper_left_finger_joint' : 0.045,
            'gripper_right_finger_joint': 0.045,
            'head_1_joint':0,
            'head_2_joint': 0}
        
        self.default_grab_pose = {
            'torso_lift_joint' : self.lift_height,
            'arm_1_joint' : 1.5708/4,
            'arm_2_joint' : 1.02,
            'arm_3_joint' : -1.0,
            'arm_4_joint' : 2.0,
            'arm_5_joint' : 0,
            'arm_6_joint' : 0.5,
            'arm_7_joint' : 2.0,
            'gripper_left_finger_joint' : 0.045,
            'gripper_right_finger_joint': 0.045,
            'head_1_joint':0,
            'head_2_joint':self.camera_tilt }
        
        self.default_release_pose_1 = {
            'torso_lift_joint' : self.lift_height,
            'arm_1_joint' : 1.5708,
            'arm_2_joint' : 0.9,
            'arm_3_joint' : -3.14,
            'arm_4_joint' : 0.99,
            'arm_5_joint' : -1.5708,
            'arm_6_joint' : 1.0,
            'arm_7_joint' : 0,
            'gripper_left_finger_joint' : 0.045,
            'gripper_right_finger_joint': 0.045,
            'head_1_joint':0,
            'head_2_joint': self.camera_tilt}

        self.default_release_pose_2 = {
            'torso_lift_joint' : self.lift_height,
            'arm_1_joint' : 1.5708/4,
            'arm_2_joint' : 1.02,
            'arm_3_joint' : -1.0,
            'arm_4_joint' : 1.5,
            'arm_5_joint' : 0,
            'arm_6_joint' : 0.5,
            'arm_7_joint' : 2.0,
            'gripper_left_finger_joint' : 0.045,
            'gripper_right_finger_joint': 0.045,
            'head_1_joint':0,
            'head_2_joint':self.camera_tilt }

        self.default_hold_pose_1 = {
            'torso_lift_joint' : self.lift_height,
            'arm_1_joint' : 1.5708/4,
            'arm_2_joint' : 1.02,
            'arm_3_joint' : -1.7,
            'arm_4_joint' : 2.29,
            'arm_5_joint' : 0,
            'arm_6_joint' : 0.5,
            'arm_7_joint' : 2.0,
            'gripper_left_finger_joint' : 0.045,
            'gripper_right_finger_joint': 0.045,
            'head_1_joint':0,
            'head_2_joint': self.camera_tilt}
        
        self.default_hold_pose_2 = {
            'torso_lift_joint' : 0.0,
            'arm_1_joint' : 1.5708/4,
            'arm_2_joint' : 1.02,
            'arm_3_joint' : -2.0,
            'arm_4_joint' : 0.99,
            'arm_5_joint' : 0,
            'arm_6_joint' : 0.5,
            'arm_7_joint' : 2.0,
            'gripper_left_finger_joint' : 0.045,
            'gripper_right_finger_joint': 0.045,
            'head_1_joint':0,
            'head_2_joint': self.camera_tilt}
        
        self.default_hold_pose_3 = {
            'torso_lift_joint' : 0.0,
            'arm_1_joint' : 1.5708,
            'arm_2_joint' : 0.9,
            'arm_3_joint' : -3.14,
            'arm_4_joint' : 0.99,
            'arm_5_joint' : -1.5708,
            'arm_6_joint' : 1.0,
            'arm_7_joint' : 0,
            'gripper_left_finger_joint' : 0.045,
            'gripper_right_finger_joint': 0.045,
            'head_1_joint':0,
            'head_2_joint': self.camera_tilt}
        
    def write(self, key, value):
        self.data[key] = value
    def read(self, key):
        return self.data.get(key)
    def get_keys(self):
        return self.data.keys()
    
    # set up joints and encoders
    def setup_arm(self):
        # set up actuators
        for name in self.joint_names:
            device = self.robot.getDevice(name)
            if device:
                self.joints[name] = device
                self.joints[name].setPosition(self.default_arm_pos[name])
            else:
                print(f"Device, {name}, does not exist.")

        self.joints['gripper_left_finger_joint'].enableForceFeedback(self.timestep)
        self.joints['gripper_right_finger_joint'].enableForceFeedback(self.timestep)

        # set up sensors
        self.encoders['gripper_left_finger_joint'] = self.robot.getDevice('gripper_left_sensor_finger_joint')
        self.encoders['gripper_right_finger_joint'] = self.robot.getDevice('gripper_right_sensor_finger_joint')

        self.encoders['gripper_left_finger_joint'].enable(self.timestep)
        self.encoders['gripper_right_finger_joint'].enable(self.timestep)

        for name in self.joint_names:
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
        for name in self.joint_names:
            if ignore_fingers and (name == 'gripper_left_finger_joint' or name == 'gripper_right_finger_joint'):
                continue
            diff_sum += abs(pose[name] - self.encoders[name].getValue())
        return diff_sum
    
    # get current encoder values as a pose dictionary 
    def get_pose(self):
        curr_pose = {}
        for name in self.joint_names:
            curr_pose[name] = min(self.bounds[name][1], max(self.bounds[name][0], self.encoders[name].getValue()))
        return curr_pose
    
    # set arm pose
    def set_pose(self, pose, ignore_fingers=False):
        for name in self.joint_names:
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
        for name in self.joint_names:
            self.joints[name].setPosition(self.encoders[name].getValue())
    
blackboard = Blackboard()
