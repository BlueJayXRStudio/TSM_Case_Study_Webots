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

# separate collection of arm configurations. Also kept as a singleton.
class DefaultPoses:
    # ensure that only one instance can exist
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
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
        
        self.default_nudge_pose = {
            'torso_lift_joint' : self.lift_height,
            'arm_1_joint' : 1.5708,
            'arm_2_joint' : -0.15,
            'arm_3_joint' : 0,
            'arm_4_joint' : 0.0,
            'arm_5_joint' : 0,
            'arm_6_joint' : 1,
            'arm_7_joint' : 0,
            'gripper_left_finger_joint' : 0.045,
            'gripper_right_finger_joint': 0.045,
            'head_1_joint':0,
            'head_2_joint': self.camera_tilt}
        
        self.default_nudge_pose_L = {
            'torso_lift_joint' : self.lift_height,
            'arm_1_joint' : 1.5708,
            'arm_2_joint' : -0.15,
            'arm_3_joint' : 0,
            'arm_4_joint' : 0.0,
            'arm_5_joint' : 0,
            'arm_6_joint' : 1,
            'arm_7_joint' : 0,
            'gripper_left_finger_joint' : 0.045,
            'gripper_right_finger_joint': 0.045,
            'head_1_joint':0,
            'head_2_joint': self.camera_tilt}
        
        self.default_nudge_pose_R = {
            'torso_lift_joint' : self.lift_height,
            'arm_1_joint' : 1.5708,
            'arm_2_joint' : -0.15,
            'arm_3_joint' : 0,
            'arm_4_joint' : 0.0,
            'arm_5_joint' : 0,
            'arm_6_joint' : -1,
            'arm_7_joint' : 0,
            'gripper_left_finger_joint' : 0.045,
            'gripper_right_finger_joint': 0.045,
            'head_1_joint':0,
            'head_2_joint': self.camera_tilt}
        
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
        
        self.release_pose_down = {
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
        
        self.release_pose_up = {
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
        
        self.jar_release_pose_intermediate = {
            'torso_lift_joint' : self.lift_height,
            'arm_1_joint' : 1.5708,
            'arm_2_joint' : 0,
            'arm_3_joint' : -3.0,
            'arm_4_joint' : 1.5708/2,
            'arm_5_joint' : 0,
            'arm_6_joint' : 0,
            'arm_7_joint' : -1.5708/4,
            'gripper_left_finger_joint' : 0.045,
            'gripper_right_finger_joint': 0.045,
            'head_1_joint':0,
            'head_2_joint':self.camera_tilt }

        self.jar_release_pose_down = {
            'torso_lift_joint' : self.lift_height,
            'arm_1_joint' : 1.5708,
            'arm_2_joint' : -1.5708/2,
            'arm_3_joint' : -3.0,
            'arm_4_joint' : 1.5708/2,
            'arm_5_joint' : 0,
            'arm_6_joint' : 0,
            'arm_7_joint' : -1.5708/4,
            'gripper_left_finger_joint' : 0.045,
            'gripper_right_finger_joint': 0.045,
            'head_1_joint':0,
            'head_2_joint':self.camera_tilt }
        
        self.jar_release_pose_up = {
            'torso_lift_joint' : self.lift_height,
            'arm_1_joint' : 1.5708,
            'arm_2_joint' : -1.5708/2,
            'arm_3_joint' : -3.0,
            'arm_4_joint' : 1.5708/1.5,
            'arm_5_joint' : 0,
            'arm_6_joint' : 0,
            'arm_7_joint' : -1.5708/4,
            'gripper_left_finger_joint' : 0.045,
            'gripper_right_finger_joint': 0.045,
            'head_1_joint':0,
            'head_2_joint':self.camera_tilt }
        
        
        self.default_release_pose_2_original = {
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

defaultPoses = DefaultPoses()
