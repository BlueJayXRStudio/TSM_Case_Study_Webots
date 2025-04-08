# ikpy reference: https://gist.github.com/ItsMichal/4a8fcb330d04f2ccba582286344dd9a7
# Note to graders: this project uses ikpy inverse kinematics library as suggested by the project instructions.
# Please install ikpy by "pip install ikpy" so that this project may compile.

# basic python modules
import numpy as np
import time
from matplotlib import pyplot as plt
from scipy import signal
# BT 
import py_trees
# webots and ikpy
from controller import Supervisor
from ikpy.chain import Chain
from ikpy.link import URDFLink
# custom modules
from helpers.misc_helpers import *
from scipy.spatial.transform import Rotation as R

from blackboard.blackboard import blackboard
from IK_behaviours.stability import Stability

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

# Detect and grab soda can on the dining table
class GrabCan(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(GrabCan, self).__init__(name)

        # IK Setup: Use URDF file export/import
        # note: URDF file was exported from the Tiago robot node directly from Webots
        # self.robot_arm_chain = Chain.from_urdf_file("./IK_behaviours/Robot.urdf", last_link_vector=[0.035,-0.01,-0.25], base_elements=["Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_joint", "TIAGo front arm"])
        self.robot_arm_chain = Chain.from_urdf_file("./IK_behaviours/Robot.urdf", last_link_vector=[0.0,0.0,-0.245], base_elements=["Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_joint", "TIAGo front arm"])

        self.robot_arm_chain_no_offset = Chain.from_urdf_file("./IK_behaviours/Robot.urdf", last_link_vector=[0.0,0.0,0.0], base_elements=["Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_joint", "TIAGo front arm"])

        self.robot_camera_chain = Chain.from_urdf_file("./IK_behaviours/Robot.urdf", base_elements=["Torso", "torso_lift_joint", "torso_lift_link", "head_1_joint", "head_1_link", "head_2_joint", "head_2_link", "head_2_link_camera_joint", "camera"])
        
        # URDF doesn't reflect the true rotation of the camera for some reason.
        # Bug must be on Webot's end not ikpy since the error is reflected in the URDF itself
        roll = -np.pi / 2
        pitch = 0
        yaw = -np.pi / 2
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(roll), -np.sin(roll), 0],
            [0, np.sin(roll),  np.cos(roll), 0], 
            [0, 0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch), 0],
            [0, 1, 0, 0],
            [-np.sin(pitch), 0, np.cos(pitch), 0], 
            [0, 0, 0, 1]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0, 0],
            [np.sin(yaw),  np.cos(yaw), 0, 0],
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ])
        self.R_camera = Rz @ Ry @ Rx

        self.ideal_distance = 0.72

        self.closest_can_id = None
        self.recent_can = None

        self.states = set(['MOVECLOSER', 'MOVEARM', 'CLOSEGAP', 'GRASP'])
        self.state = 'MOVECLOSER'

        self.finger_position = 0.045
        self.temp_var = None

    def setup(self):
        print(self.robot_arm_chain)
        print(self.robot_camera_chain)

        self.logger.debug("  %s [GrabCan::setup()]" % self.name)

    def initialise(self):
        print(self.name)
        
        self.finger_position = 0.045
        self.state = 'MOVECLOSER'

        objects=blackboard.camera.getRecognitionObjects()
        
        self.closest_can_id = None
        dist = float('inf')
        for _object in objects:
            print(_object.getModel(), _object.getId(), list(_object.getPosition()))
            if _object.getModel() == 'can':
                object_position = list(_object.getPosition())
                if object_position[0] < dist:
                    dist = object_position[0]
                    self.closest_can_id = _object.getId()

        self.logger.debug("  %s [GrabCan::initialise()]" % self.name)
        
        self.stability = Stability()
        self.can_stability = Stability()

    def update(self):        
        self.logger.debug("  %s [GrabCan::update()]" % self.name)
        self.stability.update()
        # print("instability: ", self.stability.instability_score)

        relative_pos = None
        objects=blackboard.camera.getRecognitionObjects()
        for _object in objects:
            # print(_object.getModel(), _object.getId(), list(_object.getPosition()))
            if _object.getId() == self.closest_can_id:
                relative_pos = list(_object.getPosition())
                relative_pos = [relative_pos[0], relative_pos[1], relative_pos[2], 1]
                relative_pos = np.array(relative_pos)
        # print(relative_pos)

        left_finger_force = blackboard.joints['gripper_left_finger_joint'].getForceFeedback()
        right_finger_force = blackboard.joints['gripper_right_finger_joint'].getForceFeedback()

        # PREP object relative position
        relative_pos = None
        objects=blackboard.camera.getRecognitionObjects()
        for _object in objects:
            # print(_object.getModel(), _object.getId(), list(_object.getPosition()))
            if _object.getId() == self.closest_can_id:
                relative_pos = list(_object.getPosition())
                self.recent_can = _object
        if relative_pos is None and self.recent_can is not None:
            relative_pos = list(self.recent_can.getPosition())
        relative_pos = [relative_pos[0], relative_pos[1], relative_pos[2], 1]
        relative_pos = np.array(relative_pos)
        
        # get object's POSITION in the robot's base frame
        T = self.robot_camera_chain.forward_kinematics([0, blackboard.lift_height, 0, blackboard.camera_tilt, 0])
        T = T @ np.linalg.inv(self.R_camera)
        object_point =  T @ np.array(relative_pos)
        object_point = object_point[:3]
        # get object's ORIENTATION in the robot's base frame
        euler_angles = get_euler_in_robot_frame(T, self.recent_can.getOrientation())

        # get object WORLD COORDINATE
        xw, yw, theta = self.robot_world_pose()
        object_world_coord = get_object_world_coord(xw, yw, theta, T @ np.array(relative_pos))

        # get object WORLD ORIENTATION
        object_world_rotation = get_object_world_rotation(xw, yw, theta, T, axis_rotation_to_rmat(self.recent_can.getOrientation()))        

        # calculate end-effector info
        curr_pos = self.get_current_position(blackboard.encoders)
        current_arm_position = self.get_end_effector_position(self.robot_arm_chain, curr_pos)
        transformation_matrix = self.robot_arm_chain.forward_kinematics(curr_pos)
        basis_vectors = get_basis_vectors(transformation_matrix)
        
        # get finger FORCE FEEDBACKS
        left_finger_force = blackboard.joints['gripper_left_finger_joint'].getForceFeedback()
        right_finger_force = blackboard.joints['gripper_right_finger_joint'].getForceFeedback()

        # print(f"robot frame coord: {object_point}, world coord: {object_world_coord}, world rotation euler: {object_world_rotation}, robot frame rotation euler: {euler_angles}")

        # ENTRY: Move closer to the can
        if self.state == 'MOVECLOSER':
            # simple P controller
            p1 = 2.0
            p2 = 7.0
            vL = relative_pos[2] * p1 + (relative_pos[0] - self.ideal_distance) * p2
            vR = -relative_pos[2] * p1 + (relative_pos[0] - self.ideal_distance) * p2
            blackboard.leftMotor.setVelocity(vL)
            blackboard.rightMotor.setVelocity(vR)
            if abs(vL) + abs(vR) < 0.01:
                blackboard.leftMotor.setVelocity(0.0)
                blackboard.rightMotor.setVelocity(0.0)
                self.state = 'MOVEARM'

        elif self.state == 'MOVEARM':
            try:
                object_point = np.array([object_point[0], object_point[1], object_point[2]+0.01])
                
                if np.linalg.norm(object_point - np.array(current_arm_position)) > 0.001:
                    result_angles = self.get_arm_position(self.robot_arm_chain, curr_pos, object_point)
                    self.actuate_arm(result_angles, blackboard.joints)

                elif self.stability.instability_score < 0.001:
                    print("ready to close the gap!", self.stability.instability_score)
                    self.state = 'CLOSEGAP'

            except Exception as e:
                print("error: ", e)
                return py_trees.common.Status.RUNNING
            
        elif self.state == 'CLOSEGAP':
            try:
                self.can_stability.update_single(relative_pos)

                object_point = np.array([object_point[0], object_point[1], object_point[2]-0.2])

                forward = -basis_vectors['z']

                result_angles = self.get_arm_position(self.robot_arm_chain, curr_pos, current_arm_position + forward * 0.02)
                self.actuate_arm(result_angles, blackboard.joints)

                if right_finger_force < -3.0 and left_finger_force < -3.0 and self.can_stability.instability_score > 0.005:
                    self.state = 'CLOSEGAP_FINE_TUNE'
                    self.temp_var = 0.0
                    print("ready to grasp!") 
            
            except Exception as e:
                print("error: ", e)
                return py_trees.common.Status.RUNNING
        
        elif self.state == 'CLOSEGAP_FINE_TUNE':
            try:
                self.temp_var += blackboard.delta_t

                self.can_stability.update_single(relative_pos)

                forward = -basis_vectors['z']

                result_angles = self.get_arm_position(self.robot_arm_chain, curr_pos, current_arm_position + xy_plane(forward) * 0.02)
                self.actuate_arm(result_angles, blackboard.joints)

                if self.temp_var > 0.15:
                    self.state = 'GRASP'
                    print("ready to grasp!") 
            
            except Exception as e:
                print("error: ", e)
                return py_trees.common.Status.RUNNING
            
        elif self.state == 'GRASP':
            if left_finger_force + right_finger_force > -30.0:
                self.finger_position -= 0.002
                self.actuate_fingers(self.finger_position, self.finger_position, blackboard.joints)
                return py_trees.common.Status.RUNNING
            else:
                blackboard.write('FINGER_POSITION', self.finger_position)

                blackboard.set_pose(blackboard.default_hold_pose_1, True)

                # REMEMBER currently grabbed can
                blackboard.write('GRABBED', self.closest_can_id)
                return py_trees.common.Status.SUCCESS
            
        else:
            return py_trees.common.Status.FAILURE
        
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )
        blackboard.leftMotor.setVelocity(0.0)
        blackboard.rightMotor.setVelocity(0.0)

    def get_current_position(self, _encoders):
        # print(max(0.0, _encoders['gripper_right_finger_joint'].getValue()))
        return [0,blackboard.lift_height,0] + [ _encoders[f'arm_{i+1}_joint'].getValue() for i in range(7) ] + [0,0,min(0.045, max(0.0, _encoders['gripper_right_finger_joint'].getValue())),0]
    
    def get_arm_position(self, chain, current_position, destination):
        # R_target = np.array([[0, 0, 0], 
        #                      [0, 0, 0],
        #                      [0, 1, 0]])
        
        target_direction = np.array([-1, -1, 0.5], dtype=np.float64)
        quaternion = look_at(target_direction)
        rotation_matrix = R.from_quat(quaternion).as_matrix()

        joint_angles = chain.inverse_kinematics(destination, initial_position=current_position, max_iter=25, target_orientation = rotation_matrix, orientation_mode="all")
        return joint_angles
    
    def actuate_arm(self, angles, _joints):
        for i in range(7):
            key = f'arm_{i+1}_joint'
            _joints[key].setPosition(angles[i+3])

    def actuate_fingers(self, left, right, _joints):
        _joints['gripper_left_finger_joint'].setPosition(left)
        _joints['gripper_right_finger_joint'].setPosition(right)

    def get_end_effector_position(self, chain, current_position):
        transformation_matrix = chain.forward_kinematics(current_position)
        end_effector_position = transformation_matrix[:3, 3]
        return end_effector_position
    
    def robot_world_pose(self):
        xw = blackboard.gps.getValues()[0]
        yw = blackboard.gps.getValues()[1]
        theta = np.arctan2(blackboard.compass.getValues()[0], blackboard.compass.getValues()[1])
        return np.array((xw, yw, theta)) 