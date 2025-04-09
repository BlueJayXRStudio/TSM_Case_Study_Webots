'''
miscellaneous helper functions.
mostly 3D maths
'''

import numpy as np
from scipy.spatial.transform import Rotation

# turns world coordinates to map coordinates
def world2map(xw, yw):
    px = int((xw + 2.25) * 40)
    py = int((yw - 2) * (-50))

    px = min(px, 199)
    py = min(py, 299)
    px = max(px, 0)
    py = max(py, 0)

    return [px, py]

# turns map coordinates to world coordinates. Inverse of world2map
def map2world(px, py):
    xw = px / 40 - 2.25
    yw = py / (-50) + 2
    return [xw, yw]

# returns a dictionary of basis vectors given a 4x4 homogeneous transformation matrix
def get_basis_vectors(transformation_matrix):
    if transformation_matrix.shape != (4, 4):
        raise ValueError("Input must be a 4x4 homogeneous transformation matrix.")

    # Extract basis vectors from the rotation part
    x_basis = transformation_matrix[:3, 0]
    y_basis = transformation_matrix[:3, 1]
    z_basis = transformation_matrix[:3, 2]

    return {
        "x": x_basis,
        "y": y_basis,
        "z": z_basis }

# since webots returns rotation in axis rotation, we'll have to convert it to rotation matrix
def axis_rotation_to_rmat(rotation):
    axis = rotation[:3]
    theta = rotation[-1]

    axis = axis / np.linalg.norm(axis)
    rotation_matrix = Rotation.from_rotvec(axis * theta).as_matrix()

    return rotation_matrix

def get_coord_in_robot_frame_3(T, camera_relative_pos):
    robot_frame_coord =  T @ np.array(camera_relative_pos)
    return robot_frame_coord[:3]

def get_coord_in_robot_frame(T, camera_relative_pos):
    robot_frame_coord =  T @ np.array(camera_relative_pos)
    return robot_frame_coord

def get_euler_in_robot_frame(parent_matrix, relative_rotation):
    relative_rotation_matrix = axis_rotation_to_rmat(relative_rotation)
    global_mat = parent_matrix[:3, :3]

    q_global = Rotation.from_matrix(global_mat).as_quat()
    q_relative = Rotation.from_matrix(relative_rotation_matrix).as_quat()
    
    q_final = Rotation.from_quat(q_global) * Rotation.from_quat(q_relative)

    euler_angles = q_final.as_euler('xyz', degrees=True)

    return euler_angles

def get_rotation_in_robot_frame(parent_matrix, relative_rotation):
    global_mat = parent_matrix[:3, :3]
    relative_rotation_matrix = axis_rotation_to_rmat(relative_rotation)
    
    q_global = Rotation.from_matrix(global_mat)
    q_relative = Rotation.from_matrix(relative_rotation_matrix)
    
    return q_global * q_relative

# DEDUCE WORLD COORDINATE (takes in already converted camera relative position)
def get_object_world_coord(xw, yw, theta, object_robot_frame_coord):
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0, xw],
        [np.sin(theta),  np.cos(theta), 0, yw],
        [0, 0, 1, 0.0956197],
        [0, 0, 0, 1]
    ])
    return R @ object_robot_frame_coord

# DEDUCE WORLD ROTATION (takes in raw camera relative rotation)
def get_object_world_rotation(xw, yw, theta, _T, camera_relative_rotation):
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0, xw],
        [np.sin(theta),  np.cos(theta), 0, yw],
        [0, 0, 1, 0.0956197],
        [0, 0, 0, 1]
    ])
    parent_matrix = R @ _T
    parent_rotation = Rotation.from_matrix(parent_matrix[:3, :3])
    return (parent_rotation * Rotation.from_matrix(camera_relative_rotation)).as_euler('xyz', degrees=True)

# return look_at rotation to use for orienting arm end effector. Much easier, reliable and interpretable than euler angles.
def look_at(target_direction, up_vector=np.array([0, 0, 1], dtype=np.float64)):
    target_direction = np.array(target_direction, dtype=np.float64)
    up_vector = np.array(up_vector, dtype=np.float64)
    target_direction /= np.linalg.norm(target_direction)
    up_vector /= np.linalg.norm(up_vector)
    right_vector = np.cross(up_vector, target_direction)
    right_vector /= np.linalg.norm(right_vector)
    up_vector = np.cross(target_direction, right_vector)
    rotation_matrix = np.vstack([right_vector, up_vector, target_direction]).T
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
    return quaternion

# zero out z-axis. May turn out useful
def xy_plane(vec):
    return np.array([vec[0], vec[1], 0.0])

def compute_dot_product_error(self, current_pos, current_heading, look_pos):
    # compute vector from robot to waypoint
    vector_to_waypoint = np.array((look_pos[0] - current_pos[0], look_pos[1] - current_pos[1], 0))
    
    # normalize vectors
    mag_vtw = np.linalg.norm(vector_to_waypoint)
    mag_ch = np.linalg.norm(current_heading)
    
    # avoid division by zero
    if mag_vtw == 0 or mag_ch == 0:
        return 0  

    vector_to_waypoint = vector_to_waypoint / mag_vtw
    current_heading = current_heading / mag_ch
    
    # print(current_heading, vector_to_waypoint)
    # compute cross product
    dot_product = np.dot(current_heading, vector_to_waypoint)
    # print("cross product: ", cross_product)
    # positive: turn left; negative: turn right
    # print(dot_product)
    return dot_product

def compute_cross_product_error(self, current_pos, current_heading, look_pos):
    # compute vector from robot to waypoint
    vector_to_waypoint = np.array((look_pos[0] - current_pos[0], look_pos[1] - current_pos[1], 0))
    
    # normalize vectors
    mag_vtw = np.linalg.norm(vector_to_waypoint)
    mag_ch = np.linalg.norm(current_heading)
    
    # avoid division by zero
    if mag_vtw == 0 or mag_ch == 0:
        return 0  

    vector_to_waypoint = vector_to_waypoint / mag_vtw
    current_heading = current_heading / mag_ch
    
    # print(current_heading, vector_to_waypoint)
    # compute cross product
    cross_product = np.cross(current_heading, vector_to_waypoint)
    # print("cross product: ", cross_product)
    # positive: turn left; negative: turn right
    # print(cross_product[-1])
    return cross_product[-1]

def compute_pid_control(self, curr_pos, look_pos, wp, dt):        
    error = self.compute_cross_product_error(curr_pos, look_pos, wp)
    
    self.integral += error * dt
    derivative = (error - self.previous_error) / dt
    output = self.KP * error + self.KI * self.integral + self.KD * derivative
    
    self.previous_error = error
    return output