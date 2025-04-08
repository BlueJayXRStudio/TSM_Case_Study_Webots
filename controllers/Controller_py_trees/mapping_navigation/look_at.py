import py_trees
import numpy as np
from blackboard.blackboard import blackboard
from helpers.misc_helpers import world2map, map2world


# precise navigation with reactive corrections
class LookAt(py_trees.behaviour.Behaviour):
    def __init__(self, name, pos):
        super(LookAt, self).__init__(name)

        # PID Params
        self.KP = 0
        self.KI = 0
        self.KD = 0
        self.previous_error = 0
        self.integral = 0

        self.MAXSPEED = 0.5

        self.look_pos = pos

    def setup(self):
        self.timestep = int(blackboard.robot.getBasicTimeStep())

        self.gps = blackboard.robot.getDevice('gps')
        self.compass = blackboard.robot.getDevice('compass')
        self.lidar = blackboard.robot.getDevice('Hokuyo URG-04LX-UG01')
        self.leftMotor = blackboard.robot.getDevice('wheel_left_joint')
        self.rightMotor = blackboard.robot.getDevice('wheel_right_joint')
        self.robot_radius = 0.30 # 0.265

        self.marker = blackboard.robot.getFromDef("marker").getField("translation")

        self.logger.debug("  %s [LookAt::setup()]" % self.name)

    def initialise(self):
        print(self.name)
        
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)

        self.vL, self.vR = 0.0, 0.0

        self.index = 0

        self.reset_PID()

        self.logger.debug("  %s [LookAt::initialise()]" % self.name)
        
    
    def update(self):
        # print("navigating towards: ", self.WP[self.index], (x, y), "index =", self.index, "length of waypoints:", len(self.WP))
        
        self.logger.debug("  %s [LookAt::update()]" % self.name)

        xw = self.gps.getValues()[0]
        yw = self.gps.getValues()[1]
        
        steering_adjustment = self.compute_pid_control((xw, yw, 0), (self.compass.getValues()[1],self.compass.getValues()[0], 0), (self.look_pos[0], self.look_pos[1], 0), blackboard.delta_t)

        self.vL = max(min(-steering_adjustment, self.MAXSPEED), -self.MAXSPEED)
        self.vR = max(min(steering_adjustment, self.MAXSPEED), -self.MAXSPEED)

        self.leftMotor.setVelocity(self.vL)
        self.rightMotor.setVelocity(self.vR)

        # print(self.compute_dot_product_error((xw, yw, 0), (self.compass.getValues()[1],self.compass.getValues()[0], 0), (self.look_pos[0], self.look_pos[1], 0)))

        if self.compute_dot_product_error((xw, yw, 0), (self.compass.getValues()[1],self.compass.getValues()[0], 0), (self.look_pos[0], self.look_pos[1], 0)) > 0.98:
            return py_trees.common.Status.SUCCESS
        
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)
    
    def reset_PID(self):
        self.KP = 15.0
        self.KI = 0.4
        self.KD = 20.0
        self.previous_error = 0
        self.integral = 0

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