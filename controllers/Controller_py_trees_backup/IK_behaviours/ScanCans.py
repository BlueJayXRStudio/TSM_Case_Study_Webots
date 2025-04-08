import py_trees
import numpy as np
from blackboard.blackboard import blackboard
from helpers.misc_helpers import world2map, map2world

class ScanCans(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(ScanCans, self).__init__(name)

        # PID Params
        self.KP = 0
        self.KI = 0
        self.KD = 0
        self.previous_error = 0
        self.integral = 0

        self.MAXSPEED = 0.5

        self.index = 0
        self.look_waypoints = [(1, 1), (1, 0), (1, -1)]

    def setup(self):
        self.timestep = int(blackboard.robot.getBasicTimeStep())

        self.gps = blackboard.robot.getDevice('gps')
        self.compass = blackboard.robot.getDevice('compass')
        self.lidar = blackboard.robot.getDevice('Hokuyo URG-04LX-UG01')
        self.leftMotor = blackboard.robot.getDevice('wheel_left_joint')
        self.rightMotor = blackboard.robot.getDevice('wheel_right_joint')
        self.robot_radius = 0.30 # 0.265

        self.marker = blackboard.robot.getFromDef("marker").getField("translation")

        self.logger.debug("  %s [ScanCans::setup()]" % self.name)

    def initialise(self):
        print(self.name)
        
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)

        self.vL, self.vR = 0.0, 0.0

        self.index = 0

        self.reset_PID()

        self.logger.debug("  %s [ScanCans::initialise()]" % self.name)
        
    
    def update(self):
        self.logger.debug("  %s [ScanCans::update()]" % self.name)

        xw = self.gps.getValues()[0]
        yw = self.gps.getValues()[1]

        look_pos = np.array((xw, yw)) + np.array(self.look_waypoints[self.index])
        
        steering_adjustment = self.compute_pid_control((xw, yw, 0), (self.compass.getValues()[1],self.compass.getValues()[0], 0), (look_pos[0], look_pos[1], 0), blackboard.delta_t)

        self.vL = max(min(-steering_adjustment, self.MAXSPEED), -self.MAXSPEED)
        self.vR = max(min(steering_adjustment, self.MAXSPEED), -self.MAXSPEED)

        self.leftMotor.setVelocity(self.vL)
        self.rightMotor.setVelocity(self.vR)

        objects=blackboard.camera.getRecognitionObjects()
        for _object in objects:
            if _object.getModel() == 'can':
                return py_trees.common.Status.SUCCESS

        if self.compute_dot_product_error((xw, yw, 0), (self.compass.getValues()[1],self.compass.getValues()[0], 0), (look_pos[0], look_pos[1], 0)) > 0.98:
            self.index += 1
            if self.index > 2:
                return py_trees.common.Status.FAILURE     
        
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
        
        # compute cross product
        dot_product = np.dot(current_heading, vector_to_waypoint)

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
        
        cross_product = np.cross(current_heading, vector_to_waypoint)

        return cross_product[-1]

    def compute_pid_control(self, curr_pos, look_pos, wp, dt):        
        error = self.compute_cross_product_error(curr_pos, look_pos, wp)
        
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.KP * error + self.KI * self.integral + self.KD * derivative
        
        self.previous_error = error
        return output