import py_trees
import numpy as np
from blackboard.blackboard import blackboard
from helpers.misc_helpers import *


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
        self.robot_radius = 0.30 # 0.265
        self.marker = blackboard.robot.getFromDef("marker").getField("translation")

        self.logger.debug("  %s [LookAt::setup()]" % self.name)

    def initialise(self):
        print(self.name)
        
        blackboard.leftMotor.setVelocity(0.0)
        blackboard.rightMotor.setVelocity(0.0)

        self.vL, self.vR = 0.0, 0.0

        self.index = 0

        reset_PID(self)

        self.logger.debug("  %s [LookAt::initialise()]" % self.name)
        
    
    def update(self):
        # print("navigating towards: ", self.WP[self.index], (x, y), "index =", self.index, "length of waypoints:", len(self.WP))
        
        self.logger.debug("  %s [LookAt::update()]" % self.name)

        xw = blackboard.gps.getValues()[0]
        yw = blackboard.gps.getValues()[1]
        
        steering_adjustment = compute_pid_control((xw, yw, 0), (blackboard.compass.getValues()[1],blackboard.compass.getValues()[0], 0), (self.look_pos[0], self.look_pos[1], 0), blackboard.delta_t)

        self.vL = max(min(-steering_adjustment, self.MAXSPEED), -self.MAXSPEED)
        self.vR = max(min(steering_adjustment, self.MAXSPEED), -self.MAXSPEED)

        blackboard.leftMotor.setVelocity(self.vL)
        blackboard.rightMotor.setVelocity(self.vR)

        # print(compute_dot_product_error((xw, yw, 0), (self.compass.getValues()[1],self.compass.getValues()[0], 0), (self.look_pos[0], self.look_pos[1], 0)))

        if compute_dot_product_error((xw, yw, 0), (blackboard.compass.getValues()[1], blackboard.compass.getValues()[0], 0), (self.look_pos[0], self.look_pos[1], 0)) > 0.98:
            return py_trees.common.Status.SUCCESS
        
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )
        blackboard.leftMotor.setVelocity(0.0)
        blackboard.rightMotor.setVelocity(0.0)
    