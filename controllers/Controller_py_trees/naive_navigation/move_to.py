import py_trees
import numpy as np
from blackboard.blackboard import blackboard
from helpers.misc_helpers import *

class MoveTo(py_trees.behaviour.Behaviour):
    def __init__(self, name, WP):
        super(MoveTo, self).__init__(name)
        self.WP = WP

        # PID Params
        self.KP = 0.0
        self.KI = 0.0
        self.KD = 0.0
        self.previous_error = 0.0
        self.integral = 0.0

    def setup(self):
        self.robot_radius = 0.30 # 0.265
        self.logger.debug("  %s [Foo::setup()]" % self.name)

    def initialise(self):
        print(self.name)
        blackboard.leftMotor.setVelocity(0.0)
        blackboard.rightMotor.setVelocity(0.0)
        self.vL, self.vR = 0.0, 0.0
        reset_PID(self)

        self.logger.debug("  %s [Foo::initialise()]" % self.name)
        
    
    def update(self):
        self.logger.debug("  %s [Foo::update()]" % self.name)

        blackboard.marker.setSFVec3f([*self.WP, 0])

        xw = blackboard.gps.getValues()[0]
        yw = blackboard.gps.getValues()[1]
        
        rho = np.sqrt((xw - self.WP[0])**2 + (yw - self.WP[1])**2)
        
        # reactive correction calculation
        p1 = 0.005 # ADJUST THIS FOR STEERING INFLUENCE
        p2 = 0.01
        rx, ry = 0.0, 0.0
        # for i, angle in enumerate(blackboard.angles):
        #     if blackboard.read('ranges')[i] < 0.45:
        #         rx += 1/blackboard.read('ranges')[i] * np.cos(angle)
        #         ry += 1/blackboard.read('ranges')[i] * np.sin(angle)  

        # print(f"rx: {rx:.3f}, ry: {ry:.3f}")
        
        steering_adjustment = compute_pid_control(self, (xw, yw, 0), (blackboard.compass.getValues()[1],blackboard.compass.getValues()[0], 0), self.WP, blackboard.delta_t)

        # base_speed = 0.8 * blackboard.MAXSPEED
        base_speed = max(0.1 * blackboard.MAXSPEED, 0.8 * blackboard.MAXSPEED - p2*ry)
        self.vL = max(min(base_speed - steering_adjustment*1.0 - rx * p1, blackboard.MAXSPEED), -blackboard.MAXSPEED)
        self.vR = max(min(base_speed + steering_adjustment*1.0 + rx * p1, blackboard.MAXSPEED), -blackboard.MAXSPEED)

        blackboard.leftMotor.setVelocity(self.vL)
        blackboard.rightMotor.setVelocity(self.vR)

        if rho < 0.4:
            return py_trees.common.Status.SUCCESS
        
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )
        # ensure that the robot stops after it reaches its last way point
        blackboard.leftMotor.setVelocity(0.0)
        blackboard.rightMotor.setVelocity(0.0)
    