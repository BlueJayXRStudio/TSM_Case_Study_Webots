import py_trees
import numpy as np
from blackboard.blackboard import blackboard
from helpers.misc_helpers import *

# precise navigation with reactive corrections
class RotateClockwise(py_trees.behaviour.Behaviour):
    def __init__(self, name, preconditions, max_speed=0.5):
        super(RotateClockwise, self).__init__(name)
        self.MAXSPEED = max_speed
        self.preconditions = preconditions

    def setup(self):
        self.logger.debug("  %s [LookAt::setup()]" % self.name)

    def initialise(self):
        print(self.name)
        self.logger.debug("  %s [LookAt::initialise()]" % self.name)
        
        blackboard.leftMotor.setVelocity(0.0)
        blackboard.rightMotor.setVelocity(0.0)
        self.vL, self.vR = self.MAXSPEED, -self.MAXSPEED

    def update(self):        
        blackboard.leftMotor.setVelocity(self.vL)
        blackboard.rightMotor.setVelocity(self.vR)

        # print(f"left wheel vel: {blackboard.getLWV()}, right wheel vel: {blackboard.getRWV()}")
        print(f"true angular velocity: {blackboard.getTrueAngularVelocity()}")
        
        for condition in self.preconditions:
            result = condition.CheckRequirement()
            if result != py_trees.common.Status.RUNNING:
                return result

        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )
        
        blackboard.leftMotor.setVelocity(0.0)
        blackboard.rightMotor.setVelocity(0.0)
    