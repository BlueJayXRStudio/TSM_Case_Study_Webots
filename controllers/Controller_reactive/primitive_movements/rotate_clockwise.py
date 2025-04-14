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
        self.runtime = 0

    def setup(self):
        self.logger.debug("  %s [LookAt::setup()]" % self.name)

    def initialise(self):
        print(self.name)
        self.logger.debug("  %s [LookAt::initialise()]" % self.name)
        
        blackboard.leftMotor.setVelocity(0.0)
        blackboard.rightMotor.setVelocity(0.0)
        self.runtime = 0
        self.vL, self.vR = self.MAXSPEED, -self.MAXSPEED

    def update(self):
        # print(f"left wheel vel: {blackboard.getLWV()}, right wheel vel: {blackboard.getRWV()}")
        # print(f"true angular velocity: {blackboard.getTrueAngularVelocity()}")
        # print(f"true velocity: {blackboard.getTrueVelocity()}")
        # waypoint = (0, 0)
        # print (f"angle to waypoint {waypoint}: {blackboard.get_angle_to(waypoint)} ")

        for condition in self.preconditions:
            result = condition.CheckRequirement()
            if result != py_trees.common.Status.RUNNING:
                return result

        # Ensure action has been run for at least 1000ms
        # before checking for inactivity
        if self.runtime > 1.0 and abs(blackboard.getTrueAngularVelocity()[1]) < 3.0:
            return py_trees.common.Status.FAILURE

        blackboard.leftMotor.setVelocity(self.vL)
        blackboard.rightMotor.setVelocity(self.vR)
        
        self.runtime += blackboard.delta_t
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )
        
        blackboard.leftMotor.setVelocity(0.0)
        blackboard.rightMotor.setVelocity(0.0)
    