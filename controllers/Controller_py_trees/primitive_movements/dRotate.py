import py_trees
import numpy as np
from blackboard.blackboard import blackboard
from helpers.misc_helpers import *

# precise navigation with reactive corrections
class dRotate(py_trees.behaviour.Behaviour):
    def __init__(self, name, preconditions, rotate_by, max_speed=0.5):
        super(dRotate, self).__init__(name)
        self.MAXSPEED = max_speed
        self.rotateBy = rotate_by
        # self.initialHeading = blackboard.get_world_pose
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
    