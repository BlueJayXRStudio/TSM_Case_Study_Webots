import py_trees
import numpy as np
from blackboard.blackboard import blackboard
from helpers.misc_helpers import *

# precise navigation with reactive corrections
class MoveBackwards(py_trees.behaviour.Behaviour):
    def __init__(self, name, preconditions, max_speed=0.5, min_dist=0.05):
        super(MoveBackwards, self).__init__(name)
        self.MAXSPEED = max_speed
        self.preconditions = preconditions
        self.initial_position = None
        self.min_dist = min_dist
        self.runtime = 0

    def setup(self):
        self.logger.debug(f"  {self.name} [Foo::setup()]")

    def initialise(self):
        print(self.name)
        self.logger.debug(f"  {self.name} [Foo::initialise()]")
        
        blackboard.leftMotor.setVelocity(0.0)
        blackboard.rightMotor.setVelocity(0.0)
        self.vL, self.vR = -self.MAXSPEED, -self.MAXSPEED
        self.initial_position = blackboard.get_coord()
        self.runtime = 0

    def update(self):                
        # print(f"left wheel vel: {blackboard.getLWV()}, right wheel vel: {blackboard.getRWV()}")
        # print(f"true angular velocity: {blackboard.getTrueAngularVelocity()}")
        # print(f"true velocity: {blackboard.getTrueVelocity()}")

        for condition in self.preconditions:
            result = condition.CheckRequirement()
            if result != py_trees.common.Status.RUNNING:
                return result

        if np.linalg.norm(blackboard.get_coord()-self.initial_position) > self.min_dist:
            return py_trees.common.Status.FAILURE
        
        # Ensure action has been run for at least 1000ms
        # before checking for inactivity
        if self.runtime > 1.0 and abs(blackboard.getTrueVelocity()) < 0.01:
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
    