from PyInfiniteTree.Core.TaskStackMachine import Status
from PyInfiniteTree.Interfaces.Behavior import Behavior
import numpy as np
from helpers.misc_helpers import *

class dRotate(Behavior):
    def __init__(self, blackboard, rotate_by, max_speed=0.5):
        '''
        dRotate constructor

        Args:
            rotate_by (float): positive rotation (degrees) for clockwise rotation.
        '''
        super().__init__(blackboard)
        self.MAXSPEED = max_speed
        self.rotateBy = rotate_by
        self.initialHeading = blackboard.get_heading()
        self.runtime = 0

        # Initialize
        self.initialHeading = blackboard.get_heading()
        self.blackboard.leftMotor.setVelocity(0.0)
        self.blackboard.rightMotor.setVelocity(0.0)
        self.runtime = 0
        self.direction = self.rotateBy / abs(self.rotateBy)
        self.vL, self.vR = self.MAXSPEED * self.direction, -self.MAXSPEED * self.direction

    def Step(self, memory, blackboard, message) -> Status:
        memory.push(self)

        requirement_status = self.TraverseRequirements(memory)
        if requirement_status != Status.RUNNING:
            return requirement_status
        
        # Ensure action has been run for at least 1000ms
        # before checking for inactivity
        if self.runtime > 0.2 and abs(blackboard.getTrueAngularVelocity()[1]) < 3.0:
            self.terminate()
            return Status.FAILURE
        
        if abs(blackboard.get_angle_from_to(blackboard.get_heading(), self.initialHeading)[1]) > abs(self.rotateBy):
            self.terminate()
            return Status.SUCCESS

        # print(blackboard.get_angle_from_to(blackboard.get_heading(), self.initialHeading)[1])

        blackboard.leftMotor.setVelocity(self.vL)
        blackboard.rightMotor.setVelocity(self.vR)
        
        self.runtime += blackboard.delta_t
        return Status.RUNNING

    def CheckRequirement(self) -> Status:
        return Status.RUNNING
    
    def terminate(self):
        self.blackboard.leftMotor.setVelocity(0.0)
        self.blackboard.rightMotor.setVelocity(0.0)