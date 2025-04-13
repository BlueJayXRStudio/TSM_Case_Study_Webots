from PyInfiniteTree.Core.TaskStackMachine import Status
from PyInfiniteTree.Interfaces.Behavior import Behavior
import numpy as np
from helpers.misc_helpers import *

class dMove(Behavior):
    def __init__(self, blackboard, move_by, max_speed=0.5):
        '''

        Args:

        '''
        super().__init__(blackboard)
        self.MAXSPEED = max_speed
        self.moveBy = move_by
        self.initialCoord = blackboard.get_coord()
        self.runtime = 0

        # Initialize
        self.initialCoord = blackboard.get_coord()
        self.blackboard.leftMotor.setVelocity(0.0)
        self.blackboard.rightMotor.setVelocity(0.0)
        self.runtime = 0
        self.direction = self.moveBy / abs(self.moveBy)
        self.vL, self.vR = self.MAXSPEED * self.direction, self.MAXSPEED * self.direction

    def Step(self, memory, blackboard, message) -> Status:
        memory.push(self)

        requirement_status = self.TraverseRequirements()
        if requirement_status != Status.RUNNING:
            return requirement_status
        
        # Ensure action has been run for at least 1000ms
        # before checking for inactivity
        if self.runtime > 0.2 and abs(blackboard.getTrueVelocity()) < 0.01:
            self.terminate()
            return Status.FAILURE
        
        if np.linalg.norm(blackboard.get_coord() - self.initialCoord) > abs(self.moveBy):
            self.terminate()
            return Status.SUCCESS

        blackboard.leftMotor.setVelocity(self.vL)
        blackboard.rightMotor.setVelocity(self.vR)
        
        self.runtime += blackboard.delta_t
        return Status.RUNNING

    def CheckRequirement(self) -> Status:
        return Status.FAILURE
    
    def terminate(self):
        self.blackboard.leftMotor.setVelocity(0.0)
        self.blackboard.rightMotor.setVelocity(0.0)