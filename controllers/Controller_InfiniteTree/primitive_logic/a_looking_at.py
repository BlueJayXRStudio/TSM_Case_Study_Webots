from PyInfiniteTree.Core.TaskStackMachine import Status
from PyInfiniteTree.Interfaces.Behavior import Behavior
import numpy as np
from helpers.misc_helpers import *

class a_LookingAt(Behavior):
    def __init__(self, blackboard, WP):
        super().__init__(blackboard)
        self.WP = WP

    def Step(self, memory, blackboard, message) -> Status:
        memory.push(self)
        return self.CheckRequirement()

    def CheckRequirement(self) -> Status:
        if self.blackboard.isLookingAt(self.WP, 0.98):
            return Status.SUCCESS 
        return Status.RUNNING 
