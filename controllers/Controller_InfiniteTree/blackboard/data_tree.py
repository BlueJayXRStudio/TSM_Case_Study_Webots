from PyInfiniteTree.Core.TaskStackMachine import Status
from PyInfiniteTree.Interfaces.Behavior import Behavior
import numpy as np
from blackboard.blackboard import blackboard
from helpers.misc_helpers import *

class DataTree(Behavior):
    def __init__(self, blackboard):
        super().__init__(blackboard)
        
    def Step(self, memory, blackboard, message) -> Status:
        memory.push(self)
        blackboard.update_velocity()
        blackboard.update_true_angular_velocity()
        blackboard.update_true_velocity()
        return Status.RUNNING
        
    