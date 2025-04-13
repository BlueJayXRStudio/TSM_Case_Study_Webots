from PyInfiniteTree.Core.TaskStackMachine import Status
from PyInfiniteTree.Interfaces.Behavior import Behavior
from naive_navigation.move_to_RL import MoveToRL
import numpy as np
from blackboard.blackboard import blackboard
from helpers.misc_helpers import *

# navigation action. "Motor execution"
class FollowWaypoints(Behavior):
    def __init__(self, blackboard):
        super().__init__(blackboard)
        self.wp_set = 'mapping_waypoints'
        self.current_subtree = None
        self.index = 0
        self.WP = blackboard.read(self.wp_set)

    def CheckRequirement(self) -> Status:
        return Status.RUNNING
    
    def Step(self, memory, blackboard, message) -> Status:
        memory.push(self)

        if message == Status.RUNNING:
            memory.push(MoveToRL(blackboard, self.WP[self.index]))
            return Status.RUNNING
            
        if message == Status.SUCCESS:
            self.index += 1
            if self.index == len(self.WP):
                return Status.SUCCESS
            
        memory.push(MoveToRL(blackboard, self.WP[self.index]))
        return Status.RUNNING
