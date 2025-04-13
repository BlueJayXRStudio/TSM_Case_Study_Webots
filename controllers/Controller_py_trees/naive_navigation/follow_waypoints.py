import py_trees
from py_trees.composites import Sequence, Selector
from py_trees.decorators import Retry
# from custom_decorators.decorators import RepeatUntilSuccess
from naive_navigation.move_to_RL import MoveToRL

import numpy as np
from blackboard.blackboard import blackboard
from helpers.misc_helpers import *

# navigation action. "Motor execution"
class FollowWaypoints(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(FollowWaypoints, self).__init__(name)
        self.wp_set = 'mapping_waypoints'
        self.current_subtree = None
        self.index = 0

    def setup(self):
        self.logger.debug("  %s [Foo::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [Foo::initialise()]" % self.name)
        self.WP = blackboard.read(self.wp_set)
        self.index = 0

        self.getNewSubtree()
    
    def update(self):
        self.logger.debug("  %s [Foo::update()]" % self.name)

        self.current_subtree.tick_once()
        if self.current_subtree.status == py_trees.common.Status.SUCCESS:
            self.index += 1
            if self.index == len(self.WP):
                return py_trees.common.Status.SUCCESS
            print(f"creating new subtree, index: {self.index}")
            self.getNewSubtree()
        
        return py_trees.common.Status.RUNNING

    def getNewSubtree(self):
        self.current_subtree = MoveToRL(f"moving to waypoint{self.index}", [], self.WP[self.index])

    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )
