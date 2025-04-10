import py_trees
from py_trees.composites import Sequence, Selector
from py_trees.decorators import Retry
# from custom_decorators.decorators import RepeatUntilSuccess

from primitive_logic.a_looking_at import a_LookingAt
from primitive_movements.move_backwards import MoveBackwards
from primitive_movements.rotate_clockwise import RotateClockwise
from primitive_movements.rotate_counterclockwise import RotateCounterclockwise
from naive_navigation.move_to import MoveTo

import numpy as np
from blackboard.blackboard import blackboard
from helpers.misc_helpers import *

# navigation action. "Motor execution"
class FollowWaypoints(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(FollowWaypoints, self).__init__(name)
        self.wp_set = 'mapping_waypoints'
        self.current_subtree = None
        self.a_LookingAt = None
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
            print(f"creating new subtree, index: {self.index}")
            self.getNewSubtree()
            if self.index == len(self.WP):
                return py_trees.common.Status.SUCCESS
        
        return py_trees.common.Status.RUNNING

    def getNewSubtree(self):
        self.a_LookingAt = a_LookingAt("", self.WP[self.index])
        self.current_subtree = Sequence(f"Move To WP{self.index}", children=[
            Retry(f"Repeat Until Success {self.index}", 
                Selector(
                    "Looking At?", children=[
                        a_LookingAt("atomic looking at", self.WP[self.index]),
                        RotateCounterclockwise("rotating counter-clockwise", [self.a_LookingAt], 1),
                        RotateClockwise("rotating clockwise", [self.a_LookingAt], 1),
                        MoveBackwards("moving backwards", [self.a_LookingAt])
                    ], 
                    memory=True
                ), 
                num_failures=100000000),
            MoveTo(f"moving to waypoint{self.index}", self.WP[self.index])
        ], memory=True)

    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )

    