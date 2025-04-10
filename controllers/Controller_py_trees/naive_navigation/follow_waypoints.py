import py_trees
from py_trees.composites import Sequence, Selector
from custom_decorators.decorators import RepeatUntilSuccess

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
    
    def update(self):
        self.logger.debug("  %s [Foo::update()]" % self.name)

        self.current_subtree = Sequence(f"Move To WP{self.index}", children=[
            RepeatUntilSuccess(f"Repeat Until Success {self.index}", Selector(
                "Looking At?", children=[

                ], memory=True
            ))
        ], memory=True)
    
    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )

    