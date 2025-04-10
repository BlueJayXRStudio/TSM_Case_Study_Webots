import py_trees
import numpy as np
from blackboard.blackboard import blackboard
from helpers.misc_helpers import *

class a_LookingAt(py_trees.behaviour.Behaviour):
    def __init__(self, name, WP):
        super(a_LookingAt, self).__init__(name)

    def setup(self):
        self.logger.debug("  %s [Foo::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [Foo::initialise()]" % self.name)
    
    def update(self):
        self.logger.debug("  %s [Foo::update()]" % self.name)
    
    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )

