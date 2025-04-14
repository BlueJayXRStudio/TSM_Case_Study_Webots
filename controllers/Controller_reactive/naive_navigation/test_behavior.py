import py_trees
import numpy as np
from blackboard.blackboard import blackboard

class TestBehavior(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(TestBehavior, self).__init__(name)
        self.initial_heading = blackboard.get_heading()

    def setup(self):
        self.logger.debug("  %s [Foo::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [Foo::initialise()]" % self.name)
        self.initial_heading = blackboard.get_heading()
    
    def update(self):
        self.logger.debug("  %s [Foo::update()]" % self.name)
        print(blackboard.get_heading(), blackboard.get_angle_to((0,0)), blackboard.get_angle_from_to(blackboard.get_heading(), self.initial_heading))
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )

