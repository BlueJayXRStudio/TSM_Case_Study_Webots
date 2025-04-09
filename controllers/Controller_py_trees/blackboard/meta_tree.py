import py_trees
import numpy as np
from blackboard.blackboard import blackboard
from helpers.misc_helpers import *

# precise navigation with reactive corrections
class MetaTree(py_trees.behaviour.Behaviour):
    def __init__(self, name, preconditions, max_speed=0.5):
        super(MetaTree, self).__init__(name)

    def setup(self):
        self.logger.debug("  %s [LookAt::setup()]" % self.name)

    def initialise(self):
        print(self.name)
        self.logger.debug("  %s [LookAt::initialise()]" % self.name)
        
    def update(self):        

        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )
        
        
    