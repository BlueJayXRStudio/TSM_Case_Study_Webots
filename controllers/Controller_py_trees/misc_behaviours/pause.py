import py_trees
import numpy as np

from blackboard.blackboard import blackboard

# reset arm to safe position
class Pause(py_trees.behaviour.Behaviour):
    def __init__(self, name, pause_time):
        super(Pause, self).__init__(name)
        self.timer = 0.0
        self.pause_time = pause_time

    def setup(self):
        self.logger.debug("  %s [Pause::setup()]" % self.name)

    def initialise(self):
        print(self.name)

        self.timer = 0.0
        
        self.logger.debug("  %s [Pause::initialise()]" % self.name)

    def update(self):
        self.timer += blackboard.delta_t

        if self.timer > self.pause_time:
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )

