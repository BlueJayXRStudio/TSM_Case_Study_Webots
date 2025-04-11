import py_trees
import numpy as np
from helpers.quantizers import *
from blackboard.blackboard import blackboard

class MoveToRL(py_trees.behaviour.Behaviour):
    def __init__(self, name, preconditions, WP):
        super(MoveToRL, self).__init__(name)
        self.WP = WP
        self.preconditions = preconditions
        self.actions = {}
        self.distributions = {}

    def setup(self):
        self.logger.debug("  %s [Foo::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [Foo::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [Foo::update()]" % self.name)

        current_coord = blackboard.get_coord()
        discrete_coord = quantize_position(current_coord[0], current_coord[1], 0.0254)
        discrete_heading = quantize_angle(blackboard.get_world_pose()[2], 64)
        discrete_angle_to_WP = quantize_angle(blackboard.get_angle_to(self.WP)[0], 64)
        key = (discrete_coord, discrete_heading, discrete_angle_to_WP)

        if key not in self.distributions:
            self.distributions[key] = [0.25, 0.25, 0.25, 0.25]

        return py_trees.common.Status.RUNNING

    def rotate_CW():
        pass

    def rotate_CCW():
        pass

    def move_forward():
        pass

    def move_backwards():
        pass

    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )

