import py_trees
import numpy as np
from helpers.quantizers import *
from blackboard.blackboard import blackboard
from primitive_movements.dMove import dMove
from primitive_movements.dRotate import dRotate

class MoveToRL(py_trees.behaviour.Behaviour):
    def __init__(self, name, preconditions, WP):
        super(MoveToRL, self).__init__(name)
        self.WP = WP
        self.preconditions = preconditions
        self.cell_size = 0.0254
        self.angle_bins = 64
        self.actions = {
            0: self.rotate_CW,
            1: self.rotate_CCW,
            2: self.move_forward,
            3: self.move_backwards
        }
        self.distributions = {}
        self.current_subtree = None

    def setup(self):
        self.logger.debug("  %s [Foo::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [Foo::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [Foo::update()]" % self.name)

        if self.current_subtree:
            self.current_subtree.tick_once()
            if self.current_subtree.status == py_trees.common.Status.RUNNING:
                return py_trees.common.Status.RUNNING
            else:
                print(self.current_subtree.status)

        current_coord = blackboard.get_coord()
        discrete_coord = quantize_position(current_coord[0], current_coord[1], self.cell_size)
        discrete_heading = quantize_angle(blackboard.get_world_pose()[2], self.angle_bins)
        discrete_angle_to_WP = quantize_angle(blackboard.get_angle_to(self.WP)[0], self.angle_bins)
        key = (discrete_coord, discrete_heading, discrete_angle_to_WP)

        if key not in self.distributions:
            self.distributions[key] = [0.25, 0.25, 0.25, 0.25]

        # print(self.distributions)
        action_index = np.random.choice(len(self.distributions[key]), p=self.distributions[key])

        self.current_subtree = self.actions[action_index]()

        return py_trees.common.Status.RUNNING

    def CheckRequirement(self):
        return py_trees.common.Status.RUNNING

    def predict_pose(self):
        pass

    def predict_heading(self):
        pass
    
    def rotate_CW(self):
        return dRotate("rotate cw", [self], 360/self.angle_bins, 2)

    def rotate_CCW(self):
        return dRotate("rotate ccw", [self], -360/self.angle_bins, 2)

    def move_forward(self):
        return dMove("move forward", [self], self.cell_size, 2)

    def move_backwards(self):
        return dMove("move backwards", [self], -self.cell_size, 2)
    
    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )

