import py_trees
import numpy as np
from helpers.quantizers import *
from blackboard.blackboard import blackboard
from primitive_movements.dMove import dMove
from primitive_movements.dRotate import dRotate
import scipy.special


class MoveToRL(py_trees.behaviour.Behaviour):
    def __init__(self, name, preconditions, WP):
        super(MoveToRL, self).__init__(name)
        self.WP = WP
        self.preconditions = preconditions
        self.cell_size = 0.0254 * 5
        self.angle_bins = 64
        self.actions = {
            0: self.rotate_CW,
            1: self.rotate_CCW,
            2: self.move_forward,
            3: self.move_backwards
        }
        self.distributions = {}
        self.current_subtree = None
        self.current_action = -1
        self.current_key = None

    def setup(self):
        self.logger.debug("  %s [Foo::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [Foo::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [Foo::update()]" % self.name)

        coord = blackboard.get_coord()

        rho = np.sqrt((coord[0] - self.WP[0])**2 + (coord[1] - self.WP[1])**2)
        if rho < 0.4:
            return py_trees.common.Status.SUCCESS

        if self.current_subtree:
            self.current_subtree.tick_once()
            if self.current_subtree.status == py_trees.common.Status.RUNNING:
                return py_trees.common.Status.RUNNING
            elif self.current_subtree.status == py_trees.common.Status.FAILURE: # REINFORCE
                self.distributions[self.current_key][self.current_action] -= 0.9
                print(self.current_key)

        current_coord = blackboard.get_coord()
        discrete_coord = quantize_position(current_coord[0], current_coord[1], self.cell_size)
        discrete_heading = quantize_angle(blackboard.get_world_pose()[2], self.angle_bins)
        discrete_angle_to_WP = quantize_angle(blackboard.get_angle_to(self.WP)[0], self.angle_bins)
        key = (discrete_coord, discrete_heading, discrete_angle_to_WP)
        
        if key not in self.distributions:
            self.distributions[key] = [0.25, 0.25, 0.25, 0.25]

        # print(blackboard.get_coord(), self.movement_advantage(self.cell_size), self.movement_advantage(-self.cell_size))
        # print(blackboard.get_angle_to(self.WP)[1], self.rotation_advantage(360/self.angle_bins), self.rotation_advantage(-360/self.angle_bins))
        
        bias = np.array([ 
            self.rotation_advantage(360/self.angle_bins),
            self.rotation_advantage(-360/self.angle_bins),
            self.movement_advantage(self.cell_size) * 20,
            self.movement_advantage(-self.cell_size) * 20
        ])
    
        # print(np.log(self.distributions[key]))

        adjusted_probs = self.distributions[key] * bias * 3
        probs = scipy.special.softmax(adjusted_probs)

        print(self.distributions[key], adjusted_probs, probs)
        self.distributions[key]

        # print(self.distributions)
        action_index = np.random.choice(len(probs), p=probs)
        
        # setup next action
        self.current_key = key
        self.current_action = action_index
        self.current_subtree = self.actions[action_index]()

        return py_trees.common.Status.RUNNING

    def CheckRequirement(self):
        return py_trees.common.Status.RUNNING

    def movement_advantage(self, dp):
        prediction = np.linalg.norm(dp * blackboard.get_heading() + blackboard.get_coord() - np.array(self.WP))
        actual = np.linalg.norm(blackboard.get_coord() - np.array(self.WP))
        return actual - prediction
    
    def rotation_advantage(self, dw):
        prediction = dw + blackboard.get_angle_to(self.WP)[1]
        actual = blackboard.get_angle_to(self.WP)[1]
        return abs(actual) - abs(prediction)

    def predict_coord(self, dp):
        return dp * blackboard.get_heading() + blackboard.get_coord()

    def predict_heading(self, dw):
        return self.rotate_vector(blackboard.get_heading(), dw)
    
    # needs review
    def rotate_vector(vec, angle_rad):
        x, y = vec
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a
        return (x_new, y_new)
    
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

        blackboard.leftMotor.setVelocity(0.0)
        blackboard.rightMotor.setVelocity(0.0)