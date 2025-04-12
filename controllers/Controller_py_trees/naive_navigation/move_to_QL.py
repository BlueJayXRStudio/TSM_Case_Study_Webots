import py_trees
import numpy as np
from helpers.quantizers import *
from blackboard.blackboard import blackboard
from primitive_movements.dMove import dMove
from primitive_movements.dRotate import dRotate
import scipy.special

class MoveToQL(py_trees.behaviour.Behaviour):
    def __init__(self, name, preconditions, WP):
        super(MoveToQL, self).__init__(name)
        self.WP = WP
        self.preconditions = preconditions
        self.cell_size = 0.0254 * 10
        self.angle_bins = 16
        self.actions = {
            0: self.rotate_CW,
            1: self.rotate_CCW,
            2: self.move_forward,
            3: self.move_backwards
        }
        
        # self.sim_step = {
        #     0: (0, 360/self.angle_bins),
        #     1: (0, -360/self.angle_bins),
        #     2: (self.cell_size, 0),
        #     3: (-self.cell_size, 0)
        # }

        self.current_subtree = None
        self.A_t = -1
        self.S_t = None
        self.state_chain = []

        self.current_Q = 0.0
        self.lr = 0.95
        self.discount = 0.95
        self.penalty = -1000

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

        S_t_1 = self.get_Key(blackboard.get_coord(), blackboard.get_world_pose()[2])
        
        bias = np.array([ 
            self.rotation_advantage(360/self.angle_bins),
            self.rotation_advantage(-360/self.angle_bins),
            self.movement_advantage(self.cell_size) * 100,
            self.movement_advantage(-self.cell_size) * 100
        ])

        Q_t_1 = np.array([ blackboard.QTable[(S_t_1, action)] for action in range(4) ])
        Q_final = Q_t_1 + 100 * self.min_max_scale(bias)
        print(Q_t_1)
        print(Q_final, "!")

        if self.current_subtree:
            self.current_subtree.tick_once()
            if self.check_recurrence():
                for i in range(len(self.state_chain)-1):
                    state, action, Q = self.state_chain[i]
                    state_1, action_1, Q_1 = self.state_chain[i+1]
                    blackboard.QTable[(state, action)] = (1-self.lr)*Q + self.lr*(self.penalty + self.discount*Q_1)
                self.state_chain = []
            
            if self.current_subtree.status == py_trees.common.Status.RUNNING:
                return py_trees.common.Status.RUNNING
            elif self.current_subtree.status == py_trees.common.Status.FAILURE: # REINFORCE
                max_next_Q = max(Q_t_1)
                blackboard.QTable[(self.S_t, self.A_t)] = (1-self.lr)*blackboard.QTable[(self.S_t, self.A_t)] + self.lr*(self.penalty + self.discount*max_next_Q)
            else:
                max_next_Q = max(Q_t_1)
                blackboard.QTable[(self.S_t, self.A_t)] = (1-self.lr)*blackboard.QTable[(self.S_t, self.A_t)] + self.lr*(self.discount*max_next_Q)
        
        probs = scipy.special.softmax(Q_final)
        action_index = np.random.choice(len(probs), p=probs)

        # setup next action
        self.S_t = S_t_1
        self.A_t = action_index
        self.current_Q = blackboard.QTable[(S_t_1, action_index)]
        self.current_subtree = self.actions[action_index]()
        self.state_chain.append((self.S_t, action_index, self.current_Q))

        return py_trees.common.Status.RUNNING

    def relu(self, x):
        return np.maximum(0, x)

    def check_recurrence(self):
        recurrence_set = set()
        for state, action, Q in self.state_chain:
            if state not in recurrence_set:
                recurrence_set.add(state)
            else:
                print("recurrence detected")
                return True
        return False

    def min_max_scale(self, x):
        x_min = np.min(x)
        x_max = np.max(x)
        return (x - x_min) / (x_max - x_min + 1e-8)

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

    # def predict_coord(self, dp):
    #     return dp * blackboard.get_heading() + blackboard.get_coord()

    # def predict_heading(self, dw):
    #     return self.rotate_vector(blackboard.get_heading(), dw)
    
    def get_Key(self, coord, heading):
        discrete_coord = quantize_position(coord[0], coord[1], self.cell_size)
        discrete_heading = quantize_angle(heading, self.angle_bins)
        discrete_angle_to_WP = quantize_angle(blackboard.get_angle_to(self.WP)[0], self.angle_bins)
        key = (discrete_coord, discrete_heading, discrete_angle_to_WP)
        return key

    # # needs review
    # def rotate_vector(vec, angle_rad):
    #     x, y = vec
    #     cos_a = np.cos(-angle_rad)
    #     sin_a = np.sin(-angle_rad)
    #     x_new = x * cos_a - y * sin_a
    #     y_new = x * sin_a + y * cos_a
    #     return (x_new, y_new)
    
    def rotate_CW(self):
        return dRotate("rotate cw", [self], 360/self.angle_bins, 1)

    def rotate_CCW(self):
        return dRotate("rotate ccw", [self], -360/self.angle_bins, 1)

    def move_forward(self):
        return dMove("move forward", [self], self.cell_size, 1)

    def move_backwards(self):
        return dMove("move backwards", [self], -self.cell_size, 1)
    
    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )

        blackboard.leftMotor.setVelocity(0.0)
        blackboard.rightMotor.setVelocity(0.0)