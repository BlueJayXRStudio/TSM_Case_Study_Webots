from PyInfiniteTree.Core.TaskStackMachine import Status
from PyInfiniteTree.Interfaces.Behavior import Behavior

import numpy as np
from helpers.quantizers import *
from blackboard.blackboard import blackboard
from primitive_movements.dMove import dMove
from primitive_movements.dRotate import dRotate
import scipy.special

class MoveToRL(Behavior):
    def __init__(self, blackboard, WP):
        super().__init__(blackboard)
        self.MAXSPEED = 1.5
        self.WP = WP
        self.cell_size = 0.0254 * 7
        self.angle_bins = 24
        self.actions = {
            0: self.rotate_CW,
            1: self.rotate_CCW,
            2: self.move_forward,
            3: self.move_backwards
        }

        self.distributions = {}
        self.action_chain = []

    def setup(self):
        self.logger.debug("  %s [Foo::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [Foo::initialise()]" % self.name)

    def CheckRequirement(self) -> Status:
        coord = blackboard.get_coord()

        rho = np.sqrt((coord[0] - self.WP[0])**2 + (coord[1] - self.WP[1])**2)
        if rho < 0.4:
            return Status.SUCCESS
        
        if self.check_recurrence():
            return Status.FAILURE
    
        return Status.RUNNING

    
    def Step(self, memory, blackboard, message) -> Status:
        memory.push(self)

        if self.CheckRequirement() == Status.SUCCESS:
            self.terminate()
            return Status.SUCCESS

        if message == Status.FAILURE:
            discount = 0.01
            for action in self.action_chain[::-1]:
                self.distributions[action[0]][action[1]] *= discount
                discount *= 1.1
            
            self.action_chain = []

        key = self.get_Key(blackboard.get_coord(), blackboard.get_world_pose()[2])
        
        if key not in self.distributions:
            self.distributions[key] = [0.25, 0.25, 0.25, 0.25]
        
        vec = self.distributions[key]
        vec = vec / np.sum(vec)
        self.distributions[key] = vec
        
        bias = np.array([ 
            self.rotation_advantage(360/self.angle_bins),
            self.rotation_advantage(-360/self.angle_bins),
            self.movement_advantage(self.cell_size) * 100,
            self.movement_advantage(-self.cell_size) * 100
        ])
    
        probs = np.array(self.distributions[key]) + 100 * self.min_max_scale(bias) * np.array(self.distributions[key])
        final_probs = scipy.special.softmax(probs)
        
        print(np.array(self.distributions[key]))
        print(bias)
        print(probs)
        print(final_probs)

        action_index = np.random.choice(len(final_probs), p=final_probs)
        
        # setup next action
        self.action_chain.append((key, action_index))
        print(len(self.action_chain))

        memory.push(self.actions[action_index]())
        return Status.RUNNING

    def relu(self, x):
        return np.maximum(0, x)

    def check_recurrence(self):
        recurrence_set = set()
        for key, action in self.action_chain:
            if key not in recurrence_set:
                recurrence_set.add(key)
            else:
                print("recurrence detected")
                return True
        return False

    def min_max_scale(self, x):
        x_min = np.min(x)
        x_max = np.max(x)
        return (x - x_min) / (x_max - x_min + 1e-8)

    def CheckRequirement(self):
        return Status.RUNNING

    def movement_advantage(self, dp):
        prediction = np.linalg.norm(dp * blackboard.get_heading() + blackboard.get_coord() - np.array(self.WP))
        actual = np.linalg.norm(blackboard.get_coord() - np.array(self.WP))
        return actual - prediction
    
    def rotation_advantage(self, dw):
        prediction = dw + blackboard.get_angle_to(self.WP)[1]
        actual = blackboard.get_angle_to(self.WP)[1]
        return abs(actual) - abs(prediction)

    def get_Key(self, coord, heading):
        discrete_coord = quantize_position(coord[0], coord[1], self.cell_size)
        discrete_heading = quantize_angle(heading, self.angle_bins)
        discrete_angle_to_WP = quantize_angle(blackboard.get_angle_to(self.WP)[0], self.angle_bins)
        key = (discrete_coord, discrete_heading, discrete_angle_to_WP)
        return key

    def rotate_CW(self):
        return dRotate(self.blackboard, 360/self.angle_bins, self.MAXSPEED)

    def rotate_CCW(self):
        return dRotate(self.blackboard, -360/self.angle_bins, self.MAXSPEED)

    def move_forward(self):
        return dMove(self.blackboard, self.cell_size, self.MAXSPEED)

    def move_backwards(self):
        return dMove(self.blackboard, -self.cell_size, self.MAXSPEED)
    
    def terminate(self):
        self.blackboard.leftMotor.setVelocity(0.0)
        self.blackboard.rightMotor.setVelocity(0.0)