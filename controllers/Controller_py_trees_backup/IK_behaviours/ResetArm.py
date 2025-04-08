import py_trees
import numpy as np

from blackboard.blackboard import blackboard

# reset arm to safe position
class ResetArm(py_trees.behaviour.Behaviour):
    def __init__(self, name, target_pose, ignore_fingers=False, max_steps = 1):
        super(ResetArm, self).__init__(name)
        self.ignore_fingers = ignore_fingers

        self.pose_index = 0
        self.max_steps = max_steps
        self.start_pose = blackboard.get_pose()
        self.target_pose = target_pose
        self.current_target_pose = blackboard.lerp(self.start_pose, self.target_pose, self.pose_index, self.max_steps, self.ignore_fingers)

        self.ahead = 0.4
        self.discount = 0.995

    def setup(self):
        # confirm that blackboard singleton is working as intended
        print("2. confirming blackboard: ", blackboard)
        self.logger.debug("  %s [ResetArm::setup()]" % self.name)

    def initialise(self):
        print(self.name)

        self.pose_index = 0
        self.start_pose = blackboard.get_pose()
        self.current_target_pose = blackboard.lerp(self.start_pose, self.target_pose, self.pose_index, self.max_steps, self.ignore_fingers)

        self.ahead = 0.2
        self.discount = 0.998
        
        self.logger.debug("  %s [ResetArm::initialise()]" % self.name)

    def update(self):
        if self.pose_index > self.max_steps:
            # print(blackboard.get_pose())
            return py_trees.common.Status.SUCCESS
        elif blackboard.get_joint_diff(self.current_target_pose, self.ignore_fingers) < self.ahead:
            self.pose_index += 1
            self.ahead *= self.discount
            if self.pose_index == self.max_steps:
                self.ahead = 0.001
            self.current_target_pose = blackboard.lerp(self.start_pose, self.target_pose, self.pose_index, self.max_steps, self.ignore_fingers)
        else:
            blackboard.set_pose(self.current_target_pose, self.ignore_fingers)

        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )

