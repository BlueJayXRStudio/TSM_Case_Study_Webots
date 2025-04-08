import py_trees
import numpy as np

from blackboard.blackboard import blackboard

# reset arm to safe position
class ClosePositionGap(py_trees.behaviour.Behaviour):
    def __init__(self, name, position):
        super(ClosePositionGap, self).__init__(name)
        self.position = position

    def setup(self):
        self.logger.debug("  %s [ClosePositionGap::setup()]" % self.name)

    def initialise(self):
        print(self.name)

        self.MAXSPEED = 3.14
        
        self.logger.debug("  %s [ClosePositionGap::initialise()]" % self.name)

    def update(self):
        xw = blackboard.gps.getValues()[0]
        yw = blackboard.gps.getValues()[1]
        theta = np.arctan2(blackboard.compass.getValues()[0], blackboard.compass.getValues()[1])

        rho = np.sqrt((xw - self.position[0])**2 + (yw - self.position[1])**2)
        alpha = np.arctan2(self.position[1] - yw, self.position[0] - xw) - theta

        if (alpha > np.pi):
            alpha = alpha - 2 * np.pi
        elif (alpha < -np.pi):
            alpha = alpha + 2 * np.pi

        vL, vR = self.MAXSPEED, self.MAXSPEED

        p1 = 4
        p2 = 2
        
        vL = -p1 * alpha + p2 * rho
        vR = +p1 * alpha + p2 * rho

        vL = min(vL, self.MAXSPEED)
        vR = min(vR, self.MAXSPEED)
        vL = max(vL, -self.MAXSPEED)
        vR = max(vR, -self.MAXSPEED)

        blackboard.leftMotor.setVelocity(vL)
        blackboard.rightMotor.setVelocity(vR)
        
        if np.linalg.norm(np.array(self.position) - np.array((xw, yw))) < 0.005:
            blackboard.leftMotor.setVelocity(0)
            blackboard.rightMotor.setVelocity(0)
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )

