import py_trees
import numpy as np
from blackboard.blackboard import blackboard

# navigation action. "Motor execution"
class Navigation(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(Navigation, self).__init__(name)

    def setup(self):
        self.timestep = int(blackboard.robot.getBasicTimeStep())

        self.gps = blackboard.robot.getDevice('gps')
        self.compass = blackboard.robot.getDevice('compass')
        self.leftMotor = blackboard.robot.getDevice('wheel_left_joint')
        self.rightMotor = blackboard.robot.getDevice('wheel_right_joint')
        self.marker = blackboard.robot.getFromDef("marker").getField("translation")

        self.logger.debug("  %s [Navigation::setup()]" % self.name)

    def initialise(self):
        print(self.name)
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)
        self.index = 0

        self.logger.debug("  %s [Navigation::initialise()]" % self.name)
        self.WP = blackboard.read('mapping_waypoints')

    def update(self):
        # print("navigating towards: ", self.WP[-1])

        self.logger.debug("  %s [Navigation::update()]" % self.name)

        xw = self.gps.getValues()[0]
        yw = self.gps.getValues()[1]
        theta = np.arctan2(self.compass.getValues()[0], self.compass.getValues()[1])

        rho = np.sqrt((xw - self.WP[self.index][0])**2 + (yw - self.WP[self.index][1])**2)
        alpha = np.arctan2(self.WP[self.index][1] - yw, self.WP[self.index][0] - xw) - theta

        if (alpha > np.pi):
            alpha = alpha - 2 * np.pi
        elif (alpha < -np.pi):
            alpha = alpha + 2 * np.pi

        self.marker.setSFVec3f([*self.WP[self.index], 0])

        ranges = np.array(blackboard.lidar.getRangeImage())
        ranges[ranges==np.inf] = 0.265
        ranges = ranges[80:len(ranges)-80]
        
        # reactive correction calculation
        rx = 0.0
        for i, angle in enumerate(blackboard.angles):
            if ranges[i] < 0.45:
                rx += 1/ranges[i] * np.cos(angle)
        print(rx)
        rx = 0

        vL, vR = 6.28, 6.28

        p1 = 4
        p2 = 2
        p3 = 0.1
        
        vL = -p1 * alpha + p2 * rho - rx * p3
        vR = +p1 * alpha + p2 * rho + rx * p3

        vL = min(vL, 6.28)
        vR = min(vR, 6.28)
        vL = max(vL, -6.28)
        vR = max(vR, -6.28)

        self.leftMotor.setVelocity(vL)
        self.rightMotor.setVelocity(vR)

        if (rho < 0.4):
            # print("Reached ", self.index, len(self.WP))
            self.index = self.index + 1
            if self.index == len(self.WP):
                self.feedback_message = "Last waypoint reached"
                return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )
        # redundant, but make sure that index is reset to 0
        self.index = 0
        # ensure that the robot stops after it reaches its last way point
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)