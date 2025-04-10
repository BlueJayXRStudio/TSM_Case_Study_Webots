import py_trees
import numpy as np
from blackboard.blackboard import blackboard
from helpers.misc_helpers import world2map, map2world

# precise navigation with dynamic re-pathing, PID control, and reactive corrections.
class FineNavigation(py_trees.behaviour.Behaviour):
    def __init__(self, name, mapping):
        super(FineNavigation, self).__init__(name)
        self.mapping = mapping
        self.wp_set = 'planned_waypoints'

        # PID Params
        self.KP = 0.0
        self.KI = 0.0
        self.KD = 0.0
        self.previous_error = 0.0
        self.integral = 0.0

    def setup(self):
        self.timestep = int(blackboard.robot.getBasicTimeStep())

        self.gps = blackboard.robot.getDevice('gps')
        self.compass = blackboard.robot.getDevice('compass')
        self.lidar = blackboard.robot.getDevice('Hokuyo URG-04LX-UG01')
        self.leftMotor = blackboard.robot.getDevice('wheel_left_joint')
        self.rightMotor = blackboard.robot.getDevice('wheel_right_joint')
        self.robot_radius = 0.30 # 0.265

        self.angles = np.linspace(4.19 / 2 + np.pi/2, -4.19 / 2 + np.pi/2, 667)
        self.angles = self.angles[80:len(self.angles)-80]

        blackboard.write('skip_planning', 0.0)

        self.marker = blackboard.robot.getFromDef("marker").getField("translation")

        self.logger.debug("  %s [FineNavigation::setup()]" % self.name)

    def initialise(self):
        print(self.name)
        
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)

        self.vL, self.vR = 0.0, 0.0

        self.index = 0
        self.WP = blackboard.read(self.wp_set)

        self.reset_PID()

        self.logger.debug("  %s [FineNavigation::initialise()]" % self.name)
        
    
    def update(self):
        # print("navigating towards: ", self.WP[self.index], (x, y), "index =", self.index, "length of waypoints:", len(self.WP))
        
        self.logger.debug("  %s [FineNavigation::update()]" % self.name)

        self.marker.setSFVec3f([*self.WP[self.index], 0])

        qtree = blackboard.read('qtree')

        step = len(self.WP) // 10
        step = max(1, step)
        sampled_waypoints = self.WP[::step] 

        ############ YOU CAN (UN)COMMENT BELOW FOR DYNAMIC PATH PLANNING #############
        for point in sampled_waypoints:
            found_points = []
            qtree.query_radius((point[0]+5, point[1]+5), self.robot_radius, found_points)
            if len(found_points) > 0:
                print("detected moving obstacle. replanning")
                return py_trees.common.Status.FAILURE
        ##############################################################################

        xw = self.gps.getValues()[0]
        yw = self.gps.getValues()[1]
        
        rho = np.sqrt((xw - self.WP[self.index][0])**2 + (yw - self.WP[self.index][1])**2)

        # reactive correction calculation
        p1 = 0.005 # ADJUST THIS FOR STEERING INFLUENCE
        p2 = 0.01
        rx, ry = 0.0, 0.0
        for i, angle in enumerate(self.angles):
            if blackboard.read('ranges')[i] < 0.45:
                rx += 1/blackboard.read('ranges')[i] * np.cos(angle)
                ry += 1/blackboard.read('ranges')[i] * np.sin(angle)  

        # print(f"rx: {rx:.3f}, ry: {ry:.3f}")

        if len(self.WP) == 1 and np.linalg.norm(np.array((xw, yw)) - np.array((self.WP[-1][0], self.WP[-1][1]))) > 1.0:
            print("No path available. Please remove obstacles.")
            return py_trees.common.Status.FAILURE
        
        steering_adjustment = self.compute_pid_control((xw, yw, 0), (self.compass.getValues()[1],self.compass.getValues()[0], 0), self.WP[self.index], blackboard.delta_t)

        # base_speed = 0.8 * blackboard.MAXSPEED
        base_speed = max(0.1 * blackboard.MAXSPEED, 0.8 * blackboard.MAXSPEED - p2*ry)
        self.vL = max(min(base_speed - steering_adjustment*5.0 - rx * p1, blackboard.MAXSPEED), -blackboard.MAXSPEED)
        self.vR = max(min(base_speed + steering_adjustment*5.0 + rx * p1, blackboard.MAXSPEED), -blackboard.MAXSPEED)

        self.leftMotor.setVelocity(self.vL)
        self.rightMotor.setVelocity(self.vR)

        if self.index < len(self.WP)-1 and rho < 0.4:
            # print("Reached ", self.index, len(self.WP))
            self.index = self.index + 1
            # self.reset_PID()
        elif self.index == len(self.WP)-1 and rho < 0.4:
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
    
    def reset_PID(self):
        self.KP = 15.0
        self.KI = 0.4
        self.KD = 20.0
        self.previous_error = 0
        self.integral = 0

    def compute_cross_product_error(self, current_pos, current_heading, waypoint):
        # compute vector from robot to waypoint
        vector_to_waypoint = np.array((waypoint[0] - current_pos[0], waypoint[1] - current_pos[1], 0))
        
        # normalize vectors
        mag_vtw = np.linalg.norm(vector_to_waypoint)
        mag_ch = np.linalg.norm(current_heading)
        
        # avoid division by zero
        if mag_vtw == 0 or mag_ch == 0:
            return 0  

        vector_to_waypoint = vector_to_waypoint / mag_vtw
        current_heading = current_heading / mag_ch
        
        # print(current_heading, vector_to_waypoint)
        # compute cross product
        cross_product = np.cross(current_heading, vector_to_waypoint)
        # print("cross product: ", cross_product)
        # positive: turn left; negative: turn right
        # print(cross_product[-1])
        return cross_product[-1]

    def compute_pid_control(self, curr_pos, current_heading, wp, dt):        
        error = self.compute_cross_product_error(curr_pos, current_heading, wp)
        
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.KP * error + self.KI * self.integral + self.KD * derivative
        
        self.previous_error = error
        return output