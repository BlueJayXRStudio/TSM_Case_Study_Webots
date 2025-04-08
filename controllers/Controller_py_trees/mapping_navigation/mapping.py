import py_trees
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from blackboard.blackboard import blackboard
from helpers.misc_helpers import world2map, map2world

# lidar mapping for trajectory generation 
class Mapping(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(Mapping, self).__init__(name)
        self.hasrun = False

    def setup(self):
        self.timestep = int(blackboard.robot.getBasicTimeStep())

        self.gps = blackboard.robot.getDevice('gps')
        self.compass = blackboard.robot.getDevice('compass')
        self.lidar = blackboard.robot.getDevice('Hokuyo URG-04LX-UG01')
        self.display = blackboard.robot.getDevice('display')

        self.logger.debug("  %s [Mapping::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [Map::initialise()]" % self.name)
        print(self.name)
        self.map = np.zeros((200, 300))
        self.angles = np.linspace(4.19 / 2, -4.19 / 2, 667)
        self.angles = self.angles[80:len(self.angles)-80]

    def update(self):
        self.hasrun = True
        # get GPS and compass readings
        xw = self.gps.getValues()[0]
        yw = self.gps.getValues()[1]
        theta=np.arctan2(self.compass.getValues()[0],self.compass.getValues()[1])
        
        # DRAW line following trajectory
        px, py = world2map(xw, yw)
        self.display.setColor(0xFF0000)
        self.display.drawPixel(px,py)

        ## lidar local transform: (0.202, 0, -0.004) ##
        # transform matrix for lidar pos
        w_T_r = np.array([[np.cos(theta), -np.sin(theta), xw],
                        [np.sin(theta), np.cos(theta), yw],
                        [0, 0, 1]])
        lidarPos = w_T_r @ np.array([[0.202], [0], [1]])

        ranges = np.array(self.lidar.getRangeImage())
        ranges[ranges==np.inf] = 100
        ranges = ranges[80:len(ranges)-80]
        
        # recalculate transform matrix for lidar scans
        w_T_r = np.array([[np.cos(theta), -np.sin(theta), lidarPos[0][0]],
                        [np.sin(theta), np.cos(theta), lidarPos[1][0]],
                        [0, 0, 1]])
        X_i = np.array([ranges*np.cos(self.angles), ranges*np.sin(self.angles), np.ones((507,))])
        D = w_T_r @ X_i

        # UPDATE GRID AND DRAW. Exclude first and last 80 values.
        for point in D.T:
            px, py = world2map(point[0], point[1])
            self.map[px, py] += 0.001

            v=int(min(self.map[px,py], 1.0)*255)
            color = (v*256**2+v*256+v)
            self.display.setColor(color)
            self.display.drawPixel(px,py)

        return py_trees.common.Status.RUNNING
    
    # ensure that map is saved on disk and populated in blackboard to be used by planner  
    def terminate(self, new_status):
        self.logger.debug("  %s [Foo::terminate().terminate()][%s->%s]" % (self.name, self.status, new_status))
        if (self.hasrun):
            cmap = signal.convolve2d(self.map, np.ones((30,30)), mode='same')
            cspace = cmap > 0.9

            np.save('cspace', cspace)
            
            cspace = np.load('cspace.npy')
            plt.figure(0)
            plt.imshow(cspace)
            plt.show()

            cspace = np.load('cspace.npy')
            cspace[ cspace > 0 ] = 255
            blackboard.write('cspace', cspace)
            
            for px in range(len(cspace)):
                for py in range(len(cspace[0])):
                    if cspace[px][py] > 0:
                        self.display.setColor(0xFFFFFF)
                        self.display.drawPixel(px,py)
                        


