import py_trees
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
# Import Quad Tree structure
from QuadTree.QuadTree import Point, Rect, QuadTree
# Import blackboard as singleton
from blackboard.blackboard import blackboard
from helpers.misc_helpers import world2map, map2world

# lidar mapping for probability map and quad tree update for dynamic trajectory generation
class ReactiveMapping(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(ReactiveMapping, self).__init__(name)
        self.hasrun = False

    def setup(self):
        blackboard.write('prob_map', {})

        self.sampling_rate = 2 #16 # Period not frequency
        self.curr_sample = 0

        self.logger.debug("  %s [Mapping::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [Map::initialise()]" % self.name)
        print(self.name)
        
        # print("probability map ", blackboard.read('prob_map'))

    def update(self):
        if self.curr_sample < self.sampling_rate:
            self.curr_sample += 1
            return py_trees.common.Status.RUNNING
        else:
            self.curr_sample = 0

        self.hasrun = True
        # get GPS and compass readings
        xw = self.gps.getValues()[0]
        yw = self.gps.getValues()[1]
        theta=np.arctan2(self.compass.getValues()[0],self.compass.getValues()[1])
        
        # # DRAW line following trajectory
        # px0, py0 = world2map(xw, yw)
        # self.display.setColor(0xFF0000)
        # self.display.drawPixel(px0,py0)

        ## lidar local transform: (0.202, 0, -0.004) ##
        # transform matrix for lidar pos
        w_T_r = np.array([[np.cos(theta), -np.sin(theta), xw],
                        [np.sin(theta), np.cos(theta), yw],
                        [0, 0, 1]])
        lidarPos = w_T_r @ np.array([[0.202], [0], [1]])

        ranges = np.array(blackboard.lidar.getRangeImage())
        ranges[ranges==np.inf] = 0.265
        ranges = ranges[80:len(ranges)-80]
        
        blackboard.write('ranges', ranges)
        
        # recalculate transform matrix for lidar scans
        w_T_r = np.array([[np.cos(theta), -np.sin(theta), lidarPos[0][0]],
                        [np.sin(theta), np.cos(theta), lidarPos[1][0]],
                        [0, 0, 1]])
        X_i = np.array([ranges*np.cos(blackboard.angles), ranges*np.sin(blackboard.angles), np.ones((507,))])
        D = w_T_r @ X_i

        map = blackboard.read('prob_map')

        # maintain hashmap based probability map
        for point in D.T:
            px, py = world2map(point[0], point[1])
            map[(px,py)] = 1.0

        for point in list(map.keys()):
            map[point] -= 0.2
            if map[point] < 0.0:
                map.pop(point)

        # Construct Quad Tree
        width = 10
        height = 10
        
        domain = Rect(width/2, height/2, width, height)
        qtree = QuadTree(domain, 3)

        for point in map.keys():
            x, y = map2world(point[0], point[1])
            coord = (x+5, y+5)
            qtree.insert(Point(*coord))

        # print('Number of points in the domain =', len(qtree))
        blackboard.write('qtree', qtree)
        
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        self.logger.debug("  %s [Foo::terminate().terminate()][%s->%s]" % (self.name, self.status, new_status))

                        


