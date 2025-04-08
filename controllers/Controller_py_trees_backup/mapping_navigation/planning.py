import py_trees
import numpy as np
from matplotlib import pyplot as plt

from heapq import heapify, heappush, heappop
from collections import defaultdict

import time

from QuadTree.QuadTree import Point, Rect, QuadTree
from blackboard.blackboard import blackboard
from helpers.misc_helpers import world2map, map2world


def getNeighborsTiered(map, radius, u, goal, tier):
    neighbors = []
    # algorithmic map compression
    if np.linalg.norm(np.array(u)-np.array(goal)) < np.sqrt((tier**2) * 2):
        return [(np.linalg.norm(np.array(u)-np.array(goal)), goal)]

    for delta in ((0, tier), (0, -tier), (tier, 0), (-tier, 0), (tier, tier), (-tier, -tier), (tier, -tier), (-tier, tier)):
        candidate = (u[0] + delta[0], u[1] + delta[1])
        
        if (candidate[0] >= 0 and candidate[0] < len(map) and 
            candidate[1] >= 0 and candidate[1] < len(map[0]) and 
            map[candidate[0]][candidate[1]] == 0):
            # neighbors.append((1/np.sqrt(delta[0]**2+delta[1]**2), candidate))
            neighbors.append((np.sqrt(delta[0]**2+delta[1]**2), candidate))
    return neighbors

# path planning action. Incorporates A*
class Planning(py_trees.behaviour.Behaviour):
    def __init__(self, name, dest):
        super(Planning, self).__init__(name)
        self.robot_radius = 0.38 # 0.265 with a bit of padding
        self.robot_radius_alt = 0.38
        px, py = world2map(dest[0], dest[1])
        self.world_goal = dest
        self.goal = (px, py)

    def setup(self):
        self.timestep = int(blackboard.robot.getBasicTimeStep())
        
        self.display = blackboard.robot.getDevice('display')
        self.gps = blackboard.robot.getDevice('gps')
        self.compass = blackboard.robot.getDevice('compass')
        self.lidar = blackboard.robot.getDevice('Hokuyo URG-04LX-UG01')

        self.tier = 5

        self.rest_timer = 1.0

        self.logger.debug("  %s [Planning::setup()]" % self.name)

    def initialise(self):
        print(self.name)
        self.tier = 5
        self.rest_timer = 1.0
        # print("planning towards: ", self.world_goal)

    def update(self):
        if self.rest_timer > 0.0:
            self.rest_timer -= blackboard.delta_t
            return py_trees.common.Status.RUNNING
        
        if 'qtree' not in blackboard.get_keys():
            return py_trees.common.Status.RUNNING
        
        # start timer
        t0 = time.perf_counter()

        cspace = blackboard.read('cspace')
        
        xw = self.gps.getValues()[0]
        yw = self.gps.getValues()[1]
        theta=np.arctan2(self.compass.getValues()[0],self.compass.getValues()[1])

        px, py = world2map(xw, yw)
        start = (px, py)
        path = []

        # print(cspace[px][py], "space debug")
        # reactive movement will sometimes force the robot into technically unnavigable spaces.
        # we will have to project to the nearest navigable point and designate that as the starting point.
        # this is a temporary solution
        if cspace[px][py]:
            print ("planning error handling")
            upper = yw + self.robot_radius_alt
            lower = yw - self.robot_radius_alt
            right = xw + self.robot_radius_alt
            left = xw - self.robot_radius_alt 
            x1, y1 = world2map(left, upper)
            x2, y2 = world2map(right, lower)
            inner_left,  inner_upper= world2map(xw - self.robot_radius_alt, yw + self.robot_radius_alt)
            inner_right,  inner_upper= world2map(xw + self.robot_radius_alt, yw + self.robot_radius_alt)
            inner_left,  inner_lower= world2map(xw - self.robot_radius_alt, yw + self.robot_radius_alt)
            inner_right,  inner_lower= world2map(xw - self.robot_radius_alt, yw + self.robot_radius_alt)

            min_dist = float('inf')
            new_start = None
            for i in range(x1, x2+1):
                for j in range(y1, y2+1):
                    if i >= inner_left and i <= inner_right and j >= inner_lower and j <= inner_upper:
                        continue
                    if cspace[i][j] == 0:
                        dist = np.linalg.norm(np.array((i, j)) - np.array((px, py)))
                        if dist < min_dist:
                            min_dist = dist
                            new_start = (i, j)
            start = new_start

            if start == None:
                # print(start, new_start, cspace[new_start[0]][new_start[0]])
                # print(start, self.goal, blackboard.read('planned_waypoints'))
                raise Exception("obscure projection error")

        if start == None:
            print(start, self.goal, blackboard.read('planned_waypoints'))
            raise Exception("obscure projection error")
        
        qtree = blackboard.read('qtree')

        while len(path) <= 1 and self.tier > 0: 
            # RUN A* Pathfinding Algorithm        
            queue = [(0, start)]
            heapify(queue)

            distances = defaultdict(lambda:float('inf'))
            distances[start] = 0

            visited = set()
            parent = {}

            while queue:
                (priority, u) = heappop(queue)

                if u in visited:
                    continue

                visited.add(u)
                
                # run quadtree query during node expansion to prevent redundant queries
                found_points = []
                x, y = map2world(u[0], u[1])
                qtree.query_radius((x+5, y+5), self.robot_radius, found_points)
                if len(found_points) > 0:
                    continue

                if u == self.goal:
                    break
                
                for (costuv, v) in getNeighborsTiered(blackboard.read('cspace'), self.robot_radius, u, self.goal, self.tier):
                    if v not in visited:
                        newcost = distances[u] + costuv
                        if newcost < distances[v]:
                            distances[v] = newcost
                            heappush(queue, (newcost + np.sqrt((self.goal[0]-v[0])**2+(self.goal[1]-v[1])**2),v))
                            parent[v] = u

            path = []
            
            key = self.goal
            while key in parent.keys():
                key = parent[key]
                path.insert(0, key)

            path.append(self.goal)

            # # reset previous path on display
            # prevWP = blackboard.read('planned_waypoints')
            # if prevWP:
            #     for (x, y) in prevWP:
            #         px, py = world2map(x, y)
            #         self.display.setColor(0x000000)
            #         self.display.drawPixel(px, py)
            
            reset_color = (0, 0, 0)
            self.display.setColor(reset_color[0] << 16 | reset_color[1] << 8 | reset_color[2])
            self.display.fillRectangle(0, 0, 200, 300)

            for px in range(len(cspace)):
                for py in range(len(cspace[0])):
                    if cspace[px][py] > 0:
                        self.display.setColor(0xFFFFFF)
                        self.display.drawPixel(px,py)

            # DRAW PATH on robot display
            for (x, y) in path:
                self.display.setColor(0x00FFFF)
                self.display.drawPixel(x,y)

            # convert path map coords to world coords
            converted_path = []
            for (px, py) in path:
                x, y = map2world(px, py)
                converted_path.append((x, y))
            path = converted_path

            self.tier -= 1


        # populate path into blackboard as way points
        blackboard.write('planned_waypoints', path)
        print(f"time taken to calculate path: {time.perf_counter() - t0}")
        return py_trees.common.Status.SUCCESS