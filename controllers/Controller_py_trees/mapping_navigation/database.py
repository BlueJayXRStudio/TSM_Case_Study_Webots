from os.path import exists
import py_trees
import numpy as np
from blackboard.blackboard import blackboard

# Action for existing map quick-check
class DoesMapExist(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(DoesMapExist, self).__init__(name)
    
    def setup(self):
        self.display = blackboard.robot.getDevice('display')
        self.logger.debug("  %s [Mapping::setup()]" % self.name)

    def update(self):
        if exists('cspace.npy'):
            print("Map already exists")

            # if map already exists, populate map into blackboard and also display it on the robot's display
            cspace = np.load('cspace.npy')
            cspace = cspace.astype(int)
            cspace[ cspace > 0 ] = 255
            blackboard.write('cspace', cspace)

            for px in range(len(cspace)):
                for py in range(len(cspace[0])):
                    if cspace[px][py] > 0:
                        self.display.setColor(0xFFFFFF)
                        self.display.drawPixel(px,py)

            return py_trees.common.Status.SUCCESS
        else:
            print("Map does not exist")
            return py_trees.common.Status.FAILURE