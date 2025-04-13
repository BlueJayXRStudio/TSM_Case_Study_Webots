import numpy as np
from collections import deque
from collections import defaultdict

from blackboard.blackboard import blackboard

# not sure if the rubric allows accelerometers, so we will use object recognition instead to estimate general stability
class Stability:
    def __init__(self):
        self.max_size = 10
        self.queues = defaultdict(lambda:deque(maxlen=self.max_size))
        self.last_seens = defaultdict(lambda:None)
        self.curr_vals = defaultdict(lambda:None)
        self.instability_score = 0

    def start(self):
        pass

    def update(self):
        objects=blackboard.camera.getRecognitionObjects()

        for key in list(self.curr_vals.keys()):
            self.last_seens[key] = self.curr_vals[key]
        
        for _object in objects:
            # print(_object.getModel(), _object.getId(), list(_object.getPosition()))
            self.curr_vals[_object.getId()] = np.array(list(_object.getPosition()))

        # insert new differences
        for key in list(self.curr_vals.keys()):
            if key in self.last_seens:
                curr = self.curr_vals[key]
                prev = self.last_seens[key]
                if curr is None or prev is None:
                    continue
                diff = np.linalg.norm(curr-prev)
                self.queues[key].append(diff)

        # sum up instabilities
        total_diff = 0
        for key in list(self.queues.keys()):
            total_diff += sum(list(self.queues[key]))
        
        self.instability_score = total_diff
    
    def update_single(self, pose):        
        self.last_seens['single_item_update'] = self.curr_vals['single_item_update']
        self.curr_vals['single_item_update'] = pose

        # insert new differences
        for key in list(self.curr_vals.keys()):
            if key in self.last_seens:
                curr = self.curr_vals[key]
                prev = self.last_seens[key]
                if curr is None or prev is None:
                    continue
                diff = np.linalg.norm(curr-prev)
                self.queues[key].append(diff)

        # sum up instabilities
        total_diff = 0
        for key in list(self.queues.keys()):
            total_diff += sum(list(self.queues[key]))
        
        self.instability_score = total_diff

    def terminate(self):
        pass