from controller import Supervisor
from helpers.misc_helpers import *

robot = Supervisor()
# robot.getFromDef("marker").getField("translation")

manual_paths = [
    (False, [0, 1, 2]),
    (True, [2, 3, 4, 5, 6]),
    (False, [2, 7]),
    (False, [7, 8]),
    (True, [8, 9]),
    (True, [8, 10])
]

waypoints = []
for backtrack, path in manual_paths:
    for i in path:
            waypoints.append(robot.getFromDef(f"WP{i}").getField("translation").getSFVec3f())
    if backtrack:
        for i in path[::-1]:
            waypoints.append(robot.getFromDef(f"WP{i}").getField("translation").getSFVec3f())

print([(i[0], i[1]) for i in waypoints])