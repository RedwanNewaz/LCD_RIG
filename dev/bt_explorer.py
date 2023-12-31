import py_trees
import py_trees.console as console
from .quad_geom import Point

class Explorer(py_trees.behaviour.Behaviour):
    def __init__(self, robot, explorationTree, pub, name):
        self.robot = robot
        self.pub = pub
        self.explorationTree = explorationTree
        super().__init__(f"{name}/explorer")
        self.parent_name = name
        self.dt = 0.001

    def update(self):
        if self.robot.has_goal:
            self.robot.update(*self.robot.control())
            state = self.robot.state
            control = self.robot.control_input
            self.pub.set("/%s/state" % self.parent_name, f"{state[0]},{state[1]},{state[2]},{control[0]},{control[1]}")
            p = Point(state[0], state[1])
            self.explorationTree.insert(p)
            return self.status.RUNNING
        return self.status.SUCCESS
