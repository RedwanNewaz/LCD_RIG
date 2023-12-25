import numpy as np
import py_trees
import py_trees.console as console
import quads
from py_trees import common
from .bt_viz import Visualization



class CollisionChecker(py_trees.behaviour.Behaviour):
    def __init__(self, x, y, tree, robot_radius, name):
        self.x = x
        self.y = y
        self.tree = tree
        self.robot_radius = robot_radius
        self.__range = 2
        super(CollisionChecker, self).__init__(f"{name}/collision_checker")

    def check_nei_range(self, x, y, tree, range):

        bb = quads.BoundingBox(min_x=x-range, min_y=y-range, max_x=x+range, max_y=y+range)
        for _ in tree.within_bb(bb):
            return True
        return False
    def update(self):
        status = self.check_nei_range(self.x, self.y, self.tree, self.__range * self.robot_radius)
        if status:
            msg = f"[{self.name}] : collide"
            console.info(console.red + msg + console.reset)

        return self.status.SUCCESS if status else self.status.FAILURE


class Communicator(py_trees.behaviour.Behaviour):
    def __init__(self, x, y, tree, robot_radius, name):
        self.x = x
        self.y = y
        self.tree = tree
        self.robot_radius = robot_radius
        self.__range = 4
        super(Communicator, self).__init__(f"{name}/communicator")

    def get_neighbors(self, x, y, tree, range):
        bb = quads.BoundingBox(min_x=x - range, min_y=y - range, max_x=x + range, max_y=y + range)
        agents = []
        for point in tree.within_bb(bb):
            agents.append(point.data)
        return agents

    def update(self):
        for k, nei in enumerate(self.get_neighbors(self.x, self.y, self.tree, self.__range * self.robot_radius)):
            console.info(console.green + f"[{self.name}]: -> ({k + 1}) {nei}" + console.reset)
        return self.status.SUCCESS


class Explorer(py_trees.behaviour.Behaviour):
    def __init__(self, x_new, rng, model, evaluator, sensor, pub, name):
        self.x_new = x_new
        self.rng = rng
        self.model = model
        self.evaluator = evaluator
        self.sensor = sensor
        self.pub = pub
        self.parent_name = name
        super(Explorer, self).__init__(f"{name}/explorer")


    def update(self) -> common.Status:
        y_new = self.sensor.sense(self.x_new, self.rng).reshape(-1, 1)
        self.model.add_data(self.x_new, y_new)
        self.model.optimize(num_iter=len(y_new), verbose=False)
        mean, std, error = self.evaluator.eval_prediction(self.model)
        msg = f"[{self.name}]:  gp = {np.mean(mean):.3f} +/- {np.mean(std):.3f} | err {np.mean(error):.3f}"
        self.logger.debug(msg)
        self.pub.set("/%s/xyz" % self.parent_name, f"{self.x_new[0, 0]},{self.x_new[0, 1]},{y_new[0, 0]}")

        return self.status.SUCCESS


class Agent(py_trees.behaviour.Behaviour):
    def __init__(self,rng, model, strategy, sensor, evaluator, robotID):

        self.rng = rng
        self.model = model
        self.strategy = strategy
        self.sensor = sensor
        self.evaluator = evaluator
        self.robotID = robotID
        self.stepCount = 0
        self.robot_radius = 0.5
        name = "Agent%03d" % self.robotID
        super().__init__(name)

        self.pub = py_trees.blackboard.Client(name=name, namespace=name)
        self.pub.register_key(key="/%s/xyz" % self.name , access=py_trees.common.Access.WRITE)

        self.explorationTree = quads.QuadTree( (0, 0), 22, 22, capacity=4)

    def find_neighbors(self):
        tree = quads.QuadTree(
            (0, 0),
            40, 40
         )
        state = Visualization.decode()
        for robotName, val in state.items():
            for key, robotState in val.items():
                if isinstance(robotState, list) and key == 'xyz':
                    point = quads.Point(robotState[0], robotState[1], data=robotName)
                    tree.insert(point, data=robotName)
        return tree


    def update(self):

        self.strategy(exploration_tree=self.explorationTree)
        x_new = self.strategy.get(model=self.model)

        tree = self.find_neighbors()
        x, y = x_new[0, 0], x_new[0, 1]
        collision_checker = CollisionChecker(x, y, tree, self.robot_radius, self.name)
        communicator = Communicator(x, y, tree, self.robot_radius, self.name)
        explorer = Explorer(x_new, self.rng, self.model, self.evaluator, self.sensor, self.pub, self.name)

        root = py_trees.composites.Sequence(name="Sequence", memory=True)
        # selector = py_trees.composites.Selector(name="Selector", memory=False)
        # selector.add_children([collision_checker, communicator])
        # root.add_children([selector, explorer])
        safety_checker = py_trees.decorators.Inverter(
            name="Inverter", child=collision_checker
        )
        root.add_children([safety_checker, communicator, explorer])

        root.tick_once()
        self.explorationTree.insert(quads.Point(x_new[0, 0], x_new[0, 1], data=self.name))
        self.stepCount += 1
        return root.status


