import numpy as np
import py_trees
import py_trees.console as console
from py_trees import common
from .bt_viz import Visualization
from .quad_geom import Rectangle, QuadTree, Point


class CollisionChecker(py_trees.behaviour.Behaviour):
    def __init__(self, x, y, tree, robot_radius, name):
        self.x = x
        self.y = y
        self.tree = tree
        self.robot_radius = robot_radius
        self.__range = 2
        self.parent_name = name
        super(CollisionChecker, self).__init__(f"{name}/collision_checker")

    def check_nei_range(self, x, y, tree, range):

        bb = Rectangle(x - range/ 2, y - range/ 2, range, range)
        for p in tree.query(bb):
            if p.data is not None and p.data != self.parent_name:
                dx, dy = p.x - x, p.y - y
                if np.sqrt(dx ** 2 + dy ** 2 ) < self.__range * self.robot_radius:
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
        self.parent_name = name
        self.__range = 4
        super(Communicator, self).__init__(f"{name}/communicator")

    def get_neighbors(self, x, y, tree, range):
        bb = Rectangle(x - range / 2, y - range / 2, range, range)
        agents = []
        for p in tree.query(bb):
            dx, dy = p.x - x, p.y - y
            if np.sqrt(dx ** 2 + dy ** 2 ) < self.__range * self.robot_radius:
                if p.data is not None and p.data != self.parent_name:
                    agents.append(p.data)
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
        try:
            y_new = self.sensor.sense(self.x_new, self.rng).reshape(-1, 1)
        except:
            return self.status.SUCCESS

        self.model.add_data(self.x_new, y_new)
        self.model.optimize(num_iter=len(y_new), verbose=False)
        mean, std, error = self.evaluator.eval_prediction(self.model)
        msg = f"[{self.name}]:  gp = {np.mean(mean):.3f} +/- {np.mean(std):.3f} | err {np.mean(error):.3f}"
        self.logger.debug(msg)
        self.pub.set("/%s/xyz" % self.parent_name, f"{self.x_new[0, 0]},{self.x_new[0, 1]},{y_new[0, 0]}")

        return self.status.SUCCESS


class Agent(py_trees.behaviour.Behaviour):
    def __init__(self,rng, model, strategy, sensor, evaluator, task_extent, robotID):

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

        self.boundary = Rectangle(task_extent[0], task_extent[2], task_extent[1] - task_extent[0],
                                  task_extent[3] - task_extent[2])
        print(self.boundary)
        self.explorationTree = QuadTree( self.boundary,  capacity=8)


    def find_neighbors(self):
        tree = QuadTree( self.boundary,  capacity=8)
        state = Visualization.decode()
        for robotName, val in state.items():
            for key, robotState in val.items():
                if isinstance(robotState, list) and key == 'xyz' and robotName != self.name:
                    point = Point(robotState[0], robotState[1], data=robotName)
                    tree.insert(point)
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
        selector = py_trees.composites.Selector(name="Selector", memory=False)
        selector.add_children([collision_checker, communicator])
        root.add_children([selector, explorer])
        # safety_checker = py_trees.decorators.Inverter(
        #     name="Inverter", child=collision_checker
        # )
        # root.add_children([safety_checker, communicator, explorer])

        root.tick_once()
        point = Point(x_new[0, 0], x_new[0, 1])
        if self.boundary.contains(point):
            self.explorationTree.insert(point)
        self.stepCount += 1
        return root.status


