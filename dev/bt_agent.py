import numpy as np
import py_trees
import py_trees.console as console
from py_trees import common
from .bt_viz import Visualization
from .bt_collision_handler import CollisionHandler
from .bt_conflict_handler import ConflictHandler
from .quad_geom import Rectangle, QuadTree, Point
from time import sleep



class Communicator(py_trees.behaviour.Behaviour):
    def __init__(self, strategy, neighbors, robot_radius, name):
        self.strategy = strategy
        self.robot = strategy.robot
        self.x = self.robot.state[0]
        self.y = self.robot.state[1]
        self.neighbors = neighbors
        self.robot_radius = robot_radius
        self.parent_name = name
        self.__range = 4
        super(Communicator, self).__init__(f"{name}/communicator")

    def get_neighbors(self):
        agents = []
        local_neighbors = []
        for p in self.neighbors:
            dx, dy = p[0] - self.x, self.y - p[1]
            if np.sqrt(dx ** 2 + dy ** 2 ) < self.__range * self.robot_radius:
                agents.append(p[-1])
                local_neighbors.append(Point(p[0], p[1], p[-1]))
        self.strategy(neighbors=local_neighbors)
        return agents

    def update(self):
        for k, nei in enumerate(self.get_neighbors()):
            console.info(console.green + f"[{self.name}]: -> ({k + 1}) {nei}" + console.reset)
        return self.status.SUCCESS


class Learner(py_trees.behaviour.Behaviour):
    def __init__(self, robot, rng, model, evaluator, sensor, name):
        self.robot = robot
        self.rng = rng
        self.model = model
        self.evaluator = evaluator
        self.sensor = sensor
        self.parent_name = name
        super().__init__(f"{name}/learner")


    def update(self) -> common.Status:
        # print("learning ...")
        try:
            x_new = self.robot.commit_data()
            y_new = self.sensor.sense(x_new, self.rng).reshape(-1, 1)
            self.model.add_data(x_new, y_new)
            self.model.optimize(num_iter=len(y_new), verbose=False)
        except:
            return self.status.FAILURE
        mean, std, error = self.evaluator.eval_prediction(self.model)
        msg = f"[{self.name}]:  gp = {np.mean(mean):.3f} +/- {np.mean(std):.3f} | err {np.mean(error):.3f}"
        self.logger.debug(msg)
        # console.info(console.cyan + f"[{self.name}]: {msg}" + console.reset)

        return self.status.SUCCESS


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

class Planner(py_trees.behaviour.Behaviour):
    def __init__(self, model, strategy, explorationTree, name):
        self.model = model
        self.strategy = strategy
        self.explorationTree = explorationTree
        super().__init__(f"{name}/planner")

    def update(self):
        # print("planning ...")
        self.strategy(exploration_tree=self.explorationTree)
        self.strategy.get(model=self.model)
        return self.status.SUCCESS

class Agent(py_trees.behaviour.Behaviour):
    def __init__(self,rng, model, strategy, sensor, evaluator, task_extent, robotID):
        self.rng = rng
        self.model = model
        self.strategy = strategy
        self.robot = strategy.robot
        self.sensor = sensor
        self.evaluator = evaluator
        self.robotID = robotID
        self.stepCount = 0
        self.robot_radius = 0.5
        self.taskExtent = task_extent
        name = "Agent%03d" % self.robotID
        super().__init__(name)

        self.pub = py_trees.blackboard.Client(name=name, namespace=name)
        self.pub.register_key(key="/%s/state" % self.name , access=py_trees.common.Access.WRITE)

        self.boundary = Rectangle(task_extent[0], task_extent[2], task_extent[1] - task_extent[0],
                                  task_extent[3] - task_extent[2])
        # print(self.boundary)
        self.explorationTree = QuadTree( self.boundary,  capacity=16)

    def find_neighbors(self):

        state = Visualization.decode()
        neighbors = []
        for robotName, val in state.items():
            for key, robotState in val.items():
                if isinstance(robotState, list) and key == 'state' and robotName != self.name:
                    neighbors.append([robotState[0], robotState[1], robotState[2], robotState[3], robotState[4], robotName])
        return neighbors


    def update(self):


        hasGoal = py_trees.behaviours.Success(name="hasGoal") if self.robot.has_goal else py_trees.behaviours.Failure(name="hasGoal")
        neighbors = self.find_neighbors()

        self.strategy(neighbors=neighbors)

        collision_handler = CollisionHandler(self.strategy, self.pub, self.explorationTree, neighbors, self.robot_radius, self.name)
        conflict_handler = ConflictHandler(self.strategy, neighbors, self.robot_radius, self.taskExtent, self.name)
        communicator = Communicator(self.strategy, neighbors, self.robot_radius, self.name)
        learner = Learner(self.robot, self.rng, self.model, self.evaluator, self.sensor,  self.name)
        explorer = Explorer(self.robot, self.explorationTree, self.pub, self.name)
        planner = Planner(self.model, self.strategy, self.explorationTree, self.name)


        root = py_trees.composites.Sequence(name="RootSequence", memory=True)
        plannerSelector = py_trees.composites.Selector(name="PlannerSelector", memory=True)
        explorerSequence = py_trees.composites.Sequence(name="ExplorerSequence", memory=True)

        neighborSequence = py_trees.composites.Sequence(name="NeighborSequence", memory=True)
        plannerSelector.add_children([hasGoal, planner])
        explorerSequence.add_children([neighborSequence, explorer])

        neighborSequence.add_children([communicator, conflict_handler])
        root.add_children([plannerSelector, explorerSequence, learner])


        root.tick_once()
        return root.status


