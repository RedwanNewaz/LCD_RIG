import py_trees
import py_trees.console as console
from .bt_viz import Visualization
from .bt_conflict_handler import ConflictHandler
from .bt_communicator import Communicator
from .quad_geom import Rectangle, QuadTree
from .bt_learner import Learner
from .bt_planner import Planner
from .bt_explorer import Explorer
import subprocess

def save_dot_tree(root):
    level = "component"
    enum_level = py_trees.common.string_to_visibility_level(level)
    py_trees.display.render_dot_tree(root, enum_level)

    if py_trees.utilities.which("xdot"):
        try:
            subprocess.call(["xdot", "demo_dot_graphs_%s.dot" % level])
        except KeyboardInterrupt:
            pass
    else:
        print("")
        console.logerror(
            "No xdot viewer found, skipping display [hint: sudo apt install xdot]"
        )
        print("")




class Agent(py_trees.behaviour.Behaviour):
    def __init__(self,rng, model, strategy, sensor, evaluator, task_extent, robotID, exp_logger, robot_radius):
        self.rng = rng
        self.model = model
        self.strategy = strategy
        self.robot = strategy.robot
        self.sensor = sensor
        self.evaluator = evaluator
        self.robotID = robotID
        self.stepCount = 0
        self.robot_radius = robot_radius
        self.taskExtent = task_extent
        self.exp_logger = exp_logger
        name = "RIG%03d" % self.robotID
        super().__init__(name)

        self.pub = py_trees.blackboard.Client(name=name, namespace=name)
        self.pub.register_key(key="/%s/state" % self.name , access=py_trees.common.Access.WRITE)
        self.pub.register_key(key="/%s/rmse" % self.name, access=py_trees.common.Access.WRITE)

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

        conflict_handler = ConflictHandler(self.strategy, neighbors, self.robot_radius, self.taskExtent, self.name)
        communicator = Communicator(self.strategy, neighbors, self.robot_radius, self.name)
        learner = Learner(self.robot, self.rng, self.model, self.evaluator, self.sensor,  self.name, self.exp_logger, self.pub)
        explorer = Explorer(self.robot, self.explorationTree, self.pub, self.name)
        planner = Planner(self.model, self.strategy, self.explorationTree, self.name)


        root = py_trees.composites.Sequence(name=self.name, memory=True)
        plannerSelector = py_trees.composites.Selector(name="PlannerSelector", memory=True)
        explorerSequence = py_trees.composites.Sequence(name="ExplorerSequence", memory=True)

        neighborSequence = py_trees.composites.Sequence(name="NeighborSequence", memory=True)
        plannerSelector.add_children([hasGoal, planner])
        explorerSequence.add_children([neighborSequence, explorer])

        neighborSequence.add_children([communicator, conflict_handler])
        root.add_children([plannerSelector, explorerSequence, learner])


        root.tick_once()
        # save_dot_tree(root)

        return root.status


