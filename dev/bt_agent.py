import numpy as np
import py_trees
import py_trees.console as console
import matplotlib.pyplot as plt
import quads
from py_trees import common


class Visualization(py_trees.behaviour.Behaviour):
    name = "visualization"
    def __init__(self, task_extent, sensor):
        super().__init__(self.name)
        print("width height",  task_extent[1] - task_extent[0],
            task_extent[3] - task_extent[2])
        self.task_extent = task_extent
        self.sensor = sensor
        self.tree = quads.QuadTree(
            (0, 0),
            2 * (task_extent[1] - task_extent[0]),
            2 * (task_extent[3] - task_extent[2]),
            capacity=4
        )
        self.step_count = 0

    def update(self):
        state = self.decode()
        plt.cla()
        plt.imshow(self.sensor.env.matrix, cmap=plt.cm.gray, interpolation='nearest',
                   extent=self.sensor.env.extent)
        for _, val in state.items():
            for key, value in val.items():
                if isinstance(value, list) and key == 'xyz':
                    plt.scatter(value[0], value[1], alpha=0.6)
                    self.tree.insert(quads.Point(value[0], value[1], data=key))


        if self.step_count > 0 and self.step_count % 300 == 0:
            quads.visualize(self.tree)
        plt.axis(self.task_extent)
        plt.pause(1e-2)
        self.step_count += 1
        return self.status.SUCCESS

    @staticmethod
    def decode():
        input_string = py_trees.display.unicode_blackboard()
        # Split the input string into lines
        lines = input_string.strip().split('\n')

        # Initialize an empty dictionary
        data_dict = {}

        if len(input_string) < 1:
            return  data_dict

        # Iterate through each line and extract key-value pairs
        for line in lines:

            if ":" not in line:
                continue

            line = line.replace('\x1b[33m', '').replace('\x1b[37m', '')
            # Split each line into key and value
            key, value = map(str.strip, line.split(':'))

            # Extract agent ID from the key
            agent_id = key.split('/')[1].strip()

            # Remove leading "/" from the key
            key = key.split('/')[2].strip()

            # Create a nested dictionary for each agent ID if not already present
            if agent_id not in data_dict:
                data_dict[agent_id] = {}

            try:
                # Convert float values from string to float
                # if key in ['x', 'y', 'z']:
                #     value = float(value)
                if key in ['xyz']:
                    value = list(map(float, value.split(',')))
                # Store key-value pair in the nested dictionary
                data_dict[agent_id][key] = value

            except:
                pass
        # Print the resulting dictionary
        return data_dict


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
        selector = py_trees.composites.Selector(name="Selector", memory=False)
        selector.add_children([collision_checker, communicator])
        root.add_children([selector, explorer])

        root.tick_once()
        self.explorationTree.insert(quads.Point(x_new[0, 0], x_new[0, 1], data=self.name))
        self.stepCount += 1
        return root.status


