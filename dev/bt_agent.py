import numpy as np
import py_trees
import py_trees.console as console
import matplotlib.pyplot as plt
import quads


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
        for key, value in state.items():
            # print(key, value)
            plt.scatter(value['x'], value['y'], alpha=0.6)
            self.tree.insert(quads.Point(value['x'], value['y'], data=key))

        if self.step_count > 0 and self.step_count % 100 == 0:
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
                if key in ['x', 'y', 'z']:
                    value = float(value)

                # Store key-value pair in the nested dictionary
                data_dict[agent_id][key] = value

            except:
                pass
        # Print the resulting dictionary
        return data_dict


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
        self.pub.register_key(key="/%s/status" % self.name , access=py_trees.common.Access.WRITE)
        self.pub.register_key(key="/%s/x" % self.name, access=py_trees.common.Access.WRITE)
        self.pub.register_key(key="/%s/y" % self.name, access=py_trees.common.Access.WRITE)
        self.pub.register_key(key="/%s/z" % self.name, access=py_trees.common.Access.WRITE)

        self.explorationTree = quads.QuadTree( (0, 0), 22, 22, capacity=4)

    def find_neighbors(self):
        tree = quads.QuadTree(
            (0, 0),
            40, 40
         )
        for robotName, robotState in Visualization.decode().items():
            if robotName != self.name:
                if 'x' in robotState and 'y' in robotState:
                    point = quads.Point(robotState['x'], robotState['y'], data=robotName)
                    tree.insert(point, data=robotName)

        return tree

    def get_neighbors(self, x, y, tree, range):
        bb = quads.BoundingBox(min_x=x - range, min_y=y - range, max_x=x + range, max_y=y + range)

        agents = []
        for point in tree.within_bb(bb):
            agents.append(point.data)
        return agents

    def check_nei_range(self, x, y, tree, range):

        bb = quads.BoundingBox(min_x=x-range, min_y=y-range, max_x=x+range, max_y=y+range)
        for _ in tree.within_bb(bb):
            return True
        return False

        # robotRadius = 0.5
        # for robotName, robotState in Visualization.decode().items():
        #     if robotName != self.name:
        #         if 'x' in robotState and 'y' in robotState:
        #             dx = x - robotState['x']
        #             dy = y - robotState['y']
        #             dist = np.sqrt(dx * dx + dy * dy)
        #             if dist < 2 * robotRadius:
        #                 return True
        # return False




    def update(self):
        x_new = self.strategy.get(model=self.model)
        y_new = self.sensor.sense(x_new, self.rng).reshape(-1, 1)
        self.model.add_data(x_new, y_new)
        self.model.optimize(num_iter=len(y_new), verbose=False)
        mean, std, error = self.evaluator.eval_prediction(self.model)

        self.stepCount += 1

        msg = f"[robot {self.robotID}]: step = {self.stepCount} gp = {np.mean(mean):.3f} +/- {np.mean(std):.3f} | err {np.mean(error):.3f}"
        self.logger.debug(msg)

        nei_tree = self.find_neighbors()
        collision = False
        communicating = False
        if self.check_nei_range(x_new[0, 0], x_new[0, 1], nei_tree, 2 * self.robot_radius):
            console.info(console.red + msg + console.reset)
            collision = True
        else:
            for k, nei in enumerate(self.get_neighbors(x_new[0, 0], x_new[0, 1], nei_tree, 4 * self.robot_radius)):
                console.info(console.green + f"[robot {self.robotID}]: -> ({k + 1}) {nei}"  + console.reset)
                communicating = True

        status = "collision"  if collision else "communicating" if communicating else "exploring"

        self.pub.set("/%s/status" % self.name, status)
        self.pub.set("/%s/x" % self.name, x_new[0, 0])
        self.pub.set("/%s/y" % self.name, x_new[0, 1])
        self.pub.set("/%s/z" % self.name, y_new[0, 0])

        self.explorationTree.insert(quads.Point(x_new[0, 0], x_new[0, 1], data=self.name))


        return self.status.SUCCESS


