import numpy as np
import py_trees
import py_trees.console as console


class Trajectory:
    def __init__(self):
        self.data = None
        self.sender = None

    def __str__(self):
        return str(self.__dict__)

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def filter_points_by_distance_unsorted(points, distance_threshold):
    filtered_points = []

    for i, point1 in enumerate(points):
        if not any(calculate_distance(point1, point2) < distance_threshold for point2 in filtered_points):
            filtered_points.append(point1)

    return filtered_points

class Communicator(py_trees.behaviour.Behaviour):
    def __init__(self, strategy, neighbors, robot_radius, name):
        self.strategy = strategy
        self.robot = strategy.robot
        self.x = self.robot.state[0]
        self.y = self.robot.state[1]
        self.robot_radius = robot_radius
        self.parent_name = name
        self.__range = 4

        self.neighbors = self.get_neighbors(neighbors)
        super(Communicator, self).__init__(f"{name}/communicator")
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key("trajectory", access=py_trees.common.Access.READ)
        self.nei_blackboards = [self.attach_blackboard_client(name=f"{nei}/communicator") for nei in self.neighbors]
        for nei in self.nei_blackboards:
            nei.register_key("trajectory", access=py_trees.common.Access.WRITE)

    def get_neighbors(self, neighbors):
        agents = []
        selectedNei = []
        for p in neighbors:
            dx, dy = p[0] - self.x, self.y - p[1]
            if np.sqrt(dx ** 2 + dy ** 2 ) < self.__range * self.robot_radius:
                agents.append(p[-1])
                selectedNei.append(p[:-1])
        if len(agents) > 0:
            self.strategy(neighbors=selectedNei)
        return agents


    def update(self):

        for k, nei in enumerate(self.neighbors):
            self.nei_blackboards[k].trajectory = Trajectory()
            self.nei_blackboards[k].trajectory.sender = self.name

            data = []
            for sample in self.robot.sampling_locations:
                pA = self.robot.state[:2]
                if np.linalg.norm(pA - sample[:2]) > self.__range * self.robot_radius:
                    data.append(sample)
            # print(len(data))
            filterdPoints = filter_points_by_distance_unsorted(data, self.robot_radius)
            self.nei_blackboards[k].trajectory.data = filterdPoints
            console.info(console.green + f"[{self.name}]: -> ({k + 1}) {nei}" + console.reset)

        if ( len(self.neighbors) > 0 and  self.blackboard.get("trajectory")):
                console.info(console.green + f"[{self.name}]: "
                                             f"From ({self.blackboard.trajectory.sender} "
                                             f"samples = {len(self.blackboard.trajectory.data)})"
                            + console.reset)
                init = self.robot.sampling_locations
                init.extend(self.blackboard.trajectory.data)
                filterdPoints = filter_points_by_distance_unsorted(init, self.robot_radius)
                # print( len(self.robot.sampling_locations), len(init), len(filterdPoints))
                self.robot.sampling_locations = filterdPoints



        return self.status.SUCCESS