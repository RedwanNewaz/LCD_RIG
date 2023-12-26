import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import py_trees
class Visualization(py_trees.behaviour.Behaviour):
    name = "visualization"
    def __init__(self, task_extent, sensor):
        super().__init__(self.name)
        self.task_extent = [task_extent[0] -1, task_extent[1] + 1, task_extent[2] -1, task_extent[3] + 1]
        self.sensor = sensor
        self.step_count = 0
        self.robot_radius = 0.5
        self.communication_radius = 4
        self.collision_radius = 2

    def update(self):
        state = self.decode()
        plt.cla()
        plt.imshow(self.sensor.env.matrix, cmap=plt.cm.gray, interpolation='nearest',
                   extent=self.task_extent)

        robots = np.zeros((0, 2))
        for robotName, val in state.items():
            for key, value in val.items():
                if isinstance(value, list) and key == 'xyz':
                    robot = np.array([value[0], value[1]])
                    robots = np.vstack((robots, robot))
                    plt.scatter(value[0], value[1], s=100, alpha=1.0)

        ax = plt.gca()
        isCollision = False
        for i in range(len(robots)):
            for j in range(len(robots)):
                if i != j:
                    dist = np.linalg.norm(robots[i] - robots[j])
                    if dist < 2 * self.robot_radius:
                        redCircle = Circle( (robots[i][0], robots[i][1]), self.collision_radius * self.robot_radius, color='red', alpha=0.4)
                        ax.add_patch(redCircle)
                        isCollision = True
                    elif dist < 4 * self.robot_radius:
                        greenCircle = Circle((robots[i][0], robots[i][1]), self.communication_radius * self.robot_radius, color='green', alpha=0.4)
                        ax.add_patch(greenCircle)
        plt.axis(self.task_extent)
        plt.pause(1e-2)
        self.step_count += 1
        return self.status.SUCCESS if not isCollision else self.status.FAILURE

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
                # Convert list of float values from string to float
                if key in ['xyz']:
                    value = list(map(float, value.split(',')))
                # Store key-value pair in the nested dictionary
                data_dict[agent_id][key] = value
            except:
                pass
        # Print the resulting dictionary
        return data_dict