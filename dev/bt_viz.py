import matplotlib.pyplot as plt
import py_trees
from .quad_geom import QuadTree, Rectangle, Point
class Visualization(py_trees.behaviour.Behaviour):
    name = "visualization"
    def __init__(self, task_extent, sensor):
        super().__init__(self.name)
        self.task_extent = task_extent
        self.sensor = sensor
        self.step_count = 0
        self.boundary = Rectangle(task_extent[0], task_extent[2], task_extent[1] - task_extent[0],
                  task_extent[3] - task_extent[2])
        self.tree = QuadTree( self.boundary,  capacity=8)

    def update(self):
        state = self.decode()
        plt.cla()
        plt.imshow(self.sensor.env.matrix, cmap=plt.cm.gray, interpolation='nearest',
                   extent=self.sensor.env.extent)
        for robotName, val in state.items():
            for key, value in val.items():
                if isinstance(value, list) and key == 'xyz':
                    plt.scatter(value[0], value[1], alpha=0.6)
                    p = Point(value[0], value[1], data=robotName)
                    self.tree.insert(p)

        ax = plt.gca()
        for rect in self.tree.sortedRect():
            ax.add_patch(rect.get_rect('y'))
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
                # Convert list of float values from string to float
                if key in ['xyz']:
                    value = list(map(float, value.split(',')))
                # Store key-value pair in the nested dictionary
                data_dict[agent_id][key] = value
            except:
                pass
        # Print the resulting dictionary
        return data_dict