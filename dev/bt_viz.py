import os.path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib.patches import Circle
import py_trees
import re
from collections import defaultdict
# camera
camera_position = (30, -82.6, 116.2)
camera_focal_point = (-4.25, 4.33, 2.12)
camera_up = (-0.07, 0.78, 0.63)

class Visualization(py_trees.behaviour.Behaviour):
    name = "visualization"
    def __init__(self, args, sensor):

        super().__init__(self.name)
        self.env_extent = args.env_extent
        self.task_extent = args.task_extent
        self.sensor = sensor
        self.step_count = 0
        self.robot_radius = 0.5
        self.communication_radius = 4
        self.collision_radius = 2
        self.show_animation = args.show_animation
        self.save_video = args.save_video
        experiment_id = "/".join([
            str(args.seed),
            args.env_name,
            args.strategy,
            args.kernel + args.postfix,
        ])
        self.outfolder = args.output_dir + experiment_id + f"_team{args.num_agents}_animation_v{args.version}"
        os.makedirs(self.outfolder, exist_ok=True)
        self.num_agents = args.num_agents
        self.history = defaultdict(list)
        self.grid_map = self.generate_grid_map(args.task_extent, self.robot_radius)

    def generate_grid_map(self, task_extent, cell_size):
        # Extract rectangle properties
        xmin, xmax, ymin, ymax = task_extent
        width, height = xmax - xmin, ymax - ymin

        # Calculate the number of rows and columns based on cell size
        num_rows = int(np.ceil(height / cell_size))
        num_cols = int(np.ceil(width / cell_size))

        # Initialize the grid map with zeros
        grid_map = np.zeros((num_rows, num_cols), dtype=int)

        # Iterate over rows and columns
        for i in range(num_rows):
            for j in range(num_cols):
                # Calculate the coordinates of the grid cell center
                cell_center_x = xmin + j * cell_size + cell_size / 2
                cell_center_y = ymin + i * cell_size + cell_size / 2

                # Check if the cell center is inside the rectangle
                if xmin <= cell_center_x <= (xmin + width) and ymin <= cell_center_y <= (ymin + height):
                    # Set the value of the grid cell to 1 (or any other value)
                    grid_map[i, j] = 0

        return grid_map

    def update_grid_map(self, cell_center_x, cell_center_y):
        xmin, xmax, ymin, ymax = self.task_extent
        width, height = xmax - xmin, ymax - ymin
        # Calculate the coordinates of the grid cell center
        j = (cell_center_x - xmin  - self.robot_radius / 2) / self.robot_radius
        i = (cell_center_y - ymin  - self.robot_radius / 2) / self.robot_radius

        i, j = np.floor(i).astype(int), np.floor(j).astype(int)

        # Check if the cell center is inside the rectangle
        if xmin <= cell_center_x <= (xmin + width) and ymin <= cell_center_y <= (ymin + height):
            # Set the value of the grid cell to 1 (or any other value)
            self.grid_map[i, j] = 1

    def compute_coverage(self):
        total_cells = self.grid_map.shape[0] * self.grid_map.shape[1]
        explored_cells = np.sum(np.sum(self.grid_map, axis=1))
        # unexplored_cells = total_cells - explored_cells
        return explored_cells / total_cells

    def update(self):
        state = self.decode()
        rmses = self.get_rmse()
        if self.show_animation or self.save_video:
            plt.cla()
            plt.imshow(self.sensor.env.matrix, cmap=plt.cm.gray, interpolation='nearest',
                       extent=self.env_extent)
            # mask
            cmap = ListedColormap(['black', 'white'])
            alpha = np.where(self.grid_map == 1, 0, 1).astype(np.float32)
            data = self.grid_map.copy()
            plt.imshow(data, cmap=cmap, alpha=alpha, extent=self.env_extent)


        robots = np.zeros((0, 2))
        for robotName, val in state.items():
            for key, value in val.items():
                if isinstance(value, list) and key == 'state':
                    robot = np.array([value[0], value[1]])
                    robots = np.vstack((robots, robot))
                    self.update_grid_map(robot[0], robot[1])
                    if self.show_animation or self.save_video:
                        scatter = plt.scatter(value[0], value[1], s=100, alpha=1.0)
                        # Plotting vectors
                        dt = 0.25

                        theta = value[2] + value[4]
                        x = robot[0] + value[3] * np.cos(theta)
                        y = robot[1] + value[3] * np.sin(theta)

                        plt.quiver(robot[0], robot[1], x, y, angles='xy', scale_units='xy', scale=1 / dt,
                                   color=scatter.get_facecolor(), label='Velocity Obstacle Vector')

        isCollision = False
        for i in range(len(robots)):
            for j in range(len(robots)):
                if i != j:
                    dist = np.linalg.norm(robots[i] - robots[j])
                    if dist < 2 * self.robot_radius:
                        if self.show_animation or self.save_video:
                            ax = plt.gca()
                            redCircle = Circle( (robots[i][0], robots[i][1]), self.collision_radius * self.robot_radius, color='red', alpha=0.4)
                            ax.add_patch(redCircle)
                        isCollision = True
                    elif dist < 4 * self.robot_radius:
                        if self.show_animation or self.save_video:
                            ax = plt.gca()
                            greenCircle = Circle((robots[i][0], robots[i][1]), self.communication_radius * self.robot_radius, color='green', alpha=0.4)
                            ax.add_patch(greenCircle)

        coverage = self.compute_coverage()
        if self.show_animation or self.save_video:
            if len(rmses) == self.num_agents:
                avg_rmse = np.average(list(rmses.values()))
                plt.title(f" step = {self.step_count} coverage = {coverage:.4f} avg rmse = {avg_rmse:.4f}")
                self.history["step"].append(self.step_count )
                self.history["coverage"].append(coverage)
                self.history["avg_rmse"].append(avg_rmse)
            plt.axis(self.env_extent)
        if self.show_animation:
            plt.pause(1e-2)
        elif self.save_video:
            outfile = os.path.join(self.outfolder, "%04d.png" % self.step_count)
            plt.savefig(outfile)
        self.step_count += 1
        print(f"coverage = {coverage:.4f}", flush=True, end="\r")
        # return self.status.SUCCESS if not
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

            if ":" not in line or "state" not in line:
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
                if key in ['state']:
                    value = list(map(float, value.split(',')))
                # Store key-value pair in the nested dictionary
                data_dict[agent_id][key] = value
            except:
                pass
        # Print the resulting dictionary
        return data_dict

    @staticmethod
    def get_rmse():
        input_string = py_trees.display.unicode_blackboard()
        # Split the input string into lines
        lines = input_string.strip().split('\n')

        # Initialize an empty dictionary
        data_dict = {}

        if len(input_string) < 1:
            return data_dict

        # Iterate through each line and extract key-value pairs
        for line in lines:

            if ":" not in line or "rmse" not in line:
                continue

            # Pattern to find integers and floating-point numbers
            pattern = r'\d+\.\d+|\d+'

            # Find all numbers in the string
            numbers = re.findall(pattern, line)

            # Convert the list of strings to a list of floats
            numbers = list(map(float, numbers))
            # rmse [0.0, 0.0, 36.0, 3.0, 37.0, 33.0, 4.976662259431621]
            # print("rmse", len(numbers), numbers)
            if len(numbers) == 2:
                data_dict[int(numbers[0])] = numbers[-1]
            elif len(numbers) == 7:
                data_dict[int(numbers[3])] = numbers[-1]
        return data_dict