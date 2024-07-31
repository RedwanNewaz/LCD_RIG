import pickle
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import fcl
from collections import defaultdict
robot_size = 1.4
class CollisionChecker:
    def __init__(self, obstacles):
        self.obstacles = obstacles
        self.collision_checker = self.getObstacleChecker()


    def checkCollision(self, robot_pos, manager):
        g1 = fcl.Sphere(2.0 * robot_size)
        t1 = fcl.Transform(np.array(robot_pos))
        o1 = fcl.CollisionObject(g1, t1)

        req = fcl.CollisionRequest(num_max_contacts=10)
        rdata = fcl.CollisionData(request=req)

        manager.collide(o1, rdata, fcl.defaultCollisionCallback)

        return rdata.result.is_collision

    def minCollisionDistance(self, robot_pos, manager):
        g1 = fcl.Box(robot_size, robot_size, robot_size)
        t1 = fcl.Transform(np.array(robot_pos))
        o1 = fcl.CollisionObject(g1, t1)

        req = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=False)
        rdata = fcl.DistanceData(request=req)

        manager.distance(o1, rdata, fcl.defaultDistanceCallback)

        return rdata.result.min_distance

    def getObstacleChecker(self):
        # Define the cube positions and size
        cube_positions = []
        for obstacle in self.obstacles:
            cube_positions.append([obstacle[0], obstacle[1], 0.1])

        cube_size = robot_size
        box = fcl.Box(cube_size, cube_size, cube_size)
        objs1 = [fcl.CollisionObject(box, fcl.Transform(np.array(t))) for t in cube_positions]
        manager1 = fcl.DynamicAABBTreeCollisionManager()
        manager1.registerObjects(objs1)
        return manager1

def sparse_sampled_path(path, num_samples):
    # Function to perform sparse sampling
    def sparse_sample(x, y, num_samples):
        # Step 1: Detect peaks and valleys
        peaks, _ = find_peaks(y)
        valleys, _ = find_peaks(-y)
        important_points = np.concatenate((peaks, valleys))

        # Step 2: Add start and end points
        important_points = np.concatenate(([0], important_points, [len(y) - 1]))

        # Step 3: Uniformly sample the remaining points
        if len(important_points) < num_samples:
            remaining_samples = num_samples - len(important_points)
            all_indices = np.arange(len(y))
            remaining_indices = np.setdiff1d(all_indices, important_points)
            sampled_remaining_indices = np.linspace(0, len(remaining_indices) - 1, remaining_samples).astype(int)
            sampled_remaining_indices = remaining_indices[sampled_remaining_indices]
            sampled_indices = np.concatenate((important_points, sampled_remaining_indices))
        else:
            sampled_indices = important_points

        # Sort sampled indices
        sampled_indices = np.sort(sampled_indices)

        return sampled_indices

    indexes = sparse_sample(path[:, 0], path[:, 1], num_samples)
    return path[indexes]


def import_data(file_path, seed, sample_size=900):
    # Read the saved pickle file
    with open(file_path, 'rb') as file:
        loaded_history_robot_states = pickle.load(file)

    # Display the loaded data
    history = defaultdict(list)
    for robotName, path in loaded_history_robot_states.items():
        path = np.array(path)
        path = sparse_sampled_path(path, sample_size)
        history[robotName] = path
        # print(robotName, path.shape)
        plt.scatter(path[:, 0], path[:, 1])
        for robotOthers, pathOthers in loaded_history_robot_states.items():
            if robotOthers != robotName:
                pathOthers = np.array(pathOthers)
                pathOthers = sparse_sampled_path(pathOthers, sample_size)
                for r1, r2 in zip(path, pathOthers):
                    if np.linalg.norm(r1 - r2) < robot_size:
                        print(robotName, robotOthers)
                        plt.scatter(r1[0], r2[0], color='red' )

    plt.show()
    filepath = f"/home/airlab/PycharmProjects/LCD_RIG/outputs/{seed}/temp_data/distributed/ak_team3_path_sparsed_radius175_{sample_size}_samples.pkl"
    # Save the defaultdict to a pickle file
    with open(filepath, 'wb') as file:
        pickle.dump(history, file)


def remove_collisions(path, pathOther, robot_size):
    """
    Recursively remove points from path that are in collision with points in pathOther.
    """

    def check_and_remove(path, pathOther, robot_size, idx=0):
        if idx >= len(path):
            return path
        for r2 in pathOther:
            if np.linalg.norm(path[idx] - r2) < robot_size:
                # Collision detected, remove the point and check again
                return check_and_remove(np.delete(path, idx, axis=0), pathOther, robot_size, idx)
        return check_and_remove(path, pathOther, robot_size, idx + 1)

    return check_and_remove(path, pathOther, robot_size)

def viz_results(loaded_history_robot_states):
    for robotName, path in loaded_history_robot_states.items():
        path = np.squeeze(np.array(path))
        print(path.shape)
        plt.scatter(path[:, 0], path[:, 1])
        for robotOther, pathOther in loaded_history_robot_states.items():
            pathOther = np.squeeze(np.array(pathOther))
            # TODO remove the collision points
            if robotOther != robotName:
                for r1, r2 in zip(path, pathOther):
                    if np.linalg.norm(r1 - r2) < robot_size:
                        # collision detected
                        plt.scatter(r1[0], r2[0], color='red')

    plt.show()
def check_collision(file_path):
    # Read the saved pickle file
    with open(file_path, 'rb') as file:
        loaded_history_robot_states = pickle.load(file)
    return viz_results(loaded_history_robot_states)

def gen_path(file_path):
    with open(file_path, 'rb') as file:
        loaded_history_robot_states = pickle.load(file)

    history = {}
    for robotName, path in loaded_history_robot_states.items():
        path = np.squeeze(np.array(path))
        print(path.shape)
        # plt.scatter(path[:, 0], path[:, 1], label=f"{robotName} original")
        for robotOther, pathOther in loaded_history_robot_states.items():
            if robotOther != robotName:
                pathOther = np.squeeze(np.array(pathOther))
                path = remove_collisions(path, pathOther, robot_size)
        # plt.scatter(path[:, 0], path[:, 1], label=f"{robotName} updated")
        history[robotName] = path

    viz_results(history)

if __name__ == '__main__':
    seed = 3560
    filepath = f"/home/airlab/PycharmProjects/LCD_RIG/outputs/{seed}/temp_data/distributed/ak_team3_path_v1.pkl"
    import_data(filepath, seed)

    # filepath = "/home/airlab/PycharmProjects/LCD_RIG/outputs/3555/temp_data/distributed/ak_team3_path_sparsed_v1.pkl"
    # gen_path(filepath)