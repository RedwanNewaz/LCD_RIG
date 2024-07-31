import pickle
import numpy as np
import matplotlib.pyplot as plt
def get_trajectories(file_path):
    with open(file_path, 'rb') as file:
        loaded_history_robot_states = pickle.load(file)
    return loaded_history_robot_states

def show_animation(trajs):
    paths = [path for path in trajs.values()]
    N = len(paths)
    max_len = min(map(len, paths))
    # print(max_len)
    global_min_distance = float('inf')
    for t in range(max_len):
        plt.cla()
        robots = []
        for i in range(N):
            points = np.array(paths[i][:t + 1])
            robots.append(paths[i][t])
            plt.plot(points[:, 0], points[:, 1])

        # Compute pairwise distances
        robots = np.array(robots)
        distances = np.linalg.norm(robots[:, np.newaxis] - robots, axis=2)

        # Extract the upper triangle of the distance matrix without the diagonal
        pairwise_distances = distances[np.triu_indices(3, k=1)]

        # Find the minimum distance
        min_distance = np.min(pairwise_distances)
        global_min_distance  = min(global_min_distance, min_distance)

        plt.title(f"min distance: {min_distance:.3f}")
        plt.axis([-14, 14, -14, 14])
        plt.pause(0.001)
    print(f"max distance: {global_min_distance:.3f}")
if __name__ == '__main__':
    # seed = 3555 # min distance 4.708
    seed = 3560 # max distance: 4.607
    filepath = f"/home/airlab/PycharmProjects/LCD_RIG/outputs/{seed}/temp_data/distributed/ak_team3_path_v1.pkl"
    trajs = get_trajectories(filepath)
    show_animation(trajs)