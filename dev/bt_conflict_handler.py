import numpy as np
import py_trees
import py_trees.console as console
from .quad_geom import Point, Rectangle
import random
import heapq
class ConflictHandler(py_trees.behaviour.Behaviour):
    """
    - two robots within communication distance will change their next goal location to avoid collision
    - collision avoidance use velocity obstacle to compute the avoidance direction
    - once avoidance direction is found, a rectangle search space is generated to sample next best goal location
    - if search space is out of the task extent, it is moved inside the task extent
    - finally randomly sample a goal states within the search space if goal state exist

    """
    def __init__(self, strategy, neighbors, robot_radius, task_extent, name):
        self.strategy = strategy
        self.robot = strategy.robot
        self.neighbors = neighbors
        self.robot_radius = robot_radius
        self.task_extent = task_extent
        self.parent_name = name
        self.x = self.robot.state[0]
        self.y = self.robot.state[1]
        self.__range = 4
        self.boundary = Rectangle(task_extent[0], task_extent[2], task_extent[1] - task_extent[0],
                                  task_extent[3] - task_extent[2])
        super().__init__(f"{name}/conflict_handler")

    def check_nei_dist(self):

        agents = []
        for p in self.neighbors:
            dx, dy = p[0] - self.x, self.y - p[1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist < self.__range * self.robot_radius:
                heapq.heappush(agents, (dist, p[:-1]))

        return agents

    def move_inside(self, smaller_rect, larger_rect):
        # Calculate the translation vectors for both x and y axes
        translation_x = max(larger_rect.x - smaller_rect.x,
                            min(larger_rect.x + larger_rect.w - smaller_rect.w - smaller_rect.x, 0))
        translation_y = max(larger_rect.y - smaller_rect.y,
                            min(larger_rect.y + larger_rect.h - smaller_rect.h - smaller_rect.y, 0))

        # Update the position of the smaller rectangle
        smaller_rect.x += translation_x
        smaller_rect.y += translation_y

    def velocity_obstacle(self, agent_pos, other_agent_pos, other_agent_vel, max_speed, radius):
        relative_pos = other_agent_pos - agent_pos
        relative_vel = other_agent_vel

        d = np.linalg.norm(relative_pos)
        if d == 0:
            return np.zeros(2)

        vo = relative_vel - relative_pos * (max_speed / d)
        vo_magnitude = np.linalg.norm(vo)

        if vo_magnitude > 0:
            vo_normalized = vo / vo_magnitude
            return vo_normalized
        else:
            return np.zeros(2)

    def vo_trajectory(self, agent_pos, other_agents_pos, other_agents_vel, max_speed, radius):
        avoidance_direction = np.zeros(2)

        for i in range(len(other_agents_pos)):
            vo = self.velocity_obstacle(agent_pos, other_agents_pos[i], other_agents_vel[i], max_speed, radius)
            avoidance_direction += vo

        avoidance_direction /= len(other_agents_pos) if len(other_agents_pos) > 0 else 1
        avoidance_direction /= np.linalg.norm(avoidance_direction) if np.linalg.norm(avoidance_direction) > 0 else 1

        avoidance_trajectory = agent_pos + avoidance_direction * max_speed * 2 * radius


        return avoidance_trajectory
    def update(self):
        neighbors = self.check_nei_dist()

        if len(neighbors) > 0:

            agent_pos = self.robot.state[:2]

            other_agents_pos = np.zeros((0, 2))
            other_agents_vel = np.zeros((0, 2))
            for neighbor in neighbors:
                _, nstate = neighbor
                other_agents_pos = np.vstack((other_agents_pos, np.array([nstate[0], nstate[1]])))
                other_agents_vel = np.vstack((other_agents_vel, np.array([nstate[-2], nstate[-1]])))
            max_speed = self.robot.max_lin_vel
            radius = neighbors[-1][0]
            d = 2 * radius
            oA = self.vo_trajectory(agent_pos, other_agents_pos, other_agents_vel, max_speed, radius)
            oA = oA - 2 * radius

            search_space = Rectangle(oA[0], oA[1], d, d)
            self.move_inside(search_space, self.boundary)
            while True:
                x = random.uniform(search_space.x, search_space.x + search_space.w)
                y = random.uniform(search_space.y, search_space.y + search_space.h)

                pC = np.array([x, y])
                nearst_obs_dist = min(np.linalg.norm(other_agents_pos - pC, axis=1))
                if nearst_obs_dist < self.robot_radius * 2:
                    continue

                if len(self.robot.goal_states) > 0:
                    self.robot.goal_states[0][0] = x
                    self.robot.goal_states[0][1] = y
                break

            msg = f"[{self.name}] : [modifying goal states]  {self.robot.goal_states}"
            console.info(console.bold_yellow + msg + console.reset)

        return self.status.SUCCESS