from typing import List
import numpy as np
from ..objectives.entropy import gaussian_entropy
from ..models import IModel
from .strategy import IStrategy
from ..robots import IRobot

import random
class Point:
    def __init__(self, x, y, data=None):
        self.x = x
        self.y = y
        self.data = data

    def __iter__(self):
        yield (self.x, self.y, self.data)

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
class DistributedPlanning(IStrategy):
    """Distributed myopic informative planning."""

    def __init__(
        self,
        task_extent: List[float],
        rng: np.random.RandomState,
        num_candidates: int,
        robot: IRobot,
    ) -> None:
        """

        Parameters
        ----------
        task_extent: List[float], [xmin, xmax, ymin, ymax]
            Bounding box of the sampling task workspace.
        rng: np.random.RandomState
            Random number generator if `get` has random operations.
        num_candidates: int
            Number of candidate locations to evaluate.
        robot: IRobot
            A robot model.

        """
        super().__init__(task_extent, rng)
        self.num_candidates = num_candidates
        self.robot = robot
        self.additional_parameter = {}

    def __call__(self, *args, **kwargs):
        for key, val in kwargs.items():
            self.additional_parameter[key] = val

    def get(self, model: IModel, num_states: int = 1) -> np.ndarray:
        """Get goal states for sampling.

        Parameters
        ----------
        model: IModel, optional
            A probabilistic model that provides `mean` and `std` via `forward`.
        num_states: int
            Number of goal states.

        Returns
        -------
        goal_states: np.ndarray, shape=(num_states, dim_states)
            Sampling goal states.

        """
        if num_states != 1:
            raise ValueError("`num_states` must be 1 in InformativePlanning.")
        while len(self.robot.sampling_locations) == 0:
            # TODO from exploration tree compute candidate locations

            task_extent = self.task_extent.copy()
            if "exploration_tree" in self.additional_parameter:
                exploration_tree = self.additional_parameter["exploration_tree"]
                rects = exploration_tree.sortedRect()  # sort largest to smallest

                neighbors = self.additional_parameter['neighbors']
                bestRect = 0
                for rect in rects:
                    count = 1
                    for neighbor in neighbors:
                        if rect.contains(Point(neighbor[0], neighbor[1])):
                            count += 1
                    score = rect.area() / count
                    if len(rect.points) < exploration_tree.capacity and  score > bestRect:
                        task_extent = rect.box()
                        bestRect = score
                # print(bestRect, task_extent)


            # Propose candidate locations
            xs = self.rng.uniform(
                low=task_extent[0],
                high=task_extent[1],
                size=self.num_candidates,
            )
            ys = self.rng.uniform(
                low=task_extent[2],
                high=task_extent[3],
                size=self.num_candidates,
            )
            candidate_states = np.column_stack((xs, ys))


            # Evaluate candidates
            _, std = model(candidate_states)
            entropy = gaussian_entropy(std.ravel())
            diffs = candidate_states - self.robot.state[:2]
            dists = np.hypot(diffs[:, 0], diffs[:, 1])
            # Normalized scores
            normed_entropy = (entropy - entropy.min()) / entropy.ptp()
            normed_dists = (dists - dists.min()) / dists.ptp()
            scores = normed_entropy - normed_dists
            # Append waypoint
            prob = np.random.normal()
            sorted_indices = np.argsort(scores)

            goal_states = candidate_states[sorted_indices[-num_states:]]
            self.robot.goal_states.append(goal_states.ravel())

            return []
        #     # Controling and sampling
        #     while self.robot.has_goal:
        #         self.robot.update(*self.robot.control())
        #
        # x_new = self.robot.commit_data()
        # return x_new
