from typing import List

import numpy as np

from ..objectives.entropy import gaussian_entropy
from ..models import IModel
from .strategy import IStrategy
from ..robots import IRobot
from shapely import geometry
from time import sleep

class DistributedPlanning(IStrategy):
    """Myopic informative planning."""

    def __init__(
        self,
        task_extent: List[float],
        rng: np.random.RandomState,
        num_candidates: int,
        robot: IRobot,
        index: int,
        alpha: float
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
        index: int
            Robot index
        alpha: float
            complementary filter coefficient

        """
        super().__init__(task_extent, rng)
        self.num_candidates = num_candidates
        self.robot = robot
        self.index = index

        self.other_models = None
        self.partition = None
        self.alpha = alpha
        self.task_assigned = False


    def set(self, other_models:List[IModel], partition:np.ndarray):
        """

        Parameters
        ----------
        other_models: List[IModel]
            gp models for other robots
        partition:
            voronoi partition for current robot
        """

        self.other_models = other_models
        self.partition = partition
        self.task_assigned = True

    def get_valid_states(self, candidate_states:np.ndarray):
        """ filter out invalid states based on partition
        Parameters
        ----------
        candidate_states: np.ndarray
            a set of random samples in task_extent

        Returns
        -------
        valid_states:  np.ndarray, shape=(num_states, dim_states)
            states within geometric partition
        """

    #     construct polygon from partion
        polygon = geometry.Polygon(self.partition)
        valid_points = []
        for np_point in candidate_states:
            point = geometry.Point(np_point)
            if polygon.contains(point):
                valid_points.append(np_point)

        return np.array(valid_points)

    def get_entropy(self, model, candidate_states):
        _, std = model(candidate_states)
        entropy = gaussian_entropy(std.ravel())
        normed_entropy = (entropy - entropy.min()) / entropy.ptp()

        return normed_entropy


    def get(self, model: IModel, num_states: int = 1) -> np.ndarray:
        """      """
        if not self.task_assigned:
            raise RuntimeError("task has not been assigned check DDMP thread")

        if num_states != 1:
            raise ValueError("`num_states` must be 1 in InformativePlanning.")

        if self.other_models is None:
            raise RuntimeError("other gp models have not been updated")

        if self.partition is None:
            raise RuntimeError("voronoi partition has not been updated")

        while len(self.robot.sampling_locations) == 0:
            # Propose candidate locations
            xs = self.rng.uniform(
                low=self.task_extent[0],
                high=self.task_extent[1],
                size=self.num_candidates,
            )
            ys = self.rng.uniform(
                low=self.task_extent[2],
                high=self.task_extent[3],
                size=self.num_candidates,
            )
            sampled_states = np.column_stack((xs, ys))
            candidate_states = self.get_valid_states(sampled_states)

            if not len(candidate_states):
                print(f"[DistributedPlanning] Robot {self.index} no valid states found")
                sleep(1e-2)
                # try again
                return self.get(model, num_states)

            # Evaluate candidates
            normed_entropy = self.get_entropy(model, candidate_states)

            # accumulate entropy from other models
            # for i, m in enumerate(self.other_models):
            #     if i != self.index:
            #         entropy = self.get_entropy(m, candidate_states)
            #         normed_entropy = normed_entropy * self.alpha + (1 - self.alpha) * entropy

            others_entropy = None
            for i, m in enumerate(self.other_models):
                if i != self.index:
                    if others_entropy is None:
                        others_entropy = self.get_entropy(m, candidate_states)
                    else:
                        others_entropy += self.get_entropy(m, candidate_states)
            normed_entropy = normed_entropy * self.alpha + (1 - self.alpha) * others_entropy


            diffs = candidate_states - self.robot.state[:2]
            dists = np.hypot(diffs[:, 0], diffs[:, 1])
            # Normalized scores
            normed_dists = (dists - dists.min()) / dists.ptp()



            scores = normed_entropy - normed_dists
            # Append waypoint
            sorted_indices = np.argsort(scores)
            goal_states = candidate_states[sorted_indices[-num_states:]]
            self.robot.goal_states.append(goal_states.ravel())
            # Controling and sampling
            while self.robot.has_goal:
                self.robot.update(*self.robot.control())
        x_new = self.robot.commit_data()
        self.task_assigned = False
        return x_new
