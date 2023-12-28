import numpy as np
import py_trees
import py_trees.console as console
from .quad_geom import Point, Rectangle
import random
import heapq
class ConflictHandler(py_trees.behaviour.Behaviour):
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


    def update(self):
        return self.status.SUCCESS
        neighbors = self.check_nei_dist()
        print(len(neighbors), neighbors)
        # # TODO tweak d
        r = neighbors[-1][0]
        d = 4 * r
        # print('d', d)
        pA = self.robot.state[:2]
        assert d < self.boundary.w and d < self.boundary.h and r < self.robot_radius * self.__range
        oA = pA - 2 * r
        cornerPoints = [Point(oA[0], oA[1]), Point(oA[0] + d, oA[1]),
                        Point(oA[0] + d, oA[1] + d), Point(oA[0], oA[1] + d)]

        initOA = oA.copy()
        initCornerPoints = cornerPoints.copy()
        showDebug = False
        while not all(map(self.boundary.contains, cornerPoints)):
            te = np.array([
                [self.boundary.x, self.boundary.y],
                [self.boundary.x - d, self.boundary.y],
                [self.boundary.x, self.boundary.y - d],
                [self.boundary.x -d, self.boundary.y -d]
            ])

            print('te', te - oA)
            dists = np.linalg.norm(te + oA, axis=1)
            # print('dists', dists)
            select = np.argmin(dists)
            oA = te[select]


            cornerPoints = [Point(oA[0], oA[1]), Point(oA[0] + d, oA[1]),
                            Point(oA[0] + d, oA[1] + d), Point(oA[0], oA[1] + d)]
            # showDebug = True

        if showDebug:
            print('initOrigin ', initOA, initCornerPoints)
            print('Found ', oA, cornerPoints)

        while True:
            x = random.uniform(oA[0], oA[0] + d)
            y = random.uniform(oA[1], oA[1] + d)

            pC = np.array([x, y])

            isValid = True
            for nei in self.neighbors:
                pB = nei[:2]
                if np.linalg.norm(pB - pC - pA) < 2 * self.robot_radius:
                    isValid = False
                    break

            if isValid:
                print('pc', pC, 'oA', oA, 'd', d)
                self.robot.goal_states[0][0] = x
                self.robot.goal_states[0][1] = y
                break



        msg = f"[{self.name}] : [modifying goal states]  {self.robot.goal_states}"
        console.info(console.bold_yellow + msg + console.reset)

        return self.status.SUCCESS