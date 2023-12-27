import numpy as np
import py_trees
import py_trees.console as console
from .quad_geom import Point
import random
class CollisionHandler(py_trees.behaviour.Behaviour):
    def __init__(self, strategy, pub, explorationTree, neighbors, robot_radius, name):
        self.strategy = strategy
        self.robot = strategy.robot
        self.x = self.robot.state[0]
        self.y = self.robot.state[1]
        self.pub = pub
        self.explorationTree = explorationTree
        self.neighbors = neighbors
        self.robot_radius = robot_radius
        self.__range = 4
        self.parent_name = name
        super(CollisionHandler, self).__init__(f"{name}/collision_handler")


    def check_nei_range(self):

        agents = []
        for p in self.neighbors:
            dx, dy = p[0] - self.x, self.y - p[1]
            # if np.sqrt(dx ** 2 + dy ** 2) < 2 * self.robot_radius :
                # msg = f"[{self.name}] : collision handling with {p[-1]}"
                # console.info(console.red + msg + console.reset)
            if np.sqrt(dx ** 2 + dy ** 2) < self.__range * self.robot_radius:
                agents.append(p[:-1])

        return np.array(agents).T

    def update(self):
        neighbors = self.check_nei_range()
        # TODO implement velocity obstacle
        if self.robot.has_goal and len(neighbors) > 0:

            state = self.robot.state.copy()
            control = np.array(self.robot.control_input)
            state = np.hstack((state, control))

            dist, desiredV = self.robot.control()

            cmd = self.compute_velocity(state, neighbors, desiredV)
            # print('state', state, 'nei', neighbors.shape, 'cmd', cmd)
            if len(self.robot.goal_states) > 0:
                refined_control = (dist, cmd)
                self.robot.update(*refined_control)
                state = self.robot.state
                control = self.robot.control_input
                self.pub.set("/%s/state" % self.parent_name,
                             f"{state[0]},{state[1]},{state[2]},{control[0]},{control[1]}")
                # p = Point(state[0], state[1])
                # self.explorationTree.insert(p)
                return self.status.RUNNING

        # return self.status.SUCCESS if status else self.status.FAILURE
        # FIXME replace this status with RUNNING later
        return self.status.FAILURE

    def compute_velocity(self, robot, obstacles, v_desired):
        pA = robot[:2]
        vA = robot[-2:]
        # Compute the constraints
        # for each velocity obstacles
        number_of_obstacles = np.shape(obstacles)[1]
        Amat = np.empty((number_of_obstacles * 2, 2))
        bvec = np.empty((number_of_obstacles * 2))
        for i in range(number_of_obstacles):
            obstacle = obstacles[:, i]
            pB = obstacle[:2]
            vB = obstacle[2:]
            dispBA = pA - pB
            distBA = np.linalg.norm(dispBA)
            thetaBA = np.arctan2(dispBA[1], dispBA[0])
            if 2 * self.robot_radius > distBA:
                distBA = 2 * self.robot_radius
            phi_obst = np.arcsin(2 * self.robot_radius / distBA)
            phi_left = thetaBA + phi_obst
            phi_right = thetaBA - phi_obst

            # VO
            translation = vB
            Atemp, btemp = self.create_constraints(translation, phi_left, "left")
            Amat[i * 2, :] = Atemp
            bvec[i * 2] = btemp
            Atemp, btemp = self.create_constraints(translation, phi_right, "right")
            Amat[i * 2 + 1, :] = Atemp
            bvec[i * 2 + 1] = btemp

        # Create search-space
        th = np.linspace(0, 2 * np.pi, 50)
        vel = np.linspace(0, self.robot.max_lin_vel, 50)

        vv, thth = np.meshgrid(vel, th)

        vx_sample = (vv * np.cos(thth)).flatten()
        vy_sample = (vv * np.sin(thth)).flatten()

        v_sample = np.stack((vx_sample, vy_sample))

        v_satisfying_constraints = self.check_constraints(v_sample, Amat, bvec)

        if len(v_satisfying_constraints) < 1:
            #TODO don't sample from velocity obstacle cone
            self.robot.goal_states[0][0] = random.uniform(pA[0] - 2, pA[0] + 2)
            self.robot.goal_states[0][1] = random.uniform(pA[1] - 2, pA[1] + 2)

            msg = f"[{self.name}] : [modifying goal states] no v_satisfying_constraints {self.robot.goal_states}"
            console.info(console.red + msg + console.reset)
            # return np.array([-v_desired[0], -v_desired[1]])
            return np.array([-v_desired[0], 0])
        # Objective function
        size = np.shape(v_satisfying_constraints)[1]
        diffs = v_satisfying_constraints - \
                ((v_desired).reshape(2, 1) @ np.ones(size).reshape(1, size))
        norm = np.linalg.norm(diffs, axis=0)
        min_index = np.where(norm == np.amin(norm))[0][0]
        cmd_vel = (v_satisfying_constraints[:, min_index])

        return cmd_vel

    def check_constraints(self, v_sample, Amat, bvec):
        length = np.shape(bvec)[0]

        for i in range(int(length / 2) + 1):
            v_sample = self.check_inside(v_sample, Amat[2 * i:2 * i + 2, :], bvec[2 * i:2 * i + 2])

        return v_sample
    def check_inside(self, v, Amat, bvec):
        v_out = []
        if len(v) > 0:
            for i in range(np.shape(v)[1]):
                if not ((Amat @ v[:, i] < bvec).all()):
                    v_out.append(v[:, i])
        return np.array(v_out).T

    def create_constraints(self, translation, angle, side):
        # create line
        origin = np.array([0, 0, 1])
        point = np.array([np.cos(angle), np.sin(angle)])
        line = np.cross(origin, point)
        line = self.translate_line(line, translation)

        if side == "left":
            line *= -1

        A = line[:2]
        b = -line[2]

        return A, b

    def translate_line(self, line, translation):
        matrix = np.eye(3)
        matrix[2, :2] = -translation[:2]
        return matrix @ line
