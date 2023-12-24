#!/home/redwan/anaconda3/envs/rig/bin/python
from pathlib import Path
from time import time
import pypolo
import numpy as np
import matplotlib.pyplot as plt
import py_trees
from dev.bt_agent import Agent, Visualization
def get_sensor(args, env):
    sensor = pypolo.sensors.Ranger(
        rate=args.sensing_rate,
        env=env,
        env_extent=args.env_extent,
        noise_scale=args.noise_scale,
    )
    return sensor


def get_pilot_data(args, rng, sensor):
    bezier = pypolo.strategies.Bezier(task_extent=args.task_extent, rng=rng)
    x_init = bezier.get(num_states=args.num_init_samples)
    y_init = sensor.sense(states=x_init, rng=rng).reshape(-1, 1)
    return x_init, y_init


def get_robot(x_init, args):
    robot = pypolo.robots.USV(
        init_state=np.array([x_init[0], x_init[1], np.pi / 2]),
        control_rate=args.control_rate,
        max_lin_vel=args.max_lin_vel,
        tolerance=args.tolerance,
        sampling_rate=args.sensing_rate,
    )
    return robot


def get_model(args, x_init, y_init):
    kernel = pypolo.experiments.utilities.get_kernel(args)
    model = pypolo.models.GPR(
        x_train=x_init,
        y_train=y_init,
        kernel=kernel,
        noise=args.init_noise,
        lr_hyper=args.lr_hyper,
        lr_nn=args.lr_nn,
        jitter=args.jitter,
    )
    model.optimize(num_iter=model.num_train, verbose=False)
    return model


def get_evaluator(args, sensor):
    evaluator = pypolo.experiments.Evaluator(
        sensor=sensor,
        task_extent=args.task_extent,
        eval_grid=args.eval_grid,
    )
    return evaluator


def get_strategy(args, rng, robot):
    """Get sampling strategy."""
    if args.strategy == "random":
        return pypolo.strategies.RandomSampling(
            task_extent=args.task_extent,
            rng=rng,
        )
    elif args.strategy == "active":
        return pypolo.strategies.ActiveSampling(
            task_extent=args.task_extent,
            rng=rng,
            num_candidates=args.num_candidates,
        )
    elif args.strategy == "myopic":
        return pypolo.strategies.MyopicPlanning(
            task_extent=args.task_extent,
            rng=rng,
            num_candidates=args.num_candidates,
            robot=robot,
        )
    elif args.strategy == "distributed":
        return pypolo.strategies.DistributedPlanning(
            task_extent=args.task_extent,
            rng=rng,
            num_candidates=args.num_candidates,
            robot=robot,
        )
    else:
        raise ValueError(f"Strategy {args.strategy} is not supported.")



def run(args, agents, sensor):

    ####################
    # create tree
    ####################
    root = py_trees.composites.Parallel(
        name="MultiAgent",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()
    )


    viz = Visualization(args.task_extent, sensor)
    root.add_children(agents)

    task = py_trees.composites.Sequence("Sequence", True)
    task.add_children([root, viz])
    ####################
    # Tree Stewardship
    ####################
    # py_trees.logging.level = py_trees.logging.Level.INFO
    behaviour_tree = py_trees.trees.BehaviourTree(task)

    simStep = 0
    while simStep < args.max_num_samples:
        behaviour_tree.tick()
        simStep += 1



def save(args, evaluator, logger):
    print("Saving metrics and logged data......")
    experiment_id = "/".join([
        str(args.seed),
        args.env_name,
        args.strategy,
        args.kernel + args.postfix,
    ])
    save_dir = args.output_dir + experiment_id
    evaluator.save(save_dir)
    logger.save(save_dir)

def get_robots_init_locs(task_extent, N):
    """
        parameters
        ----------
            task_extent: bounding box for target area [xmin, xmax, ymin, ymax]
            N: number of robots
    """
    x_rand = np.random.uniform(task_extent[0], task_extent[1], N)
    y_rand = np.random.uniform(task_extent[2], task_extent[3], N)
    Xinits = np.column_stack([x_rand, y_rand])
    return Xinits


def get_gp_models(args, sensor, rng, Xinits):
    def isValidLocation(x):
        """
        @param x : sample location
        @return true if sample within the bounding box (task_extent)
        """
        xmin, xmax, ymin, ymax = args.task_extent
        return x[0] >= xmin and x[0] < xmax and x[1] >= ymin and x[1] < ymax

    bezier = pypolo.strategies.Bezier(task_extent=args.task_extent, rng=rng)

    models = []
    for x0 in Xinits:
        x_samples = bezier.get(num_states=args.num_init_samples)
        # add random initial location of robot
        x_init = np.array([ x + x0 for x in x_samples if isValidLocation(x + x0)])
        y_init = sensor.sense(states=x_init, rng=rng).reshape(-1, 1)
        # construct gp model with initial samples
        model = get_model(args, x_init, y_init)
        models.append(model)

    return models

def main():
    args = pypolo.experiments.argparser.parse_arguments()
    rng = pypolo.experiments.utilities.seed_everything(args.seed)
    data_path = "data/srtm"
    Path(data_path).mkdir(exist_ok=True, parents=True)
    env = pypolo.experiments.environments.get_environment(
        args.env_name, data_path)
    sensor = get_sensor(args, env)
    num_agents = 3
    Xinits = get_robots_init_locs(args.task_extent, num_agents)
    gpModels = get_gp_models(args, sensor, rng, Xinits)
    robots = [get_robot(x_init, args) for x_init in Xinits]
    agents = []
    for i in range(num_agents):
        robot = robots[i]
        model = gpModels[i]
        evaluator = get_evaluator(args, sensor)
        strategy = get_strategy(args, rng, robot)
        agent = Agent(rng, model, strategy, sensor, evaluator, i + 1)
        agents.append(agent)

    start = time()
    run(args, agents, sensor)
    end = time()
    # save(args, evaluator, logger)
    print(f"Time used: {end - start:.1f} seconds")


if __name__ == "__main__":
    main()
