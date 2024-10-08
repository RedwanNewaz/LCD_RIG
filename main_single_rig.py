#!/home/redwan/anaconda3/envs/rig/bin/python
from pathlib import Path
from time import time
import pypolo
import numpy as np
import matplotlib.pyplot as plt

def get_sensor(args, env):
    sensor = pypolo.sensors.Ranger(
        rate=args.sensing_rate,
        env=env,
        env_extent=args.env_extent,
        noise_scale=args.noise_scale,
    )
    return sensor


def get_pilot_data(args, rng, sensor):
    bezier = pypolo.strategies.Bezier(task_extent=args.env_extent, rng=rng)
    x_init = bezier.get(num_states=args.num_init_samples)
    y_init = sensor.sense(states=x_init, rng=rng).reshape(-1, 1)
    return x_init, y_init


def get_robot(x_init, args):
    robot = pypolo.robots.USV(
        init_state=np.array([x_init[-1, 0], x_init[-1, 1], np.pi / 2]),
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
        task_extent=args.env_extent,
        eval_grid=args.eval_grid,
    )
    return evaluator


def get_strategy(args, rng, robot):
    """Get sampling strategy."""
    if args.strategy == "random":
        return pypolo.strategies.RandomSampling(
            task_extent=args.env_extent,
            rng=rng,
        )
    elif args.strategy == "active":
        return pypolo.strategies.ActiveSampling(
            task_extent=args.env_extent,
            rng=rng,
            num_candidates=args.num_candidates,
        )
    elif args.strategy == "myopic":
        return pypolo.strategies.MyopicPlanning(
            task_extent=args.env_extent,
            rng=rng,
            num_candidates=args.num_candidates,
            robot=robot,
        )
    elif args.strategy == "distributed":
        return pypolo.strategies.DistributedPlanning(
            task_extent=args.env_extent,
            rng=rng,
            num_candidates=args.num_candidates,
            robot=robot,
        )
    else:
        raise ValueError(f"Strategy {args.strategy} is not supported.")



def run(args, rng, model, strategy, sensor, evaluator, logger):

    def animate(x:np.ndarray):
        plt.cla()
        plt.imshow(sensor.env.matrix, cmap=plt.cm.gray, interpolation='nearest',
                         extent=sensor.env.extent)
        plt.scatter(x[:, 0], x[:, 1], c='r', alpha=0.4)
        plt.axis(args.env_extent)
        plt.pause(1e-2)

    while model.num_train < args.max_num_samples:
        x_new = strategy.get(model=model)
        y_new = sensor.sense(x_new, rng).reshape(-1, 1)
        # print(f"loc_shape = {x_new.shape} mes_shape = {y_new.shape}")
        animate(x_new)

        model.add_data(x_new, y_new)
        model.optimize(num_iter=len(y_new), verbose=False)
        mean, std, error = evaluator.eval_prediction(model)
        logger.append(mean, std, error, x_new, y_new, model.num_train)
        pypolo.experiments.utilities.print_metrics(model, evaluator)


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


def main():
    args = pypolo.experiments.argparser.parse_arguments()
    rng = pypolo.experiments.utilities.seed_everything(args.seed)
    data_path = "data/srtm"
    Path(data_path).mkdir(exist_ok=True, parents=True)
    env = pypolo.experiments.environments.get_environment(
        args.env_name, data_path)
    sensor = get_sensor(args, env)
    x_init, y_init = get_pilot_data(args, rng, sensor)
    robot = get_robot(x_init, args)
    model = get_model(args, x_init, y_init)
    evaluator = get_evaluator(args, sensor)
    logger = pypolo.experiments.Logger(evaluator.eval_outputs)
    mean, std, error = evaluator.eval_prediction(model)
    logger.append(mean, std, error, x_init, y_init, model.num_train)
    strategy = get_strategy(args, rng, robot)
    start = time()
    run(args, rng, model, strategy, sensor, evaluator, logger)
    end = time()
    save(args, evaluator, logger)
    print(f"Time used: {end - start:.1f} seconds")


if __name__ == "__main__":
    main()
