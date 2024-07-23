import numpy as np
import py_trees
import py_trees.console as console
from py_trees import common
from .data_compression import maximize_entropy_subset

class Learner(py_trees.behaviour.Behaviour):
    def __init__(self, robot, rng, model, evaluator, sensor, name, exp_logger, pub):
        self.robot = robot
        self.rng = rng
        self.model = model
        self.evaluator = evaluator
        self.sensor = sensor
        self.parent_name = name
        self.max_samples = 20
        self.exp_logger = exp_logger
        self.pub = pub
        super().__init__(f"{name}/learner")



    def update(self) -> common.Status:
        # print("learning ...")
        x_new = self.robot.commit_data()
        y_raw = self.sensor.sense(x_new, self.rng)
        y_new = y_raw.reshape(-1, 1)
        if len(y_new) > self.max_samples:
            msg = f"[{self.name}]: compressing data from original {len(y_new)} samples to {self.max_samples} samples"
            console.info(console.red + msg + console.reset)

            selected_index = maximize_entropy_subset(np.squeeze(y_new), self.max_samples)
            y_new = np.array([y_new[index] for index in selected_index if index is not None])
            x_new = np.array([x_new[index] for index in selected_index if index is not None])

        try:
            self.model.add_data(x_new, y_new)
            self.model.optimize(num_iter=len(y_new), verbose=False)
        except:
            return self.status.FAILURE
        mean, std, error = self.evaluator.eval_prediction(self.model)

        if self.exp_logger is not None:
            self.exp_logger.append(mean, std, error, x_new, y_new, self.model.num_train)

        rmse = np.sqrt(np.mean(error))
        msg = f"[{self.name}]:  gp = {np.mean(mean):.3f} +/- {np.mean(std):.3f} | err {rmse:.3f}"
        self.logger.debug(msg)
        console.info(console.cyan + f"[{self.name}]: {msg}" + console.reset)
        self.pub.set("/%s/rmse" % self.parent_name, f"{rmse}")

        return self.status.SUCCESS
