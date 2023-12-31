import py_trees
import py_trees.console as console

class Planner(py_trees.behaviour.Behaviour):
    def __init__(self, model, strategy, explorationTree, name):
        self.model = model
        self.strategy = strategy
        self.explorationTree = explorationTree
        super().__init__(f"{name}/planner")

    def update(self):
        # print("planning ...")
        self.strategy(exploration_tree=self.explorationTree)
        self.strategy.get(model=self.model)
        return self.status.SUCCESS