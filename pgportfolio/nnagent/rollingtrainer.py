from pgportfolio.nnagent.tradertrainer import TraderTrainer


class RollingTrainer(TraderTrainer):
    def __init__(self, config, **kwargs):
        config = config.copy()
        config["training"]["buffer_biased"] = config["trading"]["buffer_biased"]
        config["training"]["learning_rate"] = config["trading"]["learning_rate"]
        TraderTrainer.__init__(self, config, **kwargs)
