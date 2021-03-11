import logging
import numpy as np
import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pgportfolio.nnagent.network import CNN
from pgportfolio.nnagent.metrics import Metrics
from pgportfolio.nnagent.replay_buffer import buffer_init_helper


class TraderTrainer(pl.LightningModule):
    def __init__(self, config, online=True, directory=None):
        """
        Args:
            config: config dictionary.
            directory: root of the database working directory.
        """
        super(TraderTrainer, self).__init__()

        # seed
        np.random.seed(config["random_seed"])
        t.random.manual_seed(config["random_seed"])

        # config shortcuts
        self._config = config
        self._train_config = config["training"]
        self._input_config = config["input"]

        # major components
        self._cdm, self._buffer = buffer_init_helper(
            config, self.device, online=online, directory=directory
        )
        self._net = CNN(self._input_config["feature_number"],
                        self._input_config["coin_number"],
                        self._input_config["window_size"],
                        config["layers"],
                        device=self.device)
        self._metrics = Metrics(self._config["trading"]["trading_consumption"],
                                self._train_config["loss_function"])
        self._test_set = self._buffer.get_test_set()
        if not self._train_config["fast_train"]:
            self._train_set = self._buffer.get_train_set()

        logging.info("Maximum portfolio value upperbound of test set: {}"
                     .format(t.prod(t.max(self._test_set[:, 0, :], dim=1)[0])))

    @property
    def test_set(self):
        # Used by backtest
        return self._test_sets

    @property
    def coins(self):
        # Used by backtest.
        return self._cdm.coins

    def decide_by_history(self, x, last_w):
        return self._net(x, last_w)

    def forward(self, x, last_w):
        return self._net(x, last_w)

    def train_dataloader(self):
        return DataLoader(dataset=self._buffer.get_train_dataset())

    def training_step(self, batch, _batch_idx):
        new_w = self._net(batch["X"], batch["last_w"])
        batch["setw"](new_w[:, 1:])
        self._metrics.begin_evaluate(batch["y"], batch["last_w"], new_w)
        return self._metrics.loss

    def training_step_end(self):
        # Manually test after each training step.
        fast_train = self._train_config["fast_train"]

        self._net.eval()
        new_w = self._net(self._test_set["X"], self._test_set["last_w"])
        self._test_set["setw"](new_w)
        m = self._metrics.begin_evaluate(self._test_set["y"],
                                         self._test_set["last_w"])

        if m.portfolio_value == 1.0:
            logging.info("Average portfolio weights {}"
                         .format(m.portfolio_weights.mean(axis=0)))

        self.log_dict({
            "test_portfolio_value": m.portfolio_value,
            "test_log_mean": m.log_mean,
            "test_loss": m.loss,
            "test_log_mean_free": m.log_mean_free,
        }, prog_bar=True)

        if not fast_train:
            new_w = self._net(self._train_set["X"],
                              self._train_set["last_w"])
            self._train_set["setw"](new_w)
            m = self._metrics.begin_evaluate(self._test_set["y"],
                                             self._test_set["last_w"])
            self.log("train_loss", m.loss, prog_bar=True)

        self._net.train()

    def configure_optimizers(self):
        learning_rate = self._config["training"]["learning_rate"]
        decay_rate = self._config["training"]["decay_rate"]
        training_method = self._config["training"]["training_method"]

        if training_method == "GradientDescent":
            optim_method = t.optim.SGD
        elif training_method == "Adam":
            optim_method = t.optim.Adam
        elif training_method == "RMSProp":
            optim_method = t.optim.RMSprop
        else:
            raise ValueError("Unknown training_method(optimizer): {}"
                             .format(training_method))

        optim = optim_method([
            {"params": layer.parameters(),
             "weight_decay": layer_config.get("weight_decay") or 0}
            for layer, layer_config in
            zip(self._net.layers, self._config["layers"])
        ], lr=learning_rate)

        lr_sch = t.optim.lr_scheduler.ExponentialLR(optim, gamma=decay_rate)
        return {"optimizer": optim, "lr_scheduler": lr_sch}
