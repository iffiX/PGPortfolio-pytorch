import logging
import numpy as np
import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pgportfolio.nnagent.network import CNN
from pgportfolio.nnagent.metrics import Metrics
from pgportfolio.nnagent.replay_buffer import buffer_init_helper


class TraderTrainer(pl.LightningModule):
    def __init__(self, config, online=True, db_directory=None):
        """
        Args:
            config: config dictionary.
            db_directory: root of the database working directory.
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
        logging.info("Setting up buffer.")

        # Note: self.device is "cpu" in init
        self._cdm, self._buffer = buffer_init_helper(
            config, self.device, online=online, db_directory=db_directory
        )
        logging.info("Setting up network and metrics.")
        self._net = CNN(self._input_config["feature_number"],
                        self._input_config["coin_number"],
                        self._input_config["window_size"],
                        config["layers"])

        logging.info("Setting up data.")

        self._register_set(self._buffer.get_test_set(), "test")
        if not self._train_config["fast_train"]:
            self._register_set(self._buffer.get_train_set(), "train")

        logging.info(
            "Maximum portfolio value upperbound of test set: {}"
            .format(
                t.prod(t.max(self._test_set_y[:, 0, :], dim=1)[0])
            ))

    @property
    def test_set(self):
        # Used by backtest
        return {
            "X": self._test_set_X,
            "y": self._test_set_y,
            "last_w": self._test_set_last_w,
            "setw": self._test_set_setw
        }

    @property
    def coins(self):
        # Used by backtest.
        return self._cdm.coins

    def decide_by_history(self, x, last_w_with_cash, **kwargs):
        # expects network on cpu
        with t.no_grad():
            return self._net(t.tensor(x, dtype=t.float32).unsqueeze(0),
                             t.tensor(last_w_with_cash[1:], dtype=t.float32)
                             .unsqueeze(0)).squeeze(0).numpy()

    def forward(self, x, last_w):
        return self._net(x, last_w)

    def train_dataloader(self):
        return DataLoader(dataset=self._buffer.get_train_dataset(),
                          collate_fn=lambda x: x)

    def training_step(self, batch, _batch_idx):
        batch = batch[0]
        new_w = self._net(batch["X"], batch["last_w"])
        batch["setw"](new_w[:, 1:])
        return self._init_metrics()\
            .eval(batch["y"], batch["last_w"], new_w).loss

    def training_step_end(self, training_output):
        # Manually test after each training step.
        # Note: _test_set_X, _test_set_y, etc. are registered buffers.
        fast_train = self._train_config["fast_train"]

        self._net.eval()
        new_w = self._net(self._test_set_X, self._test_set_last_w)
        self._test_set_setw(new_w[:, 1:])
        m = self._init_metrics().eval(self._test_set_y,
                                      self._test_set_last_w,
                                      new_w)

        if m.portfolio_value == 1.0:
            logging.error("Portfolio value is the same as start, "
                          "check input data, "
                          "average portfolio weights {}"
                          .format(m.pv_vector))

        self.log_dict({
            "test_portfolio_value": m.portfolio_value,
            "test_log_mean": m.pv_log_mean,
            "test_loss": m.loss,
            "test_log_mean_free": m.pv_log_mean_no_commission,
        }, prog_bar=True)

        if not fast_train:
            new_w = self._net(self._train_set_X,
                              self._train_set_last_w)
            self._train_set_setw(new_w[:, 1:])
            m = m.evaluate(self._train_set_y, self._train_set_last_w, new_w)
            self.log("train_loss", m.loss, prog_bar=True)

        self._net.train()
        return training_output

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

    def _register_set(self, set, name):
        for k, v in set.items():
            key = "_{}_set_{}".format(name, k)
            if t.is_tensor(v):
                self.register_buffer(key, v)
            else:
                setattr(self, key, v)

    def _init_metrics(self):
        return Metrics(self._config["trading"]["trading_consumption"],
                       self._train_config["loss_function"])
