import torch as t
from pgportfolio.constants import *
from pgportfolio.tools.cache import cache
import pgportfolio.learn.network as network


class NNAgent:
    def __init__(self, config, device="cpu"):
        self._config = config
        self._device = device
        self._commission_ratio = self._config["trading"]["trading_consumption"]

        self.net = network.CNN(config["input"]["feature_number"],
                               config["input"]["coin_number"],
                               config["input"]["window_size"],
                               config["layers"],
                               device=device)
        self.loss_function = None
        self._optim = None
        self._lr_sch = None
        self._decay_steps = None
        self._train_step = 0

        # price relative vector, shape=[batch, feature, coin]
        self._y = None
        self._last_w = None
        self._net_output = None
        self._init_train(
            learning_rate=config["training"]["learning_rate"],
            decay_steps=config["training"]["decay_steps"],
            decay_rate=config["training"]["decay_rate"],
            training_method=config["training"]["training_method"]
        )

    @cache(depend_attr=["_y", "_net_output"])
    def _future_price(self):
        # shape=[batch, coin+1]
        return t.cat([t.ones([self._y.shape[0], 1], device=self._y.device),
                      self._y[:, 0, :]], dim=2)

    @cache(depend_attr=["_y", "_net_output"])
    def _future_omega(self):
        # shape=[batch, coin+1]
        value = self._future_price * self._net_output
        return value / t.sum(value, dim=1).unsqueeze(-1)

    @cache(depend_attr=["_y", "_net_output"])
    def _consumption_vector(self):
        # consumption vector (on each periods)
        # there is also a recursive solution, see paper
        # shape=[batch-1, coin+1]
        c = self._commission_ratio
        w_t = self._future_omega[:-1]  # rebalanced
        w_t1 = self._net_output[1:]
        mu = 1 - t.sum(t.abs(w_t1 - w_t), dim=1) * c
        return mu

    @cache(depend_attr=["_y", "_net_output"])
    def pv_vector(self):
        # shape=[batch, coin+1]
        cv = self._consumption_vector
        return (t.sum(self._net_output * self._future_price, dim=1) *
                (t.cat([t.ones([1, cv.shape[1]], device=cv.device), cv],
                       dim=0)))

    @cache(depend_attr=["_y", "_net_output"])
    def pv_mean(self):
        return t.mean(self.pv_vector)

    @cache(depend_attr=["_y", "_net_output"])
    def pv_std(self):
        return t.sqrt(t.mean((self.pv_vector - self.pv_mean) ** 2))

    @cache(depend_attr=["_net_output"])
    def portfolio_weights(self):
        return self._net_output

    @cache(depend_attr=["_y", "_net_output"])
    def portfolio_value(self):
        return t.prod(self.pv_vector)

    @cache(depend_attr=["_y", "_net_output"])
    def sharp_ratio(self):
        return (self.pv_mean - 1) / self.pv_std

    @cache(depend_attr=["_y", "_net_output"])
    def log_mean(self):
        return t.mean(t.log(self.pv_vector))

    @cache(depend_attr=["_y", "_net_output"])
    def log_mean_free(self):
        return t.mean(t.log(
            t.sum(self._net_output * self._future_price, dim=1)
        ))

    def begin_evaluate(self, x, last_w, y=None, setw=None):
        assert not t.any(t.isnan(x))
        assert not t.any(t.isnan(y))
        assert not t.any(t.isnan(last_w)), "the last_w is {}".format(last_w)
        new_w = self.net(x.to(self._device), last_w.to(self._device))

        # let attribute evaluations depended on y fail if y is not provided.
        self._y = y.to(self._device)
        self._last_w = last_w
        self._net_output = new_w
        if setw is not None:
            setw(new_w[:, 1:])
        return self

    def train_on(self, x, last_w, y, setw):
        loss = self.begin_evaluate(x, last_w, y, setw).loss_function()
        self.net.zero_grad()
        loss.backward()
        self._optim.step()
        self._train_step += 1
        if self._train_step % self._decay_steps == 0:
            self._lr_sch.step()

    def _init_train(self,
                    learning_rate,
                    decay_steps,
                    decay_rate,
                    training_method):
        # TODO: weight decay per parameter.

        self._decay_steps = decay_steps
        if training_method == 'GradientDescent':
            self._optim = t.optim.SGD(self.net.parameters(), lr=learning_rate)
        elif training_method == 'Adam':
            self._optim = t.optim.Adam(self.net.parameters(), lr=learning_rate)
        elif training_method == 'RMSProp':
            self._optim = t.optim.RMSprop(self.net.parameters(),
                                          lr=learning_rate)
        else:
            raise ValueError("Unknown training_method(optimizer): {}"
                             .format(training_method))

        self._lr_sch = t.optim.lr_scheduler.ExponentialLR(self._optim,
                                                          gamma=decay_rate)

        def loss_function4():
            return -t.mean(t.log(
                t.sum(self._net_output * self._future_price, dim=1)
            ))

        def loss_function5():
            return (-t.mean(t.log(t.sum(self._net_output * self._future_price,
                                        dim=1)))
                    + LAMBDA * t.mean(t.sum(-t.log(1 + 1e-6 - self._net_output),
                                            dim=1)))

        def loss_function6():
            return -t.mean(t.log(self.pv_vector))

        def loss_function7():
            return (-t.mean(t.log(self.pv_vector))
                    + LAMBDA * t.mean(t.sum(-t.log(1 + 1e-6 - self._net_output),
                                            dim=1)))

        def with_last_w():
            return (-t.mean(t.log(t.sum(self._net_output * self._future_price,
                                        dim=1)
                    - t.sum(t.abs(self._net_output[:, 1:] - self._last_w)
                            * self._commission_ratio, dim=1))))

        loss_function = loss_function5
        if self._config["training"]["loss_function"] == "loss_function4":
            loss_function = loss_function4
        elif self._config["training"]["loss_function"] == "loss_function5":
            loss_function = loss_function5
        elif self._config["training"]["loss_function"] == "loss_function6":
            loss_function = loss_function6
        elif self._config["training"]["loss_function"] == "loss_function7":
            loss_function = loss_function7
        elif self._config["training"]["loss_function"] == "loss_function8":
            loss_function = with_last_w
        self.loss_function = loss_function
