import torch as t
from pgportfolio.constants import *
from pgportfolio.utils.cache import cache


class Metrics:
    def __init__(self, commission_ratio, loss_function):
        self._commission_ratio = commission_ratio
        self._loss_function = getattr(self, loss_function)

        # price relative vector, shape=[batch, feature, coin]
        self._y = None
        self._last_w = None
        self._net_output = None

    def begin_evaluate(self, y, last_w, net_output=None):
        self._y = y
        self._last_w = last_w
        self._net_output = net_output
        return self

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
        """Portfolio value vector."""
        # shape=[batch, coin+1]
        cv = self._consumption_vector
        return (t.sum(self._net_output * self._future_price, dim=1) *
                (t.cat([t.ones([1, cv.shape[1]], device=cv.device), cv],
                       dim=0)))

    @cache(depend_attr=["_y", "_net_output"])
    def pv_mean(self):
        """Portfolio mean"""
        return t.mean(self.pv_vector)

    @cache(depend_attr=["_y", "_net_output"])
    def pv_std(self):
        """Portfolio std"""
        return t.sqrt(t.mean((self.pv_vector - self.pv_mean) ** 2))

    @cache(depend_attr=["_net_output"])
    def portfolio_weights(self):
        return self._net_output

    @cache(depend_attr=["_y", "_net_output"])
    def portfolio_value(self):
        return t.prod(self.pv_vector)

    @cache(depend_attr=["_y", "_net_output"])
    def sharpe_ratio(self):
        return (self.pv_mean - 1) / self.pv_std

    @cache(depend_attr=["_y", "_net_output"])
    def log_mean(self):
        return t.mean(t.log(self.pv_vector))

    @cache(depend_attr=["_y", "_net_output"])
    def log_mean_free(self):
        # log mean without commission fee.
        return t.mean(t.log(
            t.sum(self._net_output * self._future_price, dim=1)
        ))

    @property
    def loss(self):
        return self._loss_function()

    def loss_function4(self):
        return -t.mean(t.log(
            t.sum(self._net_output * self._future_price, dim=1)
        ))

    def loss_function5(self):
        return (-t.mean(t.log(t.sum(self._net_output * self._future_price,
                                    dim=1)))
                + LAMBDA * t.mean(t.sum(-t.log(1 + 1e-6 - self._net_output),
                                        dim=1)))

    def loss_function6(self):
        return -t.mean(t.log(self.pv_vector))

    def loss_function7(self):
        return (-t.mean(t.log(self.pv_vector))
                + LAMBDA * t.mean(t.sum(-t.log(1 + 1e-6 - self._net_output),
                                        dim=1)))

    def loss_function8(self):
        return (-t.mean(t.log(t.sum(self._net_output * self._future_price,
                                    dim=1)
                - t.sum(t.abs(self._net_output[:, 1:] - self._last_w)
                        * self._commission_ratio, dim=1))))
