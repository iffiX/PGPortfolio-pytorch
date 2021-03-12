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

    def eval(self, y, last_w, net_output=None):
        self._y = y
        self._last_w = last_w
        self._net_output = net_output
        return self

    @cache(depend_attr=["_y", "_net_output"])
    def _price_relative_vector(self):
        # shape=[batch, coin+1]
        return t.cat([t.ones([self._y.shape[0], 1], device=self._y.device),
                      self._y[:, 0, :]], dim=1)

    @cache(depend_attr=["_y", "_net_output"])
    def _w_prime(self):
        # raw portfolio vector at the end of a period
        # and before the next period begins (take commission fees)
        # shape=[batch, coin+1]
        value = self._price_relative_vector * self._net_output
        return value / t.sum(value, dim=1).unsqueeze(-1)

    @cache(depend_attr=["_y", "_net_output"])
    def _consumption_vector(self):
        # consumption vector (on each periods)
        # there is also a recursive solution, see paper
        # shape=[batch-1]
        c = self._commission_ratio
        w_t = self._w_prime[:-1]  # rebalanced
        w_t1 = self._net_output[1:]
        mu = 1 - t.sum(t.abs(w_t1 - w_t), dim=1) * c
        return mu

    @cache(depend_attr=["_y", "_net_output"])
    def pv_vector(self):
        """
        Portfolio value vector.
        Each element is the time of portfolio value increase,
        p_t = p_0 * \Pi pv_i
        """
        # shape=[batch]
        cv = self._consumption_vector
        return (t.sum(self._net_output * self._price_relative_vector, dim=1) *
                (t.cat([t.ones([1], device=cv.device), cv], dim=0)))

    @cache(depend_attr=["_y", "_net_output"])
    def pv_mean(self):
        """Portfolio value increase mean """
        return t.mean(self.pv_vector)

    @cache(depend_attr=["_y", "_net_output"])
    def pv_std(self):
        """Portfolio value increase std"""
        return t.sqrt(t.mean((self.pv_vector - self.pv_mean) ** 2))

    @cache(depend_attr=["_net_output"])
    def portfolio_weights(self):
        return self._net_output

    @cache(depend_attr=["_y", "_net_output"])
    def portfolio_value(self):
        # portfolio value, assuming starting with portfolio value p_0 = 1.0
        return t.prod(self.pv_vector)

    @cache(depend_attr=["_y", "_net_output"])
    def sharpe_ratio(self):
        return (self.pv_mean - 1) / self.pv_std

    @cache(depend_attr=["_y", "_net_output"])
    def pv_log_mean(self):
        return t.mean(t.log(self.pv_vector))

    @cache(depend_attr=["_y", "_net_output"])
    def pv_log_mean_no_commission(self):
        # log mean without commission fee.
        return t.mean(t.log(
            t.sum(self._net_output * self._price_relative_vector, dim=1)
        ))

    @property
    def loss(self):
        return self._loss_function()

    def loss_function4(self):
        return -self.pv_log_mean_no_commission

    def loss_function5(self):
        # the term behind LAMBDA is log(1-softmax(X)),
        # it prefers more unevenly distributed portfolio weights
        return (-self.pv_log_mean_no_commission
                + LAMBDA * t.mean(t.sum(-t.log(1 + 1e-6 - self._net_output),
                                        dim=1)))

    def loss_function6(self):
        return -self.pv_log_mean

    def loss_function7(self):
        # the term behind LAMBDA is log(1-softmax(X)),
        # it prefers more unevenly distributed portfolio weights
        return (-self.pv_log_mean
                + LAMBDA * t.mean(t.sum(-t.log(1 + 1e-6 - self._net_output),
                                        dim=1)))

    def loss_function8(self):
        # the second term prefers the first cash (BTC) and
        # constraint the weight of other coins.
        return (-self.pv_log_mean_no_commission
                - t.sum(t.abs(self._net_output[:, 1:] - self._last_w)
                        * self._commission_ratio, dim=1))
