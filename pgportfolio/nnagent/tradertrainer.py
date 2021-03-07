import json
import os
import time
import collections
import logging
import numpy as np
import torch as t
import pytorch_lightning as pl
from pgportfolio.nnagent.nnagent import NNAgent
from pgportfolio.marketdata.coin_data_manager import CoinDataManager
from pgportfolio.nnagent.replay_buffer import PGPBuffer
from pgportfolio.utils.misc import parse_time, get_volume_forward, \
    get_feature_list


Result = collections.namedtuple("Result",
                                [
                                 "test_pv",
                                 "test_log_mean",
                                 "test_log_mean_free",
                                 "test_history",
                                 "config",
                                 "net_dir",
                                 "backtest_test_pv",
                                 "backtest_test_history",
                                 "backtest_test_log_mean",
                                 "training_time"])


class TraderTrainer(pl.LightningModule):
    def __init__(self,
                 config,
                 restore_dir=None,
                 save_path=None,
                 device="cpu"):
        """
        Args:
            config: config dictionary
            restore_dir: path to the model trained before
            save_path: path to save the model
            device: the device used to train the network
        """
        super(TraderTrainer, self).__init__()
        # variables
        self.best_metric = 0
        self.upperbound_validation = 1
        self.upperbound_test = 1

        # seed
        np.random.seed(config["random_seed"])
        t.random.manual_seed(config["random_seed"])

        # config shortcuts
        self._config = config
        self._train_config = config["training"]
        self._input_config = config["input"]
        self._save_path = save_path
        self._window_size = self._input_config["window_size"]
        self._coin_number = self._input_config["coin_number"]
        self._batch_size = self._train_config["batch_size"]
        self._snap_shot = self._train_config["snap_shot"]
        self._device = device

        # major components
        start = parse_time(self._input_config["start_date"])
        end = parse_time(self._input_config["end_date"])
        self._cdm = CoinDataManager(
            coin_number=self._input_config["coin_number"],
            end=int(end),
            volume_average_days=self._input_config["volume_average_days"],
            volume_forward=get_volume_forward(
                int(end) - int(start),
                (self._input_config["validation_portion"] +
                 self._input_config["test_portion"]),
                self._input_config["portion_reversed"]
            )
        )
        self._buffer = PGPBuffer(
            self._cdm.get_coin_features(
                start=start,
                end=end,
                period=self._input_config["global_period"],
                features=get_feature_list(self._input_config["feature_number"])
            ),
            batch_size=self._batch_size,
            window_size=self._window_size,
            test_portion=self._input_config["test_portion"],
            validation_portion=self._input_config["validation_portion"],
            sample_bias=self._train_config["buffer_biased"],
            portion_reversed=self._input_config["portion_reversed"],
            is_unordered=self._input_config["is_unordered"],
            device=device,
        )

        self._test_set = self._buffer.get_test_set()
        if not config["training"]["fast_train"]:
            self._training_set = self._buffer.get_training_set()
        self._agent = NNAgent(config, device)

    @staticmethod
    def calculate_upperbound(y):
        array = np.maximum.reduce(y[:, 0, :], 1)
        total = 1.0
        for i in array:
            total = total * i
        return total

    def check_abnormal(self, portfolio_value, weigths):
        if portfolio_value == 1.0:
            logging.info("average portfolio weights {}".format(weigths.mean(axis=0)))

    def __init_tensor_board(self, log_file_dir):
        tf.summary.scalar('benefit', self._agent.portfolio_value)
        tf.summary.scalar('log_mean', self._agent.log_mean)
        tf.summary.scalar('loss', self._agent.loss)
        tf.summary.scalar("log_mean_free", self._agent.log_mean_free)

    def train(self, log_file_dir="./tensorboard", index="0"):
        """
        :param log_file_dir: logging of the training process
        :param index: sub-folder name under train_package
        :return: the result named tuple
        """
        starttime = time.time()

        total_data_time = 0
        total_training_time = 0
        for i in range(self._train_config["steps"]):
            step_start = time.time()
            b = self._buffer.next_batch()
            finish_data = time.time()
            total_data_time += (finish_data - step_start)
            self._agent.train(b["X"], b["last_w"], b["y"], b["setw"])
            total_training_time += time.time() - finish_data
            if i % 1000 == 0 and log_file_dir:
                logging.info("average time for data accessing is %s"%(total_data_time/1000))
                logging.info("average time for training is %s"%(total_training_time/1000))
                total_training_time = 0
                total_data_time = 0
                self.log(i)

        if self._save_path:
            best_agent = NNAgent(self._config, restore_dir=self._save_path)
            self._agent = best_agent

        pv, log_mean = self._evaluate("test", self._agent.portfolio_value, self._agent.log_mean)
        logging.warning('the portfolio value train No.%s is %s log_mean is %s,'
                        ' the training time is %d seconds' % (index, pv, log_mean, time.time() - starttime))

        return self.__log_result_csv(index, time.time() - starttime)

    def log(self, step):
        fast_train = self._train_config["fast_train"]

        summary, v_pv, v_log_mean, v_loss, log_mean_free, weights= \
            self._evaluate("test", self.summary,
                           self._agent.portfolio_value,
                           self._agent.log_mean,
                           self._agent.loss,
                           self._agent.log_mean_free,
                           self._agent.portfolio_weights)
        self.test_writer.add_summary(summary, step)

        if not fast_train:
            summary, loss_value = self._evaluate("training", self.summary, self._agent.loss)
            self.train_writer.add_summary(summary, step)

        # print 'ouput is %s' % out
        logging.info('='*30)
        logging.info('step %d' % step)
        logging.info('-'*30)
        if not fast_train:
            logging.info('training loss is %s\n' % loss_value)
        logging.info('the portfolio value on test set is %s\nlog_mean is %s\n'
                     'loss_value is %3f\nlog mean without commission fee is %3f\n' % \
                     (v_pv, v_log_mean, v_loss, log_mean_free))
        logging.info('='*30+"\n")

        if not self._snap_shot:
            self._agent.save_model(self._save_path)
        elif v_pv > self.best_metric:
            self.best_metric = v_pv
            logging.info("get better model at %s steps,"
                         " whose test portfolio value is %s" % (step, v_pv))
            if self._save_path:
                self._agent.save_model(self._save_path)
        self.check_abnormal(v_pv, weights)

    def __log_result_csv(self, index, time):
        from pgportfolio.trade import backtest
        dataframe = None
        csv_dir = './train_package/train_summary.csv'
        tflearn.is_training(False, self._agent.session)
        v_pv, v_log_mean, benefit_array, v_log_mean_free =\
            self._evaluate("test",
                           self._agent.portfolio_value,
                           self._agent.log_mean,
                           self._agent.pv_vector,
                           self._agent.log_mean_free)

        backtest = backtest.BackTest(self._config.copy(),
                                     net_dir=None,
                                     agent=self._agent)

        backtest.start_trading()
        result = Result(test_pv=[v_pv],
                        test_log_mean=[v_log_mean],
                        test_log_mean_free=[v_log_mean_free],
                        test_history=[''.join(str(e)+', ' for e in benefit_array)],
                        config=[json.dumps(self._config)],
                        net_dir=[index],
                        backtest_test_pv=[backtest.test_pv],
                        backtest_test_history=[''.join(str(e)+', ' for e in backtest.test_pc_vector)],
                        backtest_test_log_mean=[np.mean(np.log(backtest.test_pc_vector))],
                        training_time=int(time))
        new_data_frame = pd.DataFrame(result._asdict()).set_index("net_dir")
        if os.path.isfile(csv_dir):
            dataframe = pd.read_csv(csv_dir).set_index("net_dir")
            dataframe = dataframe.append(new_data_frame)
        else:
            dataframe = new_data_frame
        if int(index) > 0:
            dataframe.to_csv(csv_dir)
        return result

