import logging
import os
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pgportfolio import constants
from pgportfolio.marketdata.coin_data_manager import \
    coin_data_manager_init_helper
from pgportfolio.trade.backtest import BackTest
from pgportfolio.nnagent.tradertrainer import TraderTrainer
from pgportfolio.utils.config import load_config, save_config
from pgportfolio.utils import plot


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode", dest="mode",
                        help="start mode, train, generate, download_data "
                             "backtest",
                        metavar="MODE", default="train")
    parser.add_argument("--algos",
                        help='algo names, seperated by ","',
                        dest="algos")
    parser.add_argument("--labels", dest="labels",
                        help="names that will shown in the figure caption "
                             "or table header")
    parser.add_argument("--format", dest="format", default="raw",
                        help="format of the table printed")
    parser.add_argument("--device", dest="device", default="cpu",
                        help="device to be used to train")
    parser.add_argument("--working_dir", dest="working_dir",
                        default=constants.ROOT_DIR,
                        help="Working directory, by default it is project root")
    parser.add_argument("--config", dest="config",
                        default=constants.CONFIG_FILE,
                        help="Config file, by default it is "
                             "config.json")
    return parser


def prepare_directories(root):
    if not os.path.exists(root + "/database"):
        os.makedirs(root + "/database")
    if not os.path.exists(root + "/log"):
        os.makedirs(root + "/log")
        os.makedirs(root + "/log/tensorboard_log")
    if not os.path.exists(root + "/model"):
        os.makedirs(root + "/model")
    if not os.path.exists(root + "/result"):
        os.makedirs(root + "/result")


def main():
    parser = build_parser()
    options = parser.parse_args()
    prepare_directories(options.working_dir)

    if options.mode == "train":
        config = load_config(options.config)
        save_config(config, options.working_dir + "/config.json")
        checkpoint_callback = ModelCheckpoint(
            dirpath=options.working_dir + "/model",
            filename="{epoch:02d}-{val_loss:.2f}.ckpt",
            save_top_k=10,
            monitor="test_portfolio_value", mode="max",
            period=1, verbose=True
        )
        t_logger = TensorBoardLogger(
            options.working_dir + "/log/tensorboard_log"
        )
        trainer = pl.Trainer(
            weights_save_path=options.working_dir + "/log",
            gpus=0 if options.device == "cpu" else options.device,
            callbacks=[checkpoint_callback],
            logger=[t_logger],
            limit_train_batches=100,
            max_steps=config["training"]["steps"]
        )
        model = TraderTrainer(config, options.working_dir + "/database")
        trainer.fit(model)

    elif options.mode == "download_data":
        config = load_config(options.config)
        coin_data_manager_init_helper(
            config, download=True, directory=options.working_dir + "database"
        )
    elif options.mode == "backtest":
        logging.basicConfig(level=logging.INFO)
        config = load_config(options.config)
        save_config(config, options.working_dir + "/config.json")
        algos = options.algos.split(",")
        backtests = [BackTest(config, agent_algorithm=algo)
                     for algo in algos]
        for b in backtests:
            b.trade()
    elif options.mode == "save_test_data":
        # This is used to export the test data
        config = load_config(options.config)
        backtest = BackTest(config, agent_algorithm="not_used")
        with open(options.working_dir + "test_data.csv", 'wb') as f:
            np.savetxt(f, backtest.test_data.T, delimiter=",")
    elif options.mode == "plot":
        logging.basicConfig(level=logging.INFO)
        config = load_config(options.config)
        algos = options.algos.split(",")
        if options.labels:
            labels = options.labels.replace("_", " ")
            labels = labels.split(",")
        else:
            labels = algos
        plot.plot_backtest(config, algos, labels)
    elif options.mode == "table":
        logging.basicConfig(level=logging.INFO)
        config = load_config(options.config)
        algos = options.algos.split(",")
        if options.labels:
            labels = options.labels.replace("_", " ")
            labels = labels.split(",")
        else:
            labels = algos
        plot.table_backtest(config, algos, labels, format=options.format)


if __name__ == "__main__":
    main()
