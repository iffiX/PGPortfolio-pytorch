##  A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem

### Introduction

Link to project [Video](https://www.youtube.com/watch?v=TuBabSVXCZI&ab_channel=Iffi)

Link to project [Slides](https://docs.google.com/presentation/d/1WagL1_1kufbpzJYGrKOdwAA4oAjFO3FmcLfOpastRAQ/edit?usp=sharing)

### Overview

1. How to use the framework.
2. About the Poloniex dataset.
3. Method of the framework (algorithm overview).
4. Code implementation.
   1. Understanding framework structure.
   2. Understanding the framework dataflow.
   3. Understanding the trainer implementation.
5. Future improvements.

### Usage

### Poloniex Dataset
Poloniex(founded in 2013) is a cryptocurrency exchange that allows for the buying or selling of digital assets, such as Bitcoin (BTC), Ethereum (ETH), TRON (TRX), and other altcoins.

Poloniex is no longer available for US users, margin trading was stopped in october 2018 with regards to US laws.

Steps to access and interact with the poloniex exchange are [here](https://docs.poloniex.com/#introduction).

### Method of the framework

### Code implementation

#### Framework structure

Bellow is a list of python files used in the framework, their functions are explained in brackets on the right side.

```
.
├── constants.py			(a list of constant values)
├── marketdata				(marketdata implements functions to get data from poloniex,
│	│   					 including downloading, cleaning, and converting to numpy)
│   ├── coin_data_manager.py    (implements cleaning and conversion)
│   ├── coin_list.py		    (implements functions to select coins with biggest volumes)
│   └── poloniex.py				(implements poloniex http API accessors)
├── nnagent					(nnagent implements things related to the nn agent)
│   ├── metrics.py				(all metrics used in training, including loss)
│   ├── network.py				(network auto constructor)
│   ├── replay_buffer.py		(combines PVM with coin features, like a pytorch dataset)
│   ├── rollingtrainer.py		(rollingtrainer for online trading, extends tradertrainer)
│   └── tradertrainer.py		(pytorch lightning module for agent training)
├── tdagent					(tdagent implements all traditional algorithms)
├── trade					(trade includes all trading tests, here only backtests)
│   └── backtest.py				(backtest implementation)
└── utils					(Various utility functions)
    ├── cache.py				(A cache for python class calculated attributes)
    ├── config.py				(config processor)
    ├── indicator.py			(indicator functions like APV, SR, MDD)
    ├── misc.py					(miscellaneous functions)
    └── plot.py					(plotting and result table generation)

```

#### Framework dataflow

![architecture](images/architecture.png)

The framework dataflow could be represented with the graph above, it took a lot of effort for us to clean up original code and rename variables to understandable names. 

##### Poloniex source

The bottom data source is Poloniex data API, which  pulls raw chart data for each coin and time period, a simple demonstration of the API implementation is below:

```
class Poloniex:
    """
    This class is designed to grab online data from https://poloniex.com/

    Currently only public commands are supported.
    Private commands not implemented.
    """
    def market_ticker(self):
        return self.api('returnTicker')

    def market_volume(self):
        return self.api('return24hVolume')

    def market_status(self):
        return self.api('returnCurrencies')

    def market_loans(self, coin):
        return self.api('returnLoanOrders', {'currency': coin})

    def market_orders(self, pair='all', depth=10):
        return self.api('returnOrderBook',
                        {'currencyPair': pair, 'depth': depth})
```

Class `CoinList` is used by class `CoinDataManager` as a sub component to determine coins to select when user gives the number of coins to trade in the `"input"` section of `config.json`:

```
"input":{
  "coin_number":11
 }
```

Then `CoinList` selects top 11 coins with the biggest volume using its `top_n_volume` method:

```
def top_n_volume(self, n=5, order=True, min_volume=0):
```

##### Local SQLIte cache

since it is quite expensive to download data from the beginning every time, our framework implements a SQL based cache to store all relevant information, and it also supports expanding to a longer period of time using previously downloaded data if coin number and feature number remain the same.

As a general rule of thumb, we call the method `get_coin_features` as a general way to get features as a 3 dimensional numpy array. In the future if any other data sources needed to added, they only have to provide the same data interface to merge into our architecture:

```
def get_coin_features(self, start, end, period=300, features=('close',))
```



Since it is kind of clumsy to construct `CoinDataManager` from config values by hand, we provide the helper function `coin_data_manager_init_helper` to help users initialize this class:

```
def coin_data_manager_init_helper(config, online=True, download=False, db_directory=None)
```

##### PGPBuffer

Class `PGPBuffer` implements all the details needed to same a batch of state (history observations, x and last weight vector, w), and the relative portfolio value vector (y), needed to train the agent. To initialize it, we also provide a helper function `buffer_init_helper` as:

```
def buffer_init_helper(config, device, online=True, db_directory=None)
```

Internally it registers to torch tensors `_pvm` and `_coin_features` as buffers so they can be moved with the module automatically, in order to create a sample, it first select a starting index using method `_sample_geometric`, which produce a random index according to the geometric distribution:

```
@staticmethod
    def _sample_geometric(start, end, bias):
        """
        Generate a index within [start, end) with geometric probability.

        Args:
            bias: A value in (0, 1).
        """
        ran = np.random.geometric(bias)
        while ran > end - start:
            ran = np.random.geometric(bias)
        result = end - ran
        return result
```

Finally it pack all samples and set it to the trainer with a callback function `setw`, which is used by the trainer module to update the portfolio vector memory:

```
    def _pack_samples(self, index):
        index = np.array(index)
        last_w = self._pvm[index - 1, :]

        def setw(w):
            assert t.is_tensor(w)
            self._pvm[index, :] = w.to(self._pvm.device).detach()

        batch = t.stack([
            self._coin_features[:, :, idx:idx + self._window_size + 1]
            for idx in index
        ])
        # features, [batch, feature, coin, time]
        X = batch[:, :, :, :-1]
        # price relative vector of the last period, [batch, norm_feature, coin]
        y = batch[:, :, :, -1] / batch[:, 0, None, :, -2]
        return {"X": X, "y": y, "last_w": last_w, "setw": setw}
```

Class `PGPDataset` is a general torch `IterableDataset`, so the pytorch lightning trainer could sample from it:

```
class PGPDataset(IterableDataset)
```

#### Trainer Implementation

Thanks to the power of the pytorch lightning, we can simplify our training procedure by just defining the following three methods of a `LightningModule`:

```
def __init__(self)
def training_step(self, batch, batch_index) -> t.tensor
def configure_optimizers(self)
```

This greatly improves the readability of our implementation, below is a simplification of our full implementation, but the gist is captured well:

```
class TraderTrainer(pl.LightningModule):
    def __init__(self, config, online=True, db_directory=None):
    	...
    def training_step(self, batch, _batch_idx):
        batch = batch[0]
        new_w = self._net(batch["X"], batch["last_w"])
        batch["setw"](new_w[:, 1:])
        return self._init_metrics()\
            .eval(batch["y"], batch["last_w"], new_w).loss
            
    def configure_optimizers(self):
        learning_rate = self._config["training"]["learning_rate"]
        decay_rate = self._config["training"]["decay_rate"]
        training_method = self._config["training"]["training_method"]

        optim_method = t.optim.Adam
        optim = optim_method([
            {"params": layer.parameters(),
             "weight_decay": layer_config.get("weight_decay") or 0}
            for layer, layer_config in
            zip(self._net.layers, self._config["layers"])
        ], lr=learning_rate)

        lr_sch = t.optim.lr_scheduler.ExponentialLR(optim, gamma=decay_rate)
        return {"optimizer": optim, "lr_scheduler": lr_sch}
```

Finally we just need to pass this `LightningModule` to the trainer, and completes the training, along with other objectives like saving checkpoints, testing, etc, in `main.py`

```
    if options.mode == "train":
        # delete old models
        shutil.rmtree(options.working_dir + "/model")
        config = load_config(options.config)
        save_config(config, options.working_dir + "/config.json")
        checkpoint_callback = ModelCheckpoint(
            dirpath=options.working_dir + "/model",
            filename="{epoch:02d}-{test_portfolio_value:.2f}",
            save_top_k=1,
            monitor="test_portfolio_value", mode="max",
            period=1, verbose=True
        )
        early_stopping = EarlyStopping(
            monitor="test_portfolio_value", mode="max"
        )
        t_logger = TensorBoardLogger(
            options.working_dir + "/log/tensorboard_log"
        )
        trainer = pl.Trainer(
            gpus=0 if options.device == "cpu" else options.device,
            callbacks=[checkpoint_callback, early_stopping],
            logger=[t_logger],
            limit_train_batches=1000,
            max_steps=config["training"]["steps"]
        )
        model = TraderTrainer(config,
                              online=not options.offline,
                              db_directory=options.working_dir + "/database")
        trainer.fit(model)
```



### Future improvements
