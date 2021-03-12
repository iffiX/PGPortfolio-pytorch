## A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem

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



#### Trainer Implementation

### Future improvements
