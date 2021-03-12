import datetime
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc

from pgportfolio.utils.indicator import \
    max_drawdown, sharpe, positive_count, negative_count, moving_accumulate
from pgportfolio.utils.misc import parse_time
from pgportfolio.trade.backtest import BackTest

# The dictionary of name of indicators mapping to the function of
# related indicators.
# Input is portfolio change vector, a ndarray.
INDICATORS = {"portfolio value(fAPV)": np.prod,
              "sharpe ratio(SR)": sharpe,
              "max drawdown(MDD)": max_drawdown,
              "positive periods": positive_count,
              "negative periods": negative_count,
              "postive day":
                  lambda pcs: positive_count(moving_accumulate(pcs, 48)),
              "negative day":
                  lambda pcs: negative_count(moving_accumulate(pcs, 48)),
              "postive week":
                  lambda pcs: positive_count(moving_accumulate(pcs, 336)),
              "negative week":
                  lambda pcs: negative_count(moving_accumulate(pcs, 336)),
              "average": np.mean}

NAMES = {"best": "Best Stock (Benchmark)",
         "crp": "UCRP (Benchmark)",
         "ubah": "UBAH (Benchmark)",
         "anticor": "ANTICOR",
         "olmar": "OLMAR",
         "pamr": "PAMR",
         "cwmr": "CWMR",
         "rmr": "RMR",
         "ons": "ONS",
         "up": "UP",
         "eg": "EG",
         "bk": "BK",
         "corn": "CORN",
         "m0": "M0",
         "wmamr": "WMAMR"}


def plot_backtest(config, algos, labels=None,
                  online=True,
                  working_directory="",
                  model_directory=None,
                  db_directory=None):
    """
    Args:
        config: config dictionary.
        algos: list of strings representing the name of algorithms.
        labels: labels used in the legend.
    """
    results = []
    for i, algo in enumerate(algos):
        b = BackTest(config,
                     agent_algorithm=algo,
                     online=online,
                     model_directory=model_directory,
                     db_directory=db_directory)
        b.trade()
        results.append(np.cumprod(b.test_pc_vector))

    start, end = _extract_test(config)
    timestamps = np.linspace(start, end, len(results[0]))
    dates = [
        datetime.datetime.fromtimestamp(
            int(ts) - int(ts) % config["input"]["global_period"]
        )
        for ts in timestamps
    ]

    weeks = mdates.WeekdayLocator()
    days = mdates.DayLocator()

    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"],
                  "size": 8})

    """
    styles = [("-", None), ("--", None), ("", "+"), (":", None),
              ("", "o"), ("", "v"), ("", "*")]
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 5)
    for i, pvs in enumerate(results):
        if len(labels) > i:
            label = labels[i]
        else:
            label = NAMES[algos[i]]
        ax.semilogy(dates, pvs, linewidth=1, label=label)
        # ax.plot(dates, pvs, linewidth=1, label=label)

    plt.ylabel("portfolio value $p_t/p_0$", fontsize=12)
    plt.xlabel("time", fontsize=12)
    xfmt = mdates.DateFormatter("%m-%d %H:%M")
    ax.xaxis.set_major_locator(weeks)
    ax.xaxis.set_minor_locator(days)
    datemin = dates[0]
    datemax = dates[-1]
    ax.set_xlim(datemin, datemax)

    ax.xaxis.set_major_formatter(xfmt)
    plt.grid(True)
    plt.tight_layout()
    ax.legend(loc="upper left", prop={"size": 10})
    fig.autofmt_xdate()

    file_path = working_directory + "/backtest_start_{}_end_{}.eps".format(
        dates[0].strftime("%Y-%m-%d"),
        dates[-1].strftime("%Y-%m-%d")
    )
    logging.info("Backtest saved to " + file_path)
    logging.info("Backtest time is from " +
                 str(dates[0]) + " to " + str(dates[-1]))
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.show()


def table_backtest(config, algos, labels=None, format="raw",
                   indicators=list(INDICATORS.keys()),
                   online=True,
                   working_directory="",
                   model_directory=None,
                   db_directory=None):
    """
    Args:
        config: config dictionary.
        algos: list of strings representing the name of algorithms
        or index of pgportfolio result.
        format: "raw", "html", "latex" or "csv". If it is "csv",
        the result will be save in a csv file. otherwise only
        print it out.
    Returns:
         A string of html or latex code.
    """
    results = []
    labels = list(labels)
    for i, algo in enumerate(algos):
        b = BackTest(config, agent_algorithm=algo,
                     online=online,
                     model_directory=model_directory,
                     db_directory=db_directory)
        b.trade()
        portfolio_changes = b.test_pc_vector

        indicator_result = {}
        for indicator in indicators:
            indicator_result[indicator] = \
                INDICATORS[indicator](portfolio_changes)
        results.append(indicator_result)
        if len(labels) <= i:
            labels.append(NAMES[algo])

    dataframe = pd.DataFrame(results, index=labels)

    start, end = _extract_test(config)
    start = datetime.datetime.fromtimestamp(
        start - start % config["input"]["global_period"]
    )
    end = datetime.datetime.fromtimestamp(
        end - end % config["input"]["global_period"]
    )

    file_path = working_directory + "/backtest_start_{}_end_{}.{}".format(
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        format
    )
    logging.info("Backtest saved to " + file_path)
    logging.info("Backtest time is from " + str(start) + " to " + str(end))
    if format == "html":
        with open(file_path, "w") as f:
            f.write(dataframe.to_html())
    elif format == "latex":
        with open(file_path, "w") as f:
            f.write(dataframe.to_latex())
    elif format == "raw":
        with open(file_path, "w") as f:
            f.write(dataframe.to_latex())
    elif format == "csv":
        dataframe.to_csv(file_path)
    else:
        raise ValueError("The format " + format + " is not supported")


def _extract_test(config):
    global_start = parse_time(config["input"]["start_date"])
    global_end = parse_time(config["input"]["end_date"])
    span = global_end - global_start
    start = global_end - config["input"]["test_portion"] * span
    end = global_end
    return start, end
