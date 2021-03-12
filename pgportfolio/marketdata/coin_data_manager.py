import sqlite3
import logging
import numpy as np
import pandas as pd
from pgportfolio.marketdata.coin_list import CoinList
from pgportfolio.constants import *
from pgportfolio.utils.misc import parse_time, get_volume_forward, \
    get_feature_list
from datetime import datetime


class CoinDataManager:
    # if offline ,the coin_list could be None
    # NOTE: return of the sqlite results is a list of tuples,
    # each tuple is a row.
    def __init__(self, coin_number, end, volume_average_days=1,
                 volume_forward=0, online=True, db_directory=None):
        self._storage_period = FIVE_MINUTES  # keep this as 300
        self._coin_number = coin_number
        self._online = online
        if self._online:
            self._coin_list = CoinList(end, volume_average_days, volume_forward)
        self._volume_forward = volume_forward
        self._volume_average_days = volume_average_days
        self._coins = None
        self._db_dir = (db_directory or DATABASE_DIR) + "/data"
        self._initialize_db()

    @property
    def coins(self):
        return self._coins

    def get_coin_features(self, start, end, period=300, features=('close',)):
        """
        Args:
            start/end: Linux timestamp in seconds.
            period: Time interval of each data access point.
            features: Tuple or list of the feature names.

        Returns:
            A ndarray of shape [feature, coin, time].
        """
        from matplotlib import pyplot as plt
        start = int(start - (start % period))
        end = int(end - (end % period))
        coins = self.select_coins(
            start=end - self._volume_forward -
                  self._volume_average_days * DAY,
            end=end - self._volume_forward
        )
        self._coins = coins
        for coin in coins:
            self._update_data(start, end, coin)

        if len(coins) != self._coin_number:
            raise ValueError(
                "The length of selected coins %d is not equal to expected %d"
                % (len(coins), self._coin_number))

        logging.info("Feature type list is %s" % str(features))
        self._check_period(period)

        time_num = (end - start) // period + 1
        data = np.full([len(features), len(coins), time_num],
                       np.NAN, dtype=np.float32)

        connection = sqlite3.connect(self._db_dir)
        try:
            for coin_num, coin in enumerate(coins):
                for feature_num, feature in enumerate(features):
                    logging.info("Getting feature {} of coin {}".format(
                        feature, coin
                    ))
                    # NOTE: transform the start date to end date
                    if feature == "close":
                        sql = (
                            'SELECT date+300 AS date_norm, close '
                            'FROM History WHERE '
                            'date_norm>={start} and date_norm<={end} '
                            'and date_norm%{period}=0 and coin="{coin}" '
                            'ORDER BY date_norm'
                            .format(start=start, end=end,
                                    period=period, coin=coin)
                        )
                    elif feature == "open":
                        sql = (
                            'SELECT date+{period} AS date_norm, open '
                            'FROM History WHERE '
                            'date_norm>={start} and date_norm<={end}'
                            'and date_norm%{period}=0 and coin="{coin}" '
                            'ORDER BY date_norm'
                            .format(start=start, end=end,
                                    period=period, coin=coin)
                        )
                    elif feature == "volume":
                        sql = (
                            'SELECT date_norm, SUM(volume) '
                            'FROM (SELECT date+{period}-(date%{period}) '
                            'AS date_norm, volume, coin FROM History) '
                            'WHERE date_norm>={start} '
                            'and date_norm<={end} and coin="{coin}" '
                            'GROUP BY date_norm '
                            'ORDER BY date_norm'
                            .format(start=start, end=end,
                                    period=period, coin=coin)
                        )
                    elif feature == "high":
                        sql = (
                            'SELECT date_norm, MAX(high) '
                            'FROM (SELECT date+{period}-(date%{period}) '
                            'AS date_norm, high, coin FROM History) '
                            'WHERE date_norm>={start} '
                            'and date_norm<={end} and coin="{coin}" '
                            'GROUP BY date_norm '
                            'ORDER BY date_norm'
                            .format(start=start, end=end,
                                    period=period, coin=coin)
                        )
                    elif feature == "low":
                        sql = (
                            'SELECT date_norm, MIN(low) '
                            'FROM (SELECT date+{period}-(date%{period}) '
                            'AS date_norm, low, coin FROM History) '
                            'WHERE date_norm>={start} '
                            'and date_norm<={end} and coin="{coin}"'
                            'GROUP BY date_norm '
                            'ORDER BY date_norm'
                            .format(start=start, end=end,
                                    period=period, coin=coin)
                        )
                    else:
                        msg = ("The feature %s is not supported" % feature)
                        logging.error(msg)
                        raise ValueError(msg)
                    serial_data = pd.read_sql_query(sql, con=connection,
                                                    parse_dates=["date_norm"],
                                                    index_col="date_norm")
                    time_index = ((serial_data.index.astype(np.int64) // 10**9
                                   - start) / period).astype(np.int64)
                    data[feature_num, coin_num, time_index] = \
                        serial_data.values.squeeze()
        finally:
            connection.commit()
            connection.close()

        # use closing price to fill other features, since it is most stable
        data = self._fill_nan_and_invalid(data, bound=(0, 1),
                                          forward=True, axis=0)
        # backward fill along the period axis
        data = self._fill_nan_and_invalid(data, bound=(0, 1),
                                          forward=False, axis=2)
        assert not np.any(np.isnan(data)), "Filling nan failed, unknown error."

        # for manual checking
        # for f in range(data.shape[0]):
        #     for c in range(data.shape[1]):
        #         plt.plot(data[f, c])
        #         plt.show()
        return data

    def select_coins(self, start, end):
        """
        Select top coin_number of coins by volume from start to end.

        Args:
             start: Start timestamp in seconds.
             end: End timestamp in seconds.

        Returns:
            A list of coin name strings.
        """
        if not self._online:
            logging.info(
                "Selecting coins offline from %s to %s" %
                (datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M'),
                 datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M'))
            )
            connection = sqlite3.connect(self._db_dir)
            try:
                cursor = connection.cursor()
                cursor.execute(
                    'SELECT coin,SUM(volume) AS total_volume '
                    'FROM History WHERE '
                    'date>=? and date<=? GROUP BY coin '
                    'ORDER BY total_volume DESC LIMIT ?;',
                    (int(start), int(end), self._coin_number)
                )
                coins_tuples = cursor.fetchall()

                if len(coins_tuples) != self._coin_number:
                    logging.error("The sqlite error happened.")
            finally:
                connection.commit()
                connection.close()
            coins = []
            for tuple in coins_tuples:
                coins.append(tuple[0])
        else:
            coins = list(
                self._coin_list.top_n_volume(n=self._coin_number).index
            )
        logging.info("Selected coins are: " + str(coins))
        return coins

    def _initialize_db(self):
        with sqlite3.connect(self._db_dir) as connection:
            cursor = connection.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS History (date INTEGER,'
                           ' coin varchar(20), high FLOAT, low FLOAT,'
                           ' open FLOAT, close FLOAT, volume FLOAT, '
                           ' quoteVolume FLOAT, weightedAverage FLOAT,'
                           'PRIMARY KEY (date, coin));')
            connection.commit()

    @staticmethod
    def _check_period(period):
        if period == FIVE_MINUTES:
            return
        elif period == FIFTEEN_MINUTES:
            return
        elif period == HALF_HOUR:
            return
        elif period == TWO_HOUR:
            return
        elif period == FOUR_HOUR:
            return
        elif period == DAY:
            return
        else:
            raise ValueError(
                'peroid has to be 5min, 15min, 30min, 2hr, 4hr, or a day'
            )

    @staticmethod
    def _fill_nan_and_invalid(array, bound=(0, 1), forward=True, axis=-1):
        """
        Forward fill or backward fill nan values.
        See https://stackoverflow.com/questions/41190852
        /most-efficient-way-to-forward-fill-nan-values-in-numpy-array

        Basical idea is finding non-nan indexes, then use maximum.accumulate
        or minimum.accumulate to aggregate them
        """
        mask = np.logical_or(np.isnan(array),
                             np.logical_or(array < bound[0], array > bound[1]))

        index_shape = [1] * mask.ndim
        index_shape[axis] = mask.shape[axis]

        index = np.arange(mask.shape[axis]).reshape(index_shape)
        if forward:
            idx = np.where(~mask, index, 0)
            np.maximum.accumulate(idx, axis=axis, out=idx)
        else:
            idx = np.where(~mask, index, mask.shape[axis] - 1)
            idx = np.flip(
                np.minimum.accumulate(np.flip(idx, axis=axis),
                                      axis=axis, out=idx),
                axis=axis
            )
        return np.take_along_axis(array, idx, axis=axis)

    def _update_data(self, start, end, coin):
        """
        Add new history data into the database.
        """
        connection = sqlite3.connect(self._db_dir)
        try:
            cursor = connection.cursor()
            min_date = \
                cursor.execute('SELECT MIN(date) FROM History WHERE coin=?;',
                               (coin,)).fetchall()[0][0]
            max_date = \
                cursor.execute('SELECT MAX(date) FROM History WHERE coin=?;',
                               (coin,)).fetchall()[0][0]

            if min_date is None or max_date is None:
                self._fill_data(start, end, coin, cursor)
            else:
                if max_date + 10 * self._storage_period < end:
                    if not self._online:
                        raise Exception("Have to be online")
                    self._fill_data(max_date + self._storage_period,
                                    end,
                                    coin,
                                    cursor)
                if min_date > start and self._online:
                    self._fill_data(start,
                                    min_date - self._storage_period - 1,
                                    coin,
                                    cursor)

            # if there is no data
        finally:
            connection.commit()
            connection.close()

    def _fill_data(self, start, end, coin, cursor):
        duration = 7819200  # three months
        bk_start = start
        for bk_end in range(start + duration - 1, end, duration):
            self._fill_part_data(bk_start, bk_end, coin, cursor)
            bk_start += duration
        if bk_start < end:
            self._fill_part_data(bk_start, end, coin, cursor)

    def _fill_part_data(self, start, end, coin, cursor):
        chart = self._coin_list.get_chart_until_success(
            pair=self._coin_list.all_active_coins.at[coin, 'pair'],
            start=start,
            end=end,
            period=self._storage_period
        )
        logging.info(
            "Filling %s data from %s to %s" % (
                coin,
                datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M'),
                datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M')
            ))
        for c in chart:
            if c["date"] > 0:
                if c['weightedAverage'] == 0:
                    weightedAverage = c['close']
                else:
                    weightedAverage = c['weightedAverage']

                # NOTE here the USDT is in reversed order
                if 'reversed_' in coin:
                    cursor.execute(
                        'INSERT INTO History VALUES (?,?,?,?,?,?,?,?,?)',
                        (c['date'], coin, 1.0 / c['low'], 1.0 / c['high'],
                         1.0 / c['open'],
                         1.0 / c['close'], c['quoteVolume'], c['volume'],
                         1.0 / weightedAverage))
                else:
                    cursor.execute(
                        'INSERT INTO History VALUES (?,?,?,?,?,?,?,?,?)',
                        (c['date'], coin, c['high'], c['low'], c['open'],
                         c['close'], c['volume'], c['quoteVolume'],
                         weightedAverage))


def coin_data_manager_init_helper(config, online=True,
                                  download=False, db_directory=None):
    input_config = config["input"]
    start = parse_time(input_config["start_date"])
    end = parse_time(input_config["end_date"])
    cdm = CoinDataManager(
        coin_number=input_config["coin_number"],
        end=int(end),
        volume_average_days=input_config["volume_average_days"],
        volume_forward=get_volume_forward(
            int(end) - int(start),
            (input_config["validation_portion"] +
             input_config["test_portion"]),
            input_config["portion_reversed"]
        ),
        online=online,
        db_directory=db_directory
    )
    if not download:
        return cdm
    else:
        features = cdm.get_coin_features(
            start=start,
            end=end,
            period=input_config["global_period"],
            features=get_feature_list(input_config["feature_number"])
        )
        return cdm, features
