import sqlite3
import logging
import numpy as np
import pandas as pd
from pgportfolio.marketdata.coin_list import CoinList
from pgportfolio.tools.data import panel_fillna
from pgportfolio.constants import *
from datetime import datetime


class CoinDataManager:
    # if offline ,the coin_list could be None
    # NOTE: return of the sqlite results is a list of tuples,
    # each tuple is a row.
    def __init__(self, coin_number, end, volume_average_days=1,
                 volume_forward=0, online=True):
        self._initialize_db()
        self._storage_period = FIVE_MINUTES  # keep this as 300
        self._coin_number = coin_number
        self._online = online
        if self._online:
            self._coin_list = CoinList(end, volume_average_days, volume_forward)
        self._volume_forward = volume_forward
        self._volume_average_days = volume_average_days
        self._coins = None

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

        time_index = pd.to_datetime(list(range(start, end + 1, period)),
                                    unit='s')
        panel = pd.Panel(items=features, major_axis=coins,
                         minor_axis=time_index, dtype=np.float32)

        connection = sqlite3.connect(DATABASE_DIR)
        try:
            for row_number, coin in enumerate(coins):
                for feature in features:
                    # NOTE: transform the start date to end date
                    if feature == "close":
                        sql = (
                            'SELECT date+300 AS date_norm, close '
                            'FROM History WHERE '
                            'date_norm>={start} and date_norm<={end} '
                            'and date_norm%{period}=0 and coin="{coin}" '
                            .format(start=start, end=end,
                                    period=period, coin=coin)
                        )
                    elif feature == "open":
                        sql = (
                            'SELECT date+{period} AS date_norm, open '
                            'FROM History WHERE '
                            'date_norm>={start} and date_norm<={end}'
                            'and date_norm%{period}=0 and coin="{coin}"'
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
                            'GROUP BY date_norm'
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
                            'GROUP BY date_norm'
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
                            'GROUP BY date_norm'
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
                    panel.loc[feature, coin, serial_data.index] = \
                        serial_data.squeeze()
                    panel = panel_fillna(panel, "both")
        finally:
            connection.commit()
            connection.close()
        return panel.values

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
            connection = sqlite3.connect(DATABASE_DIR)
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
        logging.debug("Selected coins are: " + str(coins))
        return coins

    def _initialize_db(self):
        with sqlite3.connect(DATABASE_DIR) as connection:
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

    def _update_data(self, start, end, coin):
        """
        Add new history data into the database.
        """
        connection = sqlite3.connect(DATABASE_DIR)
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
