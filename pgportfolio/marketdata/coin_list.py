import pandas as pd
import logging
from pgportfolio.marketdata.poloniex import Poloniex
from datetime import datetime
from pgportfolio.constants import *


class CoinList(object):
    def __init__(self, end, volume_average_days=1, volume_forward=0):
        self._polo = Poloniex()
        # connect the internet to access volumes
        logging.info("Checking market values.")
        vol = self._polo.market_volume()
        logging.info("Checking tickers.")
        ticker = self._polo.market_ticker()
        pairs = []
        coins = []
        volumes = []
        prices = []

        start_time = end - (DAY * volume_average_days) - volume_forward
        end_time = end - volume_forward
        logging.info(
            "selecting coin online from %s to %s"
            % (datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M'),
               datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M')))

        for k, v in vol.items():
            if k.startswith("BTC_") or k.endswith("_BTC"):
                pairs.append(k)
                for c, val in v.items():
                    if c != 'BTC':
                        if k.endswith('_BTC'):
                            coins.append('reversed_' + c)
                            prices.append(1.0 / float(ticker[k]['last']))
                        else:
                            coins.append(c)
                            prices.append(float(ticker[k]['last']))
                    else:
                        volumes.append(
                            self._get_total_volume(pair=k,
                                                   global_end=end,
                                                   days=volume_average_days,
                                                   forward=volume_forward)
                        )
        self._df = pd.DataFrame({'coin': coins, 'pair': pairs,
                                 'volume': volumes, 'price': prices})
        self._df = self._df.set_index('coin')

    @property
    def all_active_coins(self):
        return self._df

    @property
    def all_coins(self):
        return self._polo.market_status().keys()

    def top_n_volume(self, n=5, order=True, min_volume=0):
        if min_volume == 0:
            r = self._df.loc[self._df['price'] > 2e-6]
            r = r.sort_values(by='volume', ascending=False)[:n]
            print(r)
            if order:
                return r
            else:
                return r.sort_index()
        else:
            return self._df[self._df.volume >= min_volume]

    def get_chart_until_success(self, pair, start, period, end):
        is_connect_success = False
        chart = {}
        while not is_connect_success:
            try:
                chart = self._polo.market_chart(
                    pair=pair, start=int(start),
                    period=int(period), end=int(end)
                )
                is_connect_success = True
            except Exception as e:
                print(e)
        return chart

    # get several days volume
    def _get_total_volume(self, pair, global_end, days, forward):
        start = global_end - (DAY * days) - forward
        end = global_end - forward
        chart = self.get_chart_until_success(
            pair=pair, period=DAY, start=start, end=end
        )
        result = 0
        for one_day in chart:
            if pair.startswith("BTC_"):
                result += one_day['volume']
            else:
                result += one_day["quoteVolume"]
        return result


