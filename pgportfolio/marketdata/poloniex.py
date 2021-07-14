import ssl
import socks
import time
import json
import logging
from pgportfolio import constants
from sockshandler import SocksiPyHandler
from urllib.request import HTTPSHandler, Request, build_opener
from urllib.parse import urlencode

minute = 60
hour = minute * 60
day = hour * 24
week = day * 7
month = day * 30
year = day * 365

# Possible Commands
PUBLIC_COMMANDS = ['returnTicker',
                   'return24hVolume', 'returnOrderBook', 'returnTradeHistory',
                   'returnChartData', 'returnCurrencies', 'returnLoanOrders']


class Poloniex:
    """
    This class is designed to grab online data from https://poloniex.com/

    Currently only public commands are supported.
    Private commands not implemented.
    """

    def __init__(self, api_key='', secret=''):
        """
        Args:
            api_key: Your API key to poloniex.
            secret: Your secret text to poloniex.
        """
        self.api_key = api_key.encode()
        self.secret = secret.encode()

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

    def market_chart(self, pair, period=day, start=None, end=None):
        start = start or time.time() - (week * 1)
        end = end or time.time()
        return self.api(
            'returnChartData',
            {'currencyPair': pair, 'period': period, 'start': start, 'end': end}
        )

    def market_trade_hist(self, pair):
        return self.api('returnTradeHistory', {'currencyPair': pair})

    def api(self, command, args=None):
        """
        Main API function.

        Returns:
            returns 'False' if invalid command or if no APIKey or Secret
            is specified (if command is "private").

            returns {"error":"<error message>"} if API error.
        """
        logging.info("Poloniex command: {}, args: {}".format(command, args))
        args = args or {}
        if command in PUBLIC_COMMANDS:
            url = 'https://poloniex.com/public?'
            args['command'] = command
            # prevent urllib from complaining when using a proxy
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            if constants.PROXY_ADDR == "" or constants.PROXY_PORT is None:
                opener = build_opener(HTTPSHandler(context=context))
            else:
                opener = build_opener(
                    SocksiPyHandler(socks.PROXY_TYPE_SOCKS5,
                                    constants.PROXY_ADDR,
                                    constants.PROXY_PORT,
                                    True, context=context))
            ret = opener.open(Request(url + urlencode(args)))
            return json.loads(ret.read().decode(encoding='UTF-8'))
        else:
            return False
