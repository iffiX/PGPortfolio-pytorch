import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.parent.absolute()

# by default, DATABASE_DIR is root_dir/database
DATABASE_DIR = str(ROOT_DIR.joinpath("database"))

# by default, CONFIG_FILE is root_dir/net_config.json
CONFIG_FILE = str(ROOT_DIR.joinpath("config.json"))

LAMBDA = 1e-4  # lambda in loss function 5 in training

PROXY_ADDR = ""

PROXY_PORT = None

# About time
NOW = 0
FIVE_MINUTES = 60 * 5
FIFTEEN_MINUTES = FIVE_MINUTES * 3
HALF_HOUR = FIFTEEN_MINUTES * 2
HOUR = HALF_HOUR * 2
TWO_HOUR = HOUR * 2
FOUR_HOUR = HOUR * 4
DAY = HOUR * 24
YEAR = DAY * 365
