import logging
import logging.config
import os
from functools import wraps
from datetime import datetime

log_directory = os.path.join(os.getcwd(), "logs")
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(levelname)s|%(asctime)s|%(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "standard",
            "level": "INFO",
            "filename": os.path.join(log_directory, "trading_bot.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "trading_bot": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger("trading_bot")


def log_trade(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        symbol = kwargs.get('symbol')  # Get symbol from the arguments
        position_size = kwargs.get('position_size')
        entry_price = kwargs.get('stop_price')  # Entry price is the stop_price in the function args
        stop_loss = kwargs.get('stop_loss')
        take_profit = kwargs.get('take_profit')

        log_message = f'{symbol}|LONG|{entry_price}|{stop_loss}|{take_profit}|{position_size}'
        logger.info(log_message)

        return func(*args, **kwargs)
    return wrapper
