# -*- coding: utf-8 -*-

import logging
import sys

from config import settings

def setup_logger(logger_name=None, logger_file=None):
    """ Generic logging setup, return logger object """
    
    # Get logger by name, or root logger if None
    logger = logging.getLogger(logger_name if logger_name else '')

    # Clear existing handlers to reconfigure
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(settings.LOGGING_LEVEL)
    formatter = logging.Formatter(settings.LOG_FORMAT, datefmt=settings.LOG_DATE_FORMAT)

    if logger_file:
        # Log to file
        file_handler = logging.FileHandler(logger_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Log to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Disable propagation to root logger
    logger.propagate = False

    return logger
