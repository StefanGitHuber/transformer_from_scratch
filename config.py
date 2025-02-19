# -*- coding: utf-8 -*-

import logging
from pathlib import Path

# Base configuration
class Config:

    NAME_PROJECT = f'Transformers from scratch'
    NAME_MODELS = f'TransformersModel'
    NAME_DATASET = f'<younameit>'

    # Configuration - File paths
    DIR_PATH = Path(f"/path/to/your/data/")
    DIR_LOG = DIR_PATH / "logs"
    DIR_MODELS = DIR_PATH / "models"
    DIR_DATA = DIR_PATH / "data_train"

    # Configuration - Logging default
    LOGGING_LEVEL = logging.INFO
    LOGGER_FILE = DIR_LOG / "logging.log"
    LOGGER_NAME = NAME_PROJECT
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    WANNABEE = True
    MLFLOW = True
    
    # Configuration - Training
    BATCH_SIZE = 64
    NUM_WORKERS = 2
    NUM_EPOCHS = 100
    MAX_SEQ_LEN = 100
    EARLY_STOPPING_PERCENT = 5 # Less then xx % triggers early stopping of training loop
    EPOCH_STORE_INTERVAL = 4
    LEARNING_RATE = 0.0001  # ADAM simple lr without warming up
    BETA1 = 0.9                      # ADAM 1st order movement, exp delay
    BETA2 = 0.98                    # ADAM 2nd order movement, exp delay
    SHUFFLE = True
    TRACE_VRAM = True
    TRAIN_NOT_EVAL = True
    FILE_PATTERN = "*.txt"

    # Configuration - Model
    D_FF = 2048
    D_MODEL = 256
    DROPOUT = 0.1
    NUM_HEADS = 8
    NUM_LAYERS = 6

    # Memory and Buffer Configuration
    BUFFER_SIZE = 2 * 1024 * 1024            # 2 MB buffer size
    MIN_BUFFER_SIZE = 4096                     # Minimum 4KB buffer
    MAX_BUFFER_SIZE = 1024 * 1024 * 10  # Maximum 10MB buffer
    INITIAL_BUFFER_MULTIPLIER = 2            # Buffer should contain at least 2 * max_len tokens

# Instantiate the configuration
settings = Config()

# Pre-initialize paths
settings.DIR_LOG.mkdir(exist_ok=True)
settings.DIR_DATA.mkdir(exist_ok=True)
