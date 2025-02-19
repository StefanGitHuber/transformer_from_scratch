#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# source /opt/intel/oneapi/setvars.sh
# source /opt/intel/oneapi/mkl/latest/env/vars.sh
# source /opt/intel/oneapi/compiler/latest/env/vars.sh

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import mlflow
import mlflow.pytorch
import os
import tiktoken
import wandb

try:
    # Intel Arc GPUs
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

# Module-level imports
from config import settings
from device_handling import device_detect
from logger_setup import setup_logger
from memory_monitor import print_vram_usage
from token_dataloader import token_dataloader
from transformer_core import Transformer
from transformer_train_infer import load_epoch_model, run_inference, train_transformer

###### Logging MLflow setup
if settings.MLFLOW:
    mlflow.set_experiment(settings.NAME_PROJECT) # Set or create MLflow experiment
    # Run bash command first
    # mlflow server --host 127.0.0.1 --port 5000
    # if tracking+visualizing project on localhost
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")

######  Logging wannabee
if settings.WANNABEE:
    wandb.init(
        # Wandb tracking: Set project to run logging
        project=settings.NAME_PROJECT,
    
        # Track hyperparameters + metadata
        config={
            "name_project": settings.NAME_PROJECT,
            "name_models": settings.NAME_MODELS,
            "name_dataset": settings.NAME_DATASET,
            "dir_path": settings.DIR_PATH,
            "batch_size": settings.BATCH_SIZE,
            "num_workers": settings.NUM_WORKERS,
            "num_epochs": settings.NUM_EPOCHS,
            "max_seq_len": settings.MAX_SEQ_LEN,
            "epoch_store_interval": settings.EPOCH_STORE_INTERVAL,
            "learning_rate": settings.LEARNING_RATE,
            "beta1": settings.BETA1,
            "beta2": settings.BETA2,
            "shuffle": settings.SHUFFLE,
            "trace_vram": settings.TRACE_VRAM,
            "train_not_eval": settings.TRAIN_NOT_EVAL,
            "file_pattern": settings.FILE_PATTERN,
            "d_ff": settings.D_FF,
            "d_model": settings.D_MODEL,
            "dropout": settings.DROPOUT,
            "num_heads": settings.NUM_HEADS,
            "num_layers": settings.NUM_LAYERS,
            "initial_buffer_multiplier": settings.INITIAL_BUFFER_MULTIPLIER,
            "buffer_size": settings.BUFFER_SIZE,
            "min_buffer_size": settings.MIN_BUFFER_SIZE,
            "max_buffer_size": settings.MAX_BUFFER_SIZE,
            "early_stopping_percent": settings.EARLY_STOPPING_PERCENT,
            "batch_size": settings.BATCH_SIZE,
            "max_seq_length": settings.MAX_SEQ_LEN,    
        }
    )

###### Logging standard
logger = setup_logger(__name__, settings.LOGGER_FILE)

###### Tokenizer and DataLoader

# GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
src_vocab_size = tgt_vocab_size = tokenizer.n_vocab # Max vocab size given tokenizer (old max token 50251) 

# Dataloader with token-based memory optimization
data_loader = token_dataloader(
    tokenizer=tokenizer,
    dir_data=settings.DIR_DATA,
    max_seq_length=settings.MAX_SEQ_LEN,
    batch_size=settings.BATCH_SIZE,
    buffer_size=settings.BUFFER_SIZE,
    file_pattern=settings.FILE_PATTERN,
    shuffle=settings.SHUFFLE,
    num_workers=settings.NUM_WORKERS
)

###### Model initialization
print_vram_usage(settings.TRACE_VRAM, "GPU VRAM before model initialization:")

# Initialize Transformer model
transformer = Transformer(src_vocab_size, tgt_vocab_size, settings.D_MODEL, settings.NUM_HEADS, settings.NUM_LAYERS, settings.D_FF, settings.MAX_SEQ_LEN, settings.DROPOUT)

# Define criterion and optimizer
optimizer = optim.Adam(transformer.parameters(), lr=settings.LEARNING_RATE, betas=(settings.BETA1, settings.BETA2), eps=1e-9)

###### Device handling
device_str, device_obj = device_detect()
logger.info(f"Using device type: {device_str}")
transformer = transformer.to(device_str)

# Optimize with IPEX on Intel Arc GPU
transformer, optimizer = ipex.optimize(transformer, optimizer=optimizer)
logger.info(f"Intel Arc GPU with IPEX version {ipex.__version__}")
print_vram_usage(settings.TRACE_VRAM, "GPU VRAM after model initialization:")

###### Load or train transformer
num_last_epoch = load_epoch_model(transformer, optimizer, settings)
if settings.TRAIN_NOT_EVAL:
    if settings.MLFLOW:
        ###### Log hyperparameters to MLflow + start MLflow run
        with mlflow.start_run(run_name=settings.NAME_PROJECT):
            mlflow.log_param("early_stopping_percent", settings.EARLY_STOPPING_PERCENT)
            mlflow.log_param("name_project", settings.NAME_PROJECT)
            mlflow.log_param("name_models", settings.NAME_MODELS)
            mlflow.log_param("name_dataset", settings.NAME_DATASET)
            mlflow.log_param("dir_path", settings.DIR_PATH)
            mlflow.log_param("batch_size", settings.BATCH_SIZE)
            mlflow.log_param("num_workers", settings.NUM_WORKERS)
            mlflow.log_param("num_epochs", settings.NUM_EPOCHS)
            mlflow.log_param("max_seq_len", settings.MAX_SEQ_LEN)
            mlflow.log_param("epoch_store_interval", settings.EPOCH_STORE_INTERVAL)
            mlflow.log_param("learning_rate", settings.LEARNING_RATE)
            mlflow.log_param("beta1", settings.BETA1)
            mlflow.log_param("beta2", settings.BETA2)
            mlflow.log_param("shuffle", settings.SHUFFLE)
            mlflow.log_param("trace_vram", settings.TRACE_VRAM)
            mlflow.log_param("train_not_eval", settings.TRAIN_NOT_EVAL)
            mlflow.log_param("file_pattern", settings.FILE_PATTERN)
            mlflow.log_param("d_ff", settings.D_FF)
            mlflow.log_param("d_model", settings.D_MODEL)
            mlflow.log_param("dropout", settings.DROPOUT)
            mlflow.log_param("num_heads", settings.NUM_HEADS)
            mlflow.log_param("num_layers", settings.NUM_LAYERS)
            mlflow.log_param("initial_buffer_multiplier", settings.INITIAL_BUFFER_MULTIPLIER)
            mlflow.log_param("buffer_size", settings.BUFFER_SIZE)
            mlflow.log_param("min_buffer_size", settings.MIN_BUFFER_SIZE)
            mlflow.log_param("max_buffer_size", settings.MAX_BUFFER_SIZE)

            logger.info(f"Train transformer from scratch or continue after epoch")
            train_transformer(transformer, optimizer, settings, data_loader, device_str, num_last_epoch, tgt_vocab_size)
            mlflow.pytorch.log_model(transformer, settings.NAME_MODELS) # Log trained model as artifact
    else:
        logger.info(f"Train transformer from scratch or continue after epoch")
        train_transformer(transformer, optimizer, settings, data_loader, device_str, num_last_epoch, tgt_vocab_size)

###### Inference - Simple question on a financial data set
question = "How much tax loss remains from Baker Hughes?"
answer = run_inference(transformer, tokenizer, settings, device_str, question)
print("Answer: %s" % answer)

if settings.WANNABEE:
    wandb.finish()  # Finish wandb run
