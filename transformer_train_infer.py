# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

import mlflow
import re
import statistics
import tiktoken
import wandb

from torch.utils.data import DataLoader

from config import Config, settings
from logger_setup import setup_logger
from memory_monitor import print_vram_usage
from transformer_core import Transformer

###### Logging standard
logger = setup_logger(__name__, settings.LOGGER_FILE)

###### Helper function loads last epoch model
def load_epoch_model(
    transformer: Transformer,
    optimizer: optim,
    settings: Config) -> int:
    """ Load latest model, returns epoch number """

    logger.info(f"Load latest epoch from: {settings.DIR_MODELS}")
    num_last_epoch = 0
    model_files = list(settings.DIR_MODELS.glob(f"{settings.NAME_MODELS}*.pth"))

    if model_files:
        # Extract epoch numbers using regex
        epoch_numbers = []
        for file in model_files:
            pattern = re.compile(fr"{re.escape(settings.NAME_MODELS)}(\d+)\.pth")
            match = pattern.search(str(file))
            if match:
                epoch_numbers.append(int(match.group(1)))

        if epoch_numbers:
            # Find file with maximum epoch number
            num_last_epoch = max(epoch_numbers)
            latest_file = settings.DIR_MODELS / f"{settings.NAME_MODELS}{num_last_epoch}.pth"
            
            logger.info(f"Load latest model: {latest_file}")
            checkpoint = torch.load(latest_file)
            transformer.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logger.warning(f"No valid model files found in {settings.DIR_MODELS}")
    else:
        logger.warning(f"No model files found in {settings.DIR_MODELS}")
    return num_last_epoch

###### Training Loop
def train_transformer(
    transformer: Transformer,
    optimizer: optim,
    settings: Config,
    data_loader: DataLoader,
    device_str: str,
    num_last_epoch: int,
    tgt_vocab_size: int
) -> None:
    """ Train transformer from scratch or continue after last epoch """ 
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    transformer.train()
    early_stopping = False
    all_epoch_losses = []
    for epoch in range(num_last_epoch, settings.NUM_EPOCHS):
        batch_losses = []
        epoch_loss = 0.0
        
        # Training loop over batches
        for batch_idx, (batch_inputs, batch_targets) in enumerate(data_loader):
            batch_inputs, batch_targets = batch_inputs.to(device_str), batch_targets.to(device_str)
            optimizer.zero_grad()
            outputs = transformer(batch_inputs, batch_targets[:, :-1])
            loss = criterion(outputs.contiguous().view(-1, tgt_vocab_size), batch_targets[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            epoch_loss += batch_loss
            if settings.WANNABEE:
                wandb.log({"batch_loss": batch_loss})
            if settings.MLFLOW:
                mlflow.log_metric("batch_loss", batch_loss, step=epoch * batch_idx) # MLflow log with step

        # Logging
        epoch_mean_loss = statistics.mean(batch_losses)
        epoch_stdev_loss = statistics.stdev(batch_losses) if len(batch_losses) > 1 else 0
        if settings.WANNABEE:
            wandb.log({"epoch_mean_loss": epoch_mean_loss, "epoch_stdev_loss": epoch_stdev_loss})
        if settings.MLFLOW:
            mlflow.log_metric("epoch_mean_loss", epoch_mean_loss, step=epoch)
            mlflow.log_metric("epoch_stdev_loss", epoch_stdev_loss, step=epoch)
        logger.info(f"Epoch {epoch+1}: Mean Loss = {epoch_mean_loss:.4f}, Std Dev = {epoch_stdev_loss:.4f}")
        print_vram_usage(settings.TRACE_VRAM, f"GPU VRAM after Epoch: {epoch+1}, Batch: {batch_idx}")

        # Early stopping: If mu_loss < thresh_percent
        all_epoch_losses.append(epoch_mean_loss)
        if abs(epoch - num_last_epoch) > 1:
            mean_diff_percent = abs((all_epoch_losses[-2] - all_epoch_losses[-1]) / all_epoch_losses[-1]) * 100
            if mean_diff_percent < settings.EARLY_STOPPING_PERCENT and all_epoch_losses[-1] < all_epoch_losses[-2]:
                logger.info(f"Early stopping triggered at epoch {epoch+1}: ")
                early_stopping = True

        # Checkpointing: Early stopping or after each EPOCH_STORE_INTERVAL
        if early_stopping is True or (epoch + 1) % settings.EPOCH_STORE_INTERVAL == 0:
            model_name = settings.DIR_MODELS / f'{settings.NAME_MODELS}{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_name)
        if early_stopping is True:
            break

###### Inference
def run_inference(
    transformer: Transformer,
    tokenizer: tiktoken.Encoding,
    settings: Config,
    device_str: str,
    question: str
) -> str:
    """Model inference: Tokenized prompt/question answered by ranking probabilities per (next) token"""

    transformer.eval()
    inp_tokens = tokenizer.encode(question)
    inp_tensor = torch.tensor(inp_tokens).unsqueeze(0).to(device_str)

    start_token_id = tokenizer.encode("`")[0]
    tgt_tokens = [start_token_id]
    tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device_str)

    # Generate token by output token
    with torch.no_grad():
        for fake in range(settings.MAX_SEQ_LEN):
            output = transformer(inp_tensor, tgt_tensor)
            next_token = output.argmax(dim=-1)[:, -1].item()
            tgt_tensor = torch.cat([tgt_tensor, torch.tensor([[next_token]]).to(device_str)], dim=-1)
            if next_token == tokenizer.eot_token:  # Use the GPT-2 end-of-text token ID
                break

    # Decode output tokens
    output_tokens = tgt_tensor.squeeze().tolist()
    answer = tokenizer.decode(output_tokens)
    return answer
