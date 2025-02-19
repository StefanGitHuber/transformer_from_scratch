# -*- coding: utf-8 -*-

import glob
import mmap
import os
import tiktoken
import torch

from collections import deque
from torch.utils import data
from typing import Tuple

from config import settings

###### Logging standard
from logger_setup import setup_logger
logger = setup_logger(__name__, settings.LOGGER_FILE)

class TokenDataset(data.Dataset):
    def __init__(
        self,
        tokenizer,
        dir_data: str,
        max_len: int,
        buffer_size: int,
        file_pattern: str
    ):
        """
        Memory efficient dataset processes all token from all text files
        
        Args:
            dir_data (str): Path to directory containing text files
            tokenizer: Tokenizer instance
            max_len (int): Maximum sequence length
            buffer_size (int): Size of text buffer to process at once
            file_pattern (str): Pattern for glob to match files
        """

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        
        # Get and sort files alphabetically
        self.file_paths = sorted(glob.glob(os.path.join(dir_data, file_pattern)))
        if not self.file_paths:
            raise ValueError(f"No files found in {dir_data} matching pattern {file_pattern}")

        # Count tokens over files
        self.files = []
        self.mms = []
        self.tokens_per_file = {}
        total_tokens = 0
        for file_path in self.file_paths:

            # Maps entire text file into memory: Watch your size! => TODO: Parse only sub-chunks
            file = open(file_path, 'rb')
            mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)

            # Count tokens per file
            position = 0
            file_tokens = 0
            while position < len(mm):
                chunk_size = min(buffer_size, len(mm) - position)
                new_position, text_chunk = self._read_valid_utf8_chunk(mm, position, chunk_size)
                tokens = self.tokenizer.encode(text_chunk)
                file_tokens += len(tokens)
                position = new_position
            
            self.files.append(file)
            self.mms.append(mm)
            self.tokens_per_file[file_path] = file_tokens
            total_tokens += file_tokens

        # Calculate total sequences needed for all tokens
        self.total_tokens = total_tokens
        self.num_sequences = (total_tokens + max_len - 1) // max_len
        
        # Initialize processing state
        self.token_buffer = deque(maxlen=buffer_size)
        self.current_file_idx = 0
        self.current_position = 0
        self.current_file_processed_tokens = 0
        self.file_completion_logged = False
        self.total_files_processed = 0

        # Log dataset information
        logger.info(f"▶️ Found {len(self.file_paths)} files to process:")
        for file_path, count in self.tokens_per_file.items():
            logger.info(f"\t{os.path.basename(file_path)} contains {count:,} tokens")
        logger.info(f"Total tokens across all files: {self.total_tokens}")
        logger.info(f"Expected sequences: {self.num_sequences}")
        logger.info(f"✔️ Dataset initialized with {self.num_sequences:,} sequences from {total_tokens:,} total tokens")

    
    def _read_valid_utf8_chunk(self, mm: mmap.mmap, start: int, max_chunk_size: int) -> Tuple[int, str]:
        """ Read chunk ending at valid UTF-8 boundary """

        chunk = mm[start:start + max_chunk_size]
        
        # Find last valid UTF-8 character boundary
        for ind in reversed(range(len(chunk))):
            try:
                return start + ind + 1, chunk[:ind+1].decode('utf-8')
            except UnicodeDecodeError:
                continue
        
        # Fallback: Skip problematic byte (rare)
        return start + 1, chunk[:1].decode('utf-8', errors='replace')


    def _fill_buffer(self) -> None:
        """ Fill buffer from current file, moving to next when complete """
        
        # Track if we've processed all files in this cycle
        files_in_this_cycle = set()
        
        # Loop until token buffer is filled AND we've processed all files
        while len(self.token_buffer) < self.max_len or len(files_in_this_cycle) < len(self.files):
            current_file = self.file_paths[self.current_file_idx]
            mm = self.mms[self.current_file_idx]
            
            # Add current file to our cycle tracking
            files_in_this_cycle.add(current_file)
            
            # Check if current file is complete
            if self.current_file_processed_tokens >= self.tokens_per_file[current_file]:
                # Log completion of current file
                if not self.file_completion_logged:
                    logger.info(f"Processed file {os.path.basename(current_file)}: {self.tokens_per_file[current_file]:,} tokens")
                    self.file_completion_logged = True
                    self.total_files_processed += 1
                
                # Move to next file
                next_file_idx = (self.current_file_idx + 1) % len(self.files)
                
                # Check for epoch completion (all files processed in this cycle)
                if len(files_in_this_cycle) == len(self.files):
                    self.total_files_processed = 0
                    files_in_this_cycle.clear()
                
                # Update state for next file
                self.current_file_idx = next_file_idx
                self.current_position = 0
                self.current_file_processed_tokens = 0
                self.file_completion_logged = False
                continue
            
            # Read from current file
            remaining = len(mm) - self.current_position
            if remaining <= 0:
                continue
            
            # Read + tokenize chunks
            chunk_size = min(remaining, self.buffer_size)
            new_position, text_chunk = self._read_valid_utf8_chunk(mm, self.current_position, chunk_size)
            new_tokens = self.tokenizer.encode(text_chunk)
            
            # Process new tokens
            if new_tokens:
                self.token_buffer.extend(new_tokens)
                self.current_file_processed_tokens += len(new_tokens)
                
            self.current_position = new_position
    
        # Debug log at end of fill
        logger.info(f"=== Filled buffer 🏁: Processed all {self.total_tokens:,} tokens across {len(self.files)} files")


    def __getitem__(self, idx: int) -> torch.Tensor:
        """ Get next sequence of tokens, ensuring we process ALL tokens """

        # Fill token buffer (over various text files)
        while len(self.token_buffer) < self.max_len:
            self._fill_buffer()
        
        token_ids = [self.token_buffer.popleft() for _ in range(self.max_len)]
        return torch.tensor(token_ids)


    def __len__(self) -> int:
        """ Total number of sequences needed for all tokens """

        return self.num_sequences


    def __del__(self):
        """ Cleanup resources """

        if hasattr(self, 'mms') and hasattr(self, 'files'):
            for mm, file in zip(self.mms, self.files):
                mm.close()
                file.close()


def token_dataloader(
    tokenizer,
    dir_data: str,
    max_seq_length: int,
    batch_size: int,
    buffer_size: int,
    file_pattern: str,
    shuffle: bool,
    num_workers: int
) -> data.DataLoader:
    """
    Efficient token-based data loader
    
    Args:
        tokenizer: Tokenizer instance
        dir_data (str): Path to directory containing text files
        max_seq_length (int): Maximum sequence length
        batch_size (int): Batch size
        buffer_size (int): Text buffer size
        file_pattern (str): Pattern for matching files (*.txt)
        shuffle (bool): Shuffle data
        num_workers (int): Number of worker processes
    
    Returns:
        DataLoader: Data loader for token sequences
    """
    dataset = TokenDataset(
        tokenizer=tokenizer,
        dir_data=dir_data,
        max_len=max_seq_length,
        buffer_size=buffer_size,
        file_pattern=file_pattern
    )
    
    def collate_fn(batch):
        inputs = torch.stack(batch)
        targets = torch.stack(batch)
        return inputs, targets
    
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader