# -*- coding: utf-8 -*-

import psutil
import torch
from datetime import datetime

# Logging standard
from logger_setup import setup_logger
logger = setup_logger(__name__)

# Constants for memory conversion
BYTES_TO_MB = 1024 * 1024
BYTES_TO_GB = 1024 * 1024 * 1024

def memory_usage():
    """Print + log memory usage of current process in MB"""

    process = psutil.Process()
    mem_info = process.memory_info()
    
    rss_mb = mem_info.rss / BYTES_TO_MB               # Physical memory
    vms_mb = mem_info.vms / BYTES_TO_MB           # Virtual memory
    shared_mb = mem_info.shared / BYTES_TO_MB   # Shared memory

    logger.info(f"RSS (Physical): \t\t{rss_mb:.2f} MB")
    logger.info(f"VMS (Virtual): \t\t{vms_mb:.2f} MB")
    logger.info(f"Shared: \t\t\t{shared_mb:.2f} MB")

def print_vram_usage(beactive: bool, txt_when: str):
    """Print + log current allocated/reserved VRAM"""

    if beactive:
        logger.info(txt_when)
        allocated_memory = torch.xpu.memory_allocated(0) / (1024 ** 3)
        reserved_memory = torch.xpu.memory_reserved(0) / (1024 ** 3)
        logger.info(f"Allocated memory: \t{allocated_memory:.2f} GB")
        logger.info(f"Reserved  memory: \t{reserved_memory:.2f} GB")
        
        memory_usage()
