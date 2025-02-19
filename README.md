# transformer_from_scratch
Train pure transformer on uncurated text files; Uses Intel Arc; Smart token+buffer processing

## Repository purpose
Comparison of my deterministic + causal ML framework for trustworthy + transparent xAI (reliable, robust, efficient, logic + symbolic, at scale, requires 1 curated example)
on its superb efficiency over the sub-symbolic blind pattern matching "AI" called Deep Learning with it's State-of-the-Art architecture "transformer" (inefficient, redundant, highest compute+data consumption, requires millions of uncurated examples)

### Installation
Please run
**pip install -r requirements.txt**
to install the following dependencies:
*   `torch`: The core PyTorch library.
*   `psutil`: Used for monitoring memory usage.
*   `tiktoken`: Used for tokenization with GPT-2.
*   `wandb`: Used for experiment tracking and visualization.
*   `mlflow`: Used for experiment tracking and model logging.
*   `intel_extension_for_pytorch`: Required for optimizing PyTorch on Intel GPUs.

### Usage
Please configure stuff in config.py according to your environment. Especially provide all the text files indented to train your tiny transformer in the corresponding data path. Then just run **./main.py** at your bash. Simplest training without validation while and evaluation data after convergence. Triggers early stopping if percentage over epochs saturates below threshold. Inference runs only one non-sense question. 

### Notes on files
*   `main.py`: Configures MLflow + WandB logging, instantiates tokenizer & dataloader & Transformer model, runs training and/or inference
*   `token_dataloader`: Tokenization and smart memory-efficient buffer loading.
*   `memory_monitor`: Trace memory consumption over training iterations.
*   `transformer_core`: All the common building blocks like Multi-Headed Attention, Position-wise Feed-Forward Networks, Positional Encoding, Encoder and Decoder layers, Masked Token Learning etc
*   `transformer_train_infer`: Loads latest epoch from disk (in case available) to automatically continue training; Inference generates output answer token by token

### Special notes on token_dataloader.py

Custom data loading system processes large text files efficiently for language model training.

1.  `TokenDataset Class`
    -   Efficiently processes text files, reads in chunks, converts to tokens
    -   Uses memory mapping (mmap) to handle large files without loading them entirely into memory
    -   Maintains token buffer to serve as sequences
    -   Tracks progress through files and ensures all tokens are processed
    -   Handles UTF-8 encoding properly

2.  `token_dataloader function`
    -   Common PyTorch DataLoader serves batches of token sequences
    -   Common Collate function that creates input-target pairs for language model training
    -   Supports multi-worker data loading and GPU pinned memory for better performance

3.  `Key features`
    -   Flexible: Supports various file patterns and configurable sequence lengths
    -   Robust: Handles UTF-8 encoding boundaries correctly
    -   Progress tracking: Logs progress through files and token counts
    -   Memory efficient: Uses memory mapping instead of loading entire files
    -   Complete coverage: Ensures all tokens from all files are processed

4.  `Properties`
    -   Process large text corpora
    -   Convert text to tokens
    -   Create sequences of fixed length
    -   Serve sequences in batches for training
    -   Memory-efficiently with proper resource cleanup
    -   Dataloader for next-token prediction
