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
