import os
import sys

import torch

# sys.path.append("..")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.lm_runner import language_model_sae_runner

cfg = LanguageModelSAERunnerConfig(

    # Data Generating Function (Model + Training Distibuion)
    model_name = "gpt2-small",
    train_on_full_resid = True,
    d_in = 768,
    dataset_path = "Skylion007/openwebtext",
    is_dataset_tokenized=False,
    
    # SAE Parameters
    expansion_factor = 64, # determines the dimension of the SAE.
    
    # Training Parameters
    lr = 0.0012,
    l1_coefficient = 0.00008,
    lr_scheduler_name=None,
    train_batch_size = 4096,
    context_size = 128,
    lr_warm_up_steps=5000,
    
    # Activation Store Parameters
    n_batches_in_buffer = 16,#128,
    total_training_tokens = 100_000,#1_000_000 * 1_000,
    store_batch_size = 32,
    
    # Resampling protocol
    feature_sampling_method = 'anthropic',
    feature_sampling_window = 1000,
    feature_reinit_scale = 0.2,
    resample_batches=1028,
    dead_feature_window=50000,
    dead_feature_threshold = 1e-6,
    
    # WANDB
    log_to_wandb = True,
    wandb_project= "mats_sae_training_gpt2_small_all_resid",
    wandb_entity = None,
    wandb_log_frequency=100,
    
    # Misc
    device = "mps",
    seed = 42,
    n_checkpoints = 10,
    checkpoint_path = "checkpoints",
    dtype = torch.float32,
)

sparse_autoencoder = language_model_sae_runner(cfg)