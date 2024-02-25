import os
import sys

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

from sae_training.cache_activations_runner import cache_activations_runner
from sae_training.config import (
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
)
from sae_training.lm_runner import language_model_sae_runner

# cfg = CacheActivationsRunnerConfig(

#     # Data Generating Function (Model + Training Distibuion)
#     model_name = "gpt2-small",
#     hook_point = f"blocks.{3}.hook_resid_pre",
#     hook_point_layer = 3,
#     d_in = 768,
#     dataset_path = "Skylion007/openwebtext",
#     is_dataset_tokenized=True,
#     cached_activations_path="activations/",
    
#     # Activation Store Parameters
#     n_batches_in_buffer = 16,
#     total_training_tokens = 150_000_000, 
#     store_batch_size = 64,

#     # Activation caching shuffle parameters
#     n_shuffles_final = 16,
    
#     # Misc
#     device = "cuda",
#     seed = 42,
#     dtype = torch.bfloat16,
#     )

# cache_activations_runner(cfg)

for layer in [3]:
    for use_ghost_grads in [False, True]:

        cfg = LanguageModelSAERunnerConfig(

            # Data Generating Function (Model + Training Distibuion)
            model_name = "gpt2-small",
            hook_point = f"blocks.{layer}.hook_resid_pre",
            hook_point_layer = layer,
            d_in = 768,
            dataset_path = "Skylion007/openwebtext",
            is_dataset_tokenized=False,
            
            # SAE Parameters
            expansion_factor = 32, # determines the dimension of the SAE.
            b_dec_init_method = "mean", # just while getting started
            normalize_activations = True,
            use_pre_encoder_bias= False,
            
            # Training Parameters
            lr = 0.0004,
            l1_coefficient = 0.004,
            lr_scheduler_name="constantwithwarmup",
            train_batch_size = 4096,
            context_size = 128,
            lr_warm_up_steps=10000,
            adam_beta1 = 0,
            adam_beta2 = 0.9999,
            
            # Activation Store Parameters
            n_batches_in_buffer = 128,
            total_training_tokens = 1_000_000 * 300, # 200M tokens seems doable overnight.
            store_batch_size = 32,
            
            # Resampling protocol
            mse_loss_normalization = None,
            ghost_grads=None,#"residual",
            feature_sampling_window = 1000,
            dead_feature_window=5000,
            # dead_feature_window=50000,
            dead_feature_threshold = 1e-8,
            
            # WANDB
            log_to_wandb = True,
            wandb_project= "mats_sae_training_gpt2_ghost_grad_experiment",
            wandb_entity = None,
            wandb_log_frequency=100,
            
            # Misc
            device = "cuda",
            seed = 42,
            n_checkpoints = 10,
            checkpoint_path = "checkpoints",
            dtype = torch.float32,
            )

        sparse_autoencoder = language_model_sae_runner(cfg)