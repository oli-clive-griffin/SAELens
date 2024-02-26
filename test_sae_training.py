
import os
import sys

import torch

sys.path.append("..")
from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.lm_runner import language_model_sae_runner

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

for expansion_factor in [8]:
    for finetuning_method in ["decoder"]:
        cfg = LanguageModelSAERunnerConfig(

            # Data Generating Function (Model + Training Distibuion)
            model_name = "gelu-2l",
            hook_point = "blocks.0.hook_mlp_out",
            hook_point_layer = 0,
            d_in = 512,
            dataset_path = "NeelNanda/c4-tokenized-2b",
            is_dataset_tokenized=True,
            
            # SAE Parameters
            expansion_factor = expansion_factor,
            b_dec_init_method="mean", # geometric median is better but slower to get started
            normalize_activations = True,
            use_pre_encoder_bias= False,
            
            # Training Parameters
            lr = 0.0012,
            lr_scheduler_name="constantwithwarmup",
            l1_coefficient = 0.004,
            train_batch_size = 4096,
            context_size = 1024,
            adam_beta1 = 0,
            adam_beta2 = 0.9999,
            finetuning_method = finetuning_method,
            
            # Activation Store Parameters
            n_batches_in_buffer = 128,
            total_training_tokens = 1_000_000 * 100, 
            fine_tune_tokens = 1_000_000 * 100,
            store_batch_size = 32,
            
            # Resampling protocol
            mse_loss_normalization = None,#"variance",
            ghost_grads=None,#"residual",
            feature_sampling_window = 500,
            dead_feature_window=500,
            dead_feature_threshold = 1e-5,
            
            # WANDB
            log_to_wandb = True,
            wandb_project= "mats_sae_training_language_models_gelu_2l_finetuning_experiment",
            wandb_log_frequency=10,
            
            # Misc
            device = device,
            seed = 42,
            n_checkpoints = 0,
            checkpoint_path = "checkpoints",
            dtype = torch.bfloat16,
            )


        sparse_autoencoder = language_model_sae_runner(cfg)