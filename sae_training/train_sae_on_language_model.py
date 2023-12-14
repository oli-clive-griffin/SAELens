from functools import partial

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from transformer_lens import HookedTransformer

import wandb
from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.activations_store import ActivationsStore
from sae_training.optim import get_scheduler
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import calc_attn_pattern


def train_sae_on_language_model(
    model: HookedTransformer,
    sparse_autoencoder: SparseAutoencoder,
    activation_store: ActivationsStore,
    cfg: LanguageModelSAERunnerConfig,
):
    batch_size = cfg.train_batch_size
    n_checkpoints = cfg.n_checkpoints

    total_training_tokens = cfg.total_training_tokens
    total_training_steps = total_training_tokens // batch_size
    n_training_steps = 0
    n_training_tokens = 0
    n_resampled_neurons = 0
    if n_checkpoints > 0:
        checkpoint_thresholds = list(range(0, total_training_tokens, total_training_tokens // n_checkpoints))[1:]
    
    # track active features
    act_freq_scores = torch.zeros(cfg.d_sae, device=cfg.device)
    n_frac_active_tokens = 0
    
    optimizer = Adam(sparse_autoencoder.parameters(), lr = cfg.lr)
    scheduler = get_scheduler(
        cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps = cfg.lr_warm_up_steps, 
        training_steps=total_training_steps,
        lr_end=cfg.lr / 10, # heuristic for now. 
    )
    sparse_autoencoder.train()
    

    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:
        # Do a training step.
        activations = activation_store.next_batch()

        # Make sure the W_dec is still zero-norm
        sparse_autoencoder.set_decoder_norm_to_unit_norm()

        # Resample dead neurons
        if (cfg.feature_sampling_method is not None) and ((n_training_steps + 1) % cfg.dead_feature_window == 0):

            # Get the fraction of neurons active in the previous window
            feature_sparsity = act_freq_scores / n_frac_active_tokens
            is_dead = (feature_sparsity < cfg.dead_feature_threshold)
            
            # if standard resampling <- do this
            n_resampled_neurons = sparse_autoencoder.resample_neurons(
                activations,
                feature_sparsity, 
                cfg.feature_reinit_scale,
                optimizer
            )
            # for all the dead neurons, set the feature sparsity to the dead feature threshold
            act_freq_scores[is_dead] = cfg.dead_feature_threshold * n_frac_active_tokens
            if n_resampled_neurons > 0:
                print(f"Resampled {n_resampled_neurons} neurons")
            if cfg.use_wandb:
                wandb.log(
                    {
                        "metrics/n_resampled_neurons": n_resampled_neurons,
                    },
                    step=n_training_steps,
                )
            n_resampled_neurons = 0

        # Update learning rate here if using scheduler.

        # Forward and Backward Passes
        optimizer.zero_grad()
        sae_out, feature_acts, loss, mse_loss, l1_loss = sparse_autoencoder(activations)
        n_training_tokens += batch_size

        if cfg.use_pattern_reconstruction_loss:
            # Throw out the MSE loss and instead calculate the pattern reconstruction MSE
            origial_pattern = calc_attn_pattern(q_activations, k_activations, cfg, model)
            if cfg.hook_point[-6:] == 'hook_q':
                reconstructed_pattern = calc_attn_pattern(sae_out, k_activations, cfg, model)
            else:
                reconstructed_pattern = calc_attn_pattern(q_activations, sae_out, cfg, model)
            
            pattern_mse_loss = F.mse_loss(origial_pattern, reconstructed_pattern)
            loss = pattern_mse_loss + l1_loss

        with torch.no_grad():
            # Calculate the sparsities, and add it to a list, calculate sparsity metrics
            act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
            n_frac_active_tokens += batch_size
            feature_sparsity = act_freq_scores / n_frac_active_tokens

            # metrics for currents acts
            l0 = (feature_acts > 0).float().sum(1).mean()
            l2_norm_in = torch.norm(activations, dim=-1).mean()
            l2_norm_out = torch.norm(sae_out, dim=-1).mean()
            l2_norm_ratio = l2_norm_out / l2_norm_in
            current_learning_rate = optimizer.param_groups[0]["lr"]

            if cfg.use_wandb and ((n_training_steps + 1) % cfg.wandb_log_frequency == 0):
                if cfg.use_pattern_reconstruction_loss:
                    wandb.log({
                        "losses/pattern_mse_loss": pattern_mse_loss.item(),
                        "metric/unused_mse_loss": mse_loss.item(),
                    })
                else:
                    wandb.log({
                        "losses/mse_loss": mse_loss.item(),
                    })
                
                wandb.log(
                    {
                        "losses/l1_loss": l1_loss.item(),
                        "losses/overall_loss": loss.item(),
                        "metrics/l0": l0.item(),
                        "metrics/l2": l2_norm_out.item(),
                        "metrics/l2_ratio": l2_norm_ratio.item(),
                        "metrics/below_1e-5": (feature_sparsity < 1e-5)
                        .float()
                        .mean()
                        .item(),
                        "metrics/below_1e-6": (feature_sparsity < 1e-6)
                        .float()
                        .mean()
                        .item(),
                        "metrics/dead_features": (
                            feature_sparsity < cfg.dead_feature_threshold
                        )
                        .float()
                        .mean()
                        .item(),
                        "details/n_training_tokens": n_training_tokens,
                        "metrics/current_learning_rate": current_learning_rate,
                    },
                    step=n_training_steps,
                )

            # record loss frequently, but not all the time.
            if cfg.use_wandb and ((n_training_steps + 1) % (cfg.wandb_log_frequency * 10) == 0):
                # Now we want the reconstruction loss.
                recons_score, ntp_loss, recons_loss, zero_abl_loss = get_recons_loss(sparse_autoencoder, model, activation_store, num_batches=3)
                
                wandb.log(
                    {
                        "metrics/reconstruction_score": recons_score,
                        "metrics/ce_loss_without_sae": ntp_loss,
                        "metrics/ce_loss_with_sae": recons_loss,
                        "metrics/ce_loss_with_ablation": zero_abl_loss,
                        
                    },
                    step=n_training_steps,
                )
                    
            # use feature window to log feature sparsity
            if cfg.use_wandb and ((n_training_steps + 1) % cfg.feature_sampling_window == 0):
                log_feature_sparsity = torch.log10(feature_sparsity + 1e-10)
                wandb.log(
                    {
                        "plots/feature_density_histogram": wandb.Histogram(
                            log_feature_sparsity.tolist()
                        ),
                    },
                    step=n_training_steps,
                )


            pbar.set_description(
                f"{n_training_steps}| MSE Loss {mse_loss.item():.3f} | L0 {l0.item():.3f}"
            )
            pbar.update(batch_size)

        loss.backward()
        sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()
        scheduler.step()

        # checkpoint if at checkpoint frequency
        if n_checkpoints > 0 and n_training_tokens > checkpoint_thresholds[0]:
            cfg = cfg
            path = f"{cfg.checkpoint_path}/{n_training_tokens}_{sparse_autoencoder.get_name()}.pkl.gz"
            sparse_autoencoder.save_model(path)
            checkpoint_thresholds.pop(0)
            if len(checkpoint_thresholds) == 0:
                n_checkpoints = 0
            if cfg.log_to_wandb:
                model_artifact = wandb.Artifact(
                    f"{sparse_autoencoder.get_name()}", type="model", metadata=dict(cfg.__dict__)
                )
                model_artifact.add_file(path)
                wandb.log_artifact(model_artifact)
            
        n_training_steps += 1

    return sparse_autoencoder

@torch.no_grad()
def get_recons_loss(sparse_autoencder, model, activation_store, num_batches=5):
    hook_point = activation_store.cfg.hook_point
    loss_list = []
    for _ in range(num_batches):
        batch_tokens = activation_store.get_batch_tokens()
        loss = model(batch_tokens, return_type="loss")

        # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss",
        # fwd_hooks=[(utils.get_act_name("post", 0), mean_ablate_hook)])

        recons_loss = model.run_with_hooks(
            batch_tokens,
            return_type="loss",
            fwd_hooks=[(hook_point, partial(replacement_hook, encoder=sparse_autoencder))],
        )

        zero_abl_loss = model.run_with_hooks(
            batch_tokens, return_type="loss", fwd_hooks=[(hook_point, zero_ablate_hook)]
        )
        loss_list.append((loss, recons_loss, zero_abl_loss))

    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)

    # print(loss, recons_loss, zero_abl_loss)
    # print(f"{score:.2%}")
    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")

    return score, loss, recons_loss, zero_abl_loss


def replacement_hook(mlp_post, hook, encoder):
    mlp_post_reconstr = encoder(mlp_post)[0]
    return mlp_post_reconstr


def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post


def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.0
    return mlp_post
