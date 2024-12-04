import contextlib  # noqa: I001
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import wandb
from torch.optim import Adam
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule

from sae_lens import __version__
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.evals import EvalConfig, run_evals
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.optim import L1Scheduler, get_lr_scheduler
from sae_lens.training.training_sae import TrainingSAE, TrainStepOutput

# used to map between parameters which are updated during finetuning and the config str.
FINETUNING_PARAMETERS = {
    "scale": ["scaling_factor"],
    "decoder": ["scaling_factor", "W_dec", "b_dec"],
    "unrotated_decoder": ["scaling_factor", "b_dec"],
}


def _log_feature_sparsity(
    feature_sparsity: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    return torch.log10(feature_sparsity + eps).detach().cpu()


def _update_sae_lens_training_version(sae: TrainingSAE) -> None:
    """
    Make sure we record the version of SAELens used for the training run
    """
    sae.cfg.sae_lens_training_version = str(__version__)


@dataclass
class TrainSAEOutput:
    sae: TrainingSAE
    checkpoint_path: str
    log_feature_sparsities: torch.Tensor


class SAETrainer:
    """
    Core SAE class used for inference. For training, see TrainingSAE.
    """

    model: HookedRootModule
    sae: TrainingSAE
    activation_store: ActivationsStore
    cfg: LanguageModelSAERunnerConfig

    def __init__(
        self,
        model: HookedRootModule,
        sae: TrainingSAE,
        activation_store: ActivationsStore,
        cfg: LanguageModelSAERunnerConfig,
    ) -> None:
        self.model = model
        self.sae = sae
        self.activation_store = activation_store
        self.cfg = cfg
        self._compile_if_needed()

        self.n_training_steps: int = 0
        self.n_training_tokens: int = 0
        self.started_fine_tuning: bool = False

        _update_sae_lens_training_version(self.sae)

        self.checkpoint_thresholds = []
        if self.cfg.n_checkpoints > 0:
            self.checkpoint_thresholds = list(
                range(
                    0,
                    cfg.total_training_tokens,
                    cfg.total_training_tokens // self.cfg.n_checkpoints,
                )
            )[1:]

        self.act_freq_scores = torch.zeros(
            cast(int, cfg.d_sae),
            device=cfg.device,
        )
        self.n_forward_passes_since_fired = torch.zeros(
            cast(int, cfg.d_sae),
            device=cfg.device,
        )
        self.n_frac_active_tokens = 0
        # we don't train the scaling factor (initially)
        # set requires grad to false for the scaling factor
        for name, param in self.sae.named_parameters():
            if "scaling_factor" in name:
                param.requires_grad = False

        self.optimizer = Adam(
            sae.parameters(),
            lr=cfg.lr,
            betas=(
                cfg.adam_beta1,
                cfg.adam_beta2,
            ),
        )
        assert cfg.lr_end is not None  # this is set in config post-init
        self.lr_scheduler = get_lr_scheduler(
            cfg.lr_scheduler_name,
            lr=cfg.lr,
            optimizer=self.optimizer,
            warm_up_steps=cfg.lr_warm_up_steps,
            decay_steps=cfg.lr_decay_steps,
            training_steps=self.cfg.total_training_steps,
            lr_end=cfg.lr_end,
            num_cycles=cfg.n_restart_cycles,
        )
        self.l1_scheduler = L1Scheduler(
            l1_warm_up_steps=cfg.l1_warm_up_steps,
            total_steps=cfg.total_training_steps,
            final_l1_coefficient=cfg.l1_coefficient,
        )

        # Setup autocast if using
        self.scaler = torch.amp.GradScaler(
            device=self.cfg.device, enabled=self.cfg.autocast
        )

        if self.cfg.autocast:
            self.autocast_if_enabled = torch.autocast(
                device_type=self.cfg.device,
                dtype=torch.bfloat16,
                enabled=self.cfg.autocast,
            )
        else:
            self.autocast_if_enabled = contextlib.nullcontext()

        # Set up eval config

        self.trainer_eval_config = EvalConfig(
            batch_size_prompts=self.cfg.eval_batch_size_prompts,
            n_eval_reconstruction_batches=self.cfg.n_eval_batches,
            n_eval_sparsity_variance_batches=self.cfg.n_eval_batches,
            compute_ce_loss=True,
            compute_l2_norms=True,
            compute_sparsity_metrics=True,
            compute_variance_metrics=True,
            compute_kl=False,
            compute_featurewise_weight_based_metrics=False,
        )

    @property
    def feature_sparsity(self) -> torch.Tensor:
        return self.act_freq_scores / self.n_frac_active_tokens

    @property
    def log_feature_sparsity(self) -> torch.Tensor:
        return _log_feature_sparsity(self.feature_sparsity)

    @property
    def current_l1_coefficient(self) -> float:
        return self.l1_scheduler.current_l1_coefficient

    @property
    def dead_neurons(self) -> torch.Tensor:
        return (self.n_forward_passes_since_fired > self.cfg.dead_feature_window).bool()

    def _compile_if_needed(self):
        # Compile model and SAE
        #  torch.compile can provide significant speedups (10-20% in testing)
        # using max-autotune gives the best speedups but:
        # (a) increases VRAM usage,
        # (b) can't be used on both SAE and LM (some issue with cudagraphs), and
        # (c) takes some time to compile
        # optimal settings seem to be:
        # use max-autotune on SAE and max-autotune-no-cudagraphs on LM
        # (also pylance seems to really hate this)
        if self.cfg.compile_llm:
            self.model = torch.compile(
                self.model,
                mode=self.cfg.llm_compilation_mode,
            )  # type: ignore

        if self.cfg.compile_sae:
            backend = "aot_eager" if self.cfg.device == "mps" else "inductor"

            self.sae.training_forward_pass = torch.compile(  # type: ignore
                self.sae.training_forward_pass,
                mode=self.cfg.sae_compilation_mode,
                backend=backend,
            )  # type: ignore

    @torch.no_grad()
    def estimate_norm_scaling_factor(
        self, n_batches_for_norm_estimate: int = int(1e3)
    ) -> float:
        norms_per_batch: list[float] = []
        for _ in tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            # TODO(oli-clive-griffin): check there's no indexing needed here
            # TODO(oli-clive-griffin): I think this is calculating across all layers, is that right? (OHHH, is `n_layers` expected to be 1?)
            acts = self.activation_store.next_batch()
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        return np.sqrt(self.cfg.d_in) / mean_norm

    def fit(self) -> TrainingSAE:
        pbar = tqdm(total=self.cfg.total_training_tokens, desc="Training SAE")

        estimated_norm_scaling_factor = (
            self.estimate_norm_scaling_factor()
            if self.cfg.normalize_activations == "expected_average_only_in"
            else None
        )

        # Train loop
        while self.n_training_tokens < self.cfg.total_training_tokens:
            # Do a training step.
            layer_acts = self.activation_store.next_batch()[:, 0, :].to(self.sae.device)
            self.n_training_tokens += self.cfg.train_batch_size_tokens

            step_output = self._train_step(sae=self.sae, sae_in=layer_acts)

            if self.cfg.log_to_wandb:
                self._log_train_step(step_output)
                self._run_and_log_evals()

            self._checkpoint_if_needed()
            self.n_training_steps += 1
            self._update_pbar(step_output, pbar)

            ### If n_training_tokens > sae_group.cfg.training_tokens, then we should switch to fine-tuning (if we haven't already)
            self._begin_finetuning_if_needed()

        if estimated_norm_scaling_factor is not None:
            self.sae.fold_activation_norm_scaling_factor(estimated_norm_scaling_factor)

        # save final sae group to checkpoints folder
        self.save_checkpoint(
            checkpoint_name=f"final_{self.n_training_tokens}",
            wandb_aliases=["final_model"],
        )

        pbar.close()
        return self.sae

    def save_checkpoint(
        self,
        checkpoint_name: str,
        wandb_aliases: list[str] | None = None,
    ) -> None:
        """Save a checkpoint of the SAE locally and optionally to wandb."""

        base_path = Path(self.cfg.checkpoint_path) / checkpoint_name
        base_path.mkdir(exist_ok=True, parents=True)

        if self.sae.cfg.normalize_sae_decoder:
            self.sae.set_decoder_norm_to_unit_norm()

        weights_path, cfg_path, sparsity_path = self.sae.save_model(
            str(base_path),
            self.log_feature_sparsity,
        )

        # let's over write the cfg file with the trainer cfg, which is a super set of the original cfg.
        # and should not cause issues but give us more info about SAEs we trained in SAE Lens.
        config = self.cfg.to_dict()
        with open(cfg_path, "w") as f:
            json.dump(config, f)

        if self.cfg.log_to_wandb:
            # Avoid wandb saving errors such as:
            #   ValueError: Artifact name may only contain alphanumeric characters, dashes, underscores, and dots. Invalid name: sae_google/gemma-2b_etc
            sae_name = self.sae.get_name().replace("/", "__")

            # save model weights and cfg
            model_artifact = wandb.Artifact(
                sae_name,
                type="model",
                metadata=dict(self.cfg.__dict__),
            )
            model_artifact.add_file(str(weights_path))
            model_artifact.add_file(str(cfg_path))
            wandb.log_artifact(model_artifact, aliases=wandb_aliases)

            # save log feature sparsity
            sparsity_artifact = wandb.Artifact(
                f"{sae_name}_log_feature_sparsity",
                type="log_feature_sparsity",
                metadata=dict(self.cfg.__dict__),
            )
            sparsity_artifact.add_file(str(sparsity_path))
            wandb.log_artifact(sparsity_artifact)

    def _train_step(
        self,
        sae: TrainingSAE,
        sae_in: torch.Tensor,
    ) -> TrainStepOutput:
        sae.train()
        # Make sure the W_dec is still unit-norm
        if self.cfg.normalize_sae_decoder:
            sae.set_decoder_norm_to_unit_norm()

        # log and then reset the feature sparsity every feature_sampling_window steps
        if (self.n_training_steps + 1) % self.cfg.feature_sampling_window == 0:
            if self.cfg.log_to_wandb:
                sparsity_log_dict = self._build_sparsity_log_dict()
                wandb.log(sparsity_log_dict, step=self.n_training_steps)
            self._reset_running_sparsity_stats()

        # for documentation on autocasting see:
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        with self.autocast_if_enabled:
            train_step_output = self.sae.training_forward_pass(
                sae_in=sae_in,
                dead_neuron_mask=self.dead_neurons,
                current_l1_coefficient=self.current_l1_coefficient,
            )

            with torch.no_grad():
                did_fire = (train_step_output.feature_acts > 0).float().sum(-2) > 0
                self.n_forward_passes_since_fired += 1
                self.n_forward_passes_since_fired[did_fire] = 0
                self.act_freq_scores += (
                    (train_step_output.feature_acts.abs() > 0).float().sum(0)
                )
                self.n_frac_active_tokens += self.cfg.train_batch_size_tokens

        # Scaler will rescale gradients if autocast is enabled
        self.scaler.scale(
            train_step_output.loss
        ).backward()  # loss.backward() if not autocasting
        self.scaler.unscale_(self.optimizer)  # needed to clip correctly
        # TODO: Work out if grad norm clipping should be in config / how to test it.
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        self.scaler.step(self.optimizer)  # just ctx.optimizer.step() if not autocasting
        self.scaler.update()

        if self.cfg.normalize_sae_decoder:
            sae.remove_gradient_parallel_to_decoder_directions()

        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        self.l1_scheduler.step()

        return train_step_output

    @torch.no_grad()
    def _log_train_step(self, step_output: TrainStepOutput):
        if (self.n_training_steps + 1) % self.cfg.wandb_log_frequency == 0:
            wandb.log(
                self._build_train_step_log_dict(
                    output=step_output,
                    n_training_tokens=self.n_training_tokens,
                ),
                step=self.n_training_steps,
            )

    @torch.no_grad()
    def _build_train_step_log_dict(
        self,
        output: TrainStepOutput,
        n_training_tokens: int,
    ) -> dict[str, Any]:
        sae_in = output.sae_in
        sae_out = output.sae_out
        feature_acts = output.feature_acts
        loss = output.loss.item()

        # metrics for currents acts
        l0 = (feature_acts > 0).float().sum(-1).mean()
        current_learning_rate = self.optimizer.param_groups[0]["lr"]

        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
        total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
        explained_variance = 1 - per_token_l2_loss / total_variance

        log_dict = {
            # losses
            "losses/overall_loss": loss,
            # variance explained
            "metrics/explained_variance": explained_variance.mean().item(),
            "metrics/explained_variance_std": explained_variance.std().item(),
            "metrics/l0": l0.item(),
            # sparsity
            "sparsity/mean_passes_since_fired": self.n_forward_passes_since_fired.mean().item(),
            "sparsity/dead_features": self.dead_neurons.sum().item(),
            "details/current_learning_rate": current_learning_rate,
            "details/current_l1_coefficient": self.current_l1_coefficient,
            "details/n_training_tokens": n_training_tokens,
        }
        for loss_name, loss_value in output.losses.items():
            loss_item = _unwrap_item(loss_value)
            # special case for l1 loss, which we normalize by the l1 coefficient
            if loss_name == "l1_loss":
                log_dict[f"losses/{loss_name}"] = (
                    loss_item / self.current_l1_coefficient
                )
                log_dict[f"losses/raw_{loss_name}"] = loss_item
            else:
                log_dict[f"losses/{loss_name}"] = loss_item

        return log_dict

    @torch.no_grad()
    def _run_and_log_evals(self):
        # record loss frequently, but not all the time.
        if (self.n_training_steps + 1) % (
            self.cfg.wandb_log_frequency * self.cfg.eval_every_n_wandb_logs
        ) == 0:
            self.sae.eval()
            eval_metrics, _ = run_evals(
                sae=self.sae,
                activation_store=self.activation_store,
                model=self.model,
                eval_config=self.trainer_eval_config,
                model_kwargs=self.cfg.model_kwargs,
            )  # not calculating featurwise metrics here.

            # Remove eval metrics that are already logged during training
            eval_metrics.pop("metrics/explained_variance", None)
            eval_metrics.pop("metrics/explained_variance_std", None)
            eval_metrics.pop("metrics/l0", None)
            eval_metrics.pop("metrics/l1", None)
            eval_metrics.pop("metrics/mse", None)

            # Remove metrics that are not useful for wandb logging
            eval_metrics.pop("metrics/total_tokens_evaluated", None)

            W_dec_norm_dist = self.sae.W_dec.detach().float().norm(dim=1).cpu().numpy()
            eval_metrics["weights/W_dec_norms"] = wandb.Histogram(W_dec_norm_dist)  # type: ignore

            if self.sae.cfg.architecture == "standard":
                b_e_dist = self.sae.b_enc.detach().float().cpu().numpy()
                eval_metrics["weights/b_e"] = wandb.Histogram(b_e_dist)  # type: ignore
            elif self.sae.cfg.architecture == "gated":
                b_gate_dist = self.sae.b_gate.detach().float().cpu().numpy()
                eval_metrics["weights/b_gate"] = wandb.Histogram(b_gate_dist)  # type: ignore
                b_mag_dist = self.sae.b_mag.detach().float().cpu().numpy()
                eval_metrics["weights/b_mag"] = wandb.Histogram(b_mag_dist)  # type: ignore

            wandb.log(
                eval_metrics,
                step=self.n_training_steps,
            )
            self.sae.train()

    @torch.no_grad()
    def _build_sparsity_log_dict(self) -> dict[str, Any]:
        log_feature_sparsity = _log_feature_sparsity(self.feature_sparsity)
        wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())  # type: ignore
        return {
            "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
            "plots/feature_density_line_chart": wandb_histogram,
            "sparsity/below_1e-5": (self.feature_sparsity < 1e-5).sum().item(),
            "sparsity/below_1e-6": (self.feature_sparsity < 1e-6).sum().item(),
        }

    @torch.no_grad()
    def _reset_running_sparsity_stats(self) -> None:
        self.act_freq_scores = torch.zeros(
            self.cfg.d_sae,  # type: ignore
            device=self.cfg.device,
        )
        self.n_frac_active_tokens = 0

    @torch.no_grad()
    def _checkpoint_if_needed(self):
        if (
            self.checkpoint_thresholds
            and self.n_training_tokens > self.checkpoint_thresholds[0]
        ):
            self.save_checkpoint(checkpoint_name=str(self.n_training_tokens))
            self.checkpoint_thresholds.pop(0)

    @torch.no_grad()
    def _update_pbar(
        self,
        step_output: TrainStepOutput,
        pbar: tqdm,  # type: ignore
        update_interval: int = 100,
    ):
        if self.n_training_steps % update_interval == 0:
            loss_strs = " | ".join(
                f"{loss_name}: {_unwrap_item(loss_value):.5f}"
                for loss_name, loss_value in step_output.losses.items()
            )
            pbar.set_description(f"{self.n_training_steps}| {loss_strs}")
            pbar.update(update_interval * self.cfg.train_batch_size_tokens)

    def _begin_finetuning_if_needed(self):
        if (not self.started_fine_tuning) and (
            self.n_training_tokens > self.cfg.training_tokens
        ):
            self.started_fine_tuning = True

            # finetuning method should be set in the config
            # if not, then we don't finetune
            if not isinstance(self.cfg.finetuning_method, str):
                return

            for name, param in self.sae.named_parameters():
                if name in FINETUNING_PARAMETERS[self.cfg.finetuning_method]:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            self.finetuning = True


def _unwrap_item(item: float | torch.Tensor) -> float:
    return item.item() if isinstance(item, torch.Tensor) else item
