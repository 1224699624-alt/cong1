"""Native-start TSRS final-prior trainer.

This trainer is intentionally a thin nnU-Net wrapper around
``final_prior_plugin.StructuralPriorPlugin``.  The native checkpoint is loaded
for every A/B/C/D run.  A/B/C/D differ only in explicit branch switches; all
backbone calls receive an image plus an all-zero second channel.
"""
from __future__ import annotations

import copy
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

try:  # The workstation does not install nnU-Net; the server does.
    from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler  # noqa: F401
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    _NNUNET_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - exercised on local CI only
    _NNUNET_IMPORT_ERROR = exc

    class nnUNetTrainer:  # type: ignore[no-redef]
        """Import fallback so plugin/network unit tests work without nnU-Net."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "nnUNetTrainerFinalPrior requires nnunetv2; install/use the "
                "server environment before constructing the trainer"
            ) from _NNUNET_IMPORT_ERROR

from final_prior_plugin import (
    NATIVE_CHECKPOINT_SHA256,
    StructuralPriorPlugin,
    assert_native_checkpoint,
    configure_plugin_trainability,
    load_r351_output_refiner,
    load_state_dict_from_checkpoint,
    make_run_manifest,
    normalize_variant,
    sha256_file,
    update_manifest,
    variant_config,
    write_manifest_exclusive,
    zero_native_prior_channel,
)
from r349_parameter_gradient_gate import assign_gradients, route_parameter_gradients
from r351_dual_prior import local_terms


DATASET_NAME = "Dataset204_TSRS_RSNAEpiphysisMaturePrior2D"


class RatioPolyLRScheduler:
    """Polynomial schedule that preserves AdamW parameter-group LR ratios.

    nnU-Net's stock scheduler has historically treated all optimizer groups as
    one group.  The final matched runs need a backbone LR of 2e-6 and a prior
    LR of 2e-5 under the same schedule, so each group receives the same scalar
    polynomial multiplier while retaining its configured base LR.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lrs: list[float],
        max_steps: int,
        exponent: float = 0.9,
    ) -> None:
        if len(base_lrs) != len(optimizer.param_groups):
            raise ValueError("base_lrs must match optimizer parameter groups")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.optimizer = optimizer
        self.base_lrs = [float(value) for value in base_lrs]
        self.max_steps = int(max_steps)
        self.exponent = float(exponent)
        self.last_step = 0

    def step(self, current_step: int | None = None) -> None:
        if current_step is None:
            current_step = self.last_step + 1
        self.last_step = int(current_step)
        fraction = max(0.0, 1.0 - self.last_step / float(self.max_steps))
        multiplier = fraction ** self.exponent
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * multiplier

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_lrs": list(self.base_lrs),
            "max_steps": self.max_steps,
            "exponent": self.exponent,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.base_lrs = [float(value) for value in state_dict["base_lrs"]]
        self.max_steps = int(state_dict["max_steps"])
        self.exponent = float(state_dict["exponent"])
        self.last_step = int(state_dict.get("last_step", 0))
        self.step(self.last_step)


class FinalPriorNetwork(nn.Module):
    """Backbone wrapper with a strict zero native-prior input channel."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        variant: str = "D",
        output_residual_scale: float = 1.0,
        overlap_context_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.variant = normalize_variant(variant)
        branches = variant_config(self.variant)
        self.plugin = StructuralPriorPlugin(
            enable_input=branches["enable_input_prior"],
            enable_output=branches["enable_output_prior"],
            output_residual_scale=output_residual_scale,
            overlap_context_scale=overlap_context_scale,
        )

    @property
    def input_prior(self) -> nn.Module:
        return self.plugin.input_adapter

    @property
    def output_prior(self) -> nn.Module:
        return self.plugin.output_refiner

    @property
    def overlap_prior(self) -> nn.Module:
        return self.plugin.overlap_prior

    @property
    def decoder(self) -> Any:
        return getattr(self.backbone, "decoder", None)

    def set_training_mode(self, training_mode: str) -> dict[str, Any]:
        mode = str(training_mode).strip().lower()
        if mode not in {"adapter_only", "joint_finetune"}:
            raise ValueError("training_mode must be adapter_only or joint_finetune")
        if mode == "adapter_only" and self.variant == "A":
            raise ValueError("variant A has no active prior parameters; use joint_finetune")
        self.backbone.requires_grad_(mode == "joint_finetune")
        prior = configure_plugin_trainability(self.plugin, training_mode=mode)
        return {
            "training_mode": mode,
            "backbone_trainable": mode == "joint_finetune",
            **prior,
        }

    def active_backbone_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]

    def active_prior_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.plugin.parameters() if parameter.requires_grad]

    def native_input(self, image: torch.Tensor, seam: torch.Tensor) -> torch.Tensor:
        """Expose the input contract for teacher and unit-test audits."""

        return zero_native_prior_channel(image, seam)

    def forward(self, data: torch.Tensor) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]:
        """Forward image and seam with no access to a target/ground truth."""

        if data.ndim != 4 or data.shape[1] < 2:
            raise ValueError(f"final TSRS network expects BCHW with two channels, got {data.shape}")
        image, seam = data[:, :1], data[:, 1:2]
        overlap = self.plugin.overlap_probability(image, seam)
        adapted = self.plugin.adapt_input(image, seam, overlap)
        native = self.native_input(adapted, seam)
        raw_output = self.backbone(native)
        high = raw_output[0] if isinstance(raw_output, (list, tuple)) else raw_output
        if high.ndim != 4 or high.shape[1] < 2:
            raise ValueError(f"final TSRS network expects two output classes, got {high.shape}")

        if self.plugin.enable_output_prior and self.plugin.output_residual_scale > 0:
            margin = high[:, 1:2].float() - high[:, :1].float()
            # Add the residual to the original two-class logits.  This keeps
            # zero-init exactly identity while preserving output-prior grads.
            refined_margin = self.plugin.refine_output(
                image.float(), margin, seam.float(), overlap.float()
            )
            delta = refined_margin - margin
            refined = high + torch.cat((torch.zeros_like(delta), delta), dim=1).to(high.dtype)
        else:
            refined = high

        if isinstance(raw_output, tuple):
            return (refined, *raw_output[1:])
        if isinstance(raw_output, list):
            return [refined, *raw_output[1:]]
        return refined


class nnUNetTrainerFinalPrior(nnUNetTrainer):
    """Final native-start TSRS A/B/C/D trainer."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        if _NNUNET_IMPORT_ERROR is not None:  # pragma: no cover - local guard
            raise RuntimeError(
                "nnUNetTrainerFinalPrior requires nnunetv2 in the execution environment"
            ) from _NNUNET_IMPORT_ERROR
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.plans_manager.dataset_name != DATASET_NAME:
            raise RuntimeError(self.plans_manager.dataset_name)

        self.variant = normalize_variant(os.environ.get("FINAL_PRIOR_VARIANT", "D"))
        self.variant_name = variant_config(self.variant)["name"]
        self.training_mode = os.environ.get("FINAL_PRIOR_TRAINING_MODE", "joint_finetune").strip().lower()
        if self.training_mode not in {"adapter_only", "joint_finetune"}:
            raise ValueError("FINAL_PRIOR_TRAINING_MODE must be adapter_only or joint_finetune")
        if self.training_mode == "adapter_only" and self.variant == "A":
            raise ValueError("A/native_continue has no active prior parameters; adapter_only is invalid")

        self.backbone_lr = float(os.environ.get("FINAL_PRIOR_BACKBONE_LR", "2e-6"))
        self.prior_lr = float(os.environ.get("FINAL_PRIOR_ADAPTER_LR", "2e-5"))
        self.weight_decay_final = float(os.environ.get("FINAL_PRIOR_WEIGHT_DECAY", "1e-4"))
        self.num_epochs = int(os.environ.get("FINAL_PRIOR_EPOCHS", "2"))
        self.num_iterations_per_epoch = int(os.environ.get("FINAL_PRIOR_ITERATIONS", "100"))
        self.num_val_iterations_per_epoch = int(os.environ.get("FINAL_PRIOR_VAL_ITERATIONS", "25"))
        self.seed = int(os.environ.get("FINAL_PRIOR_SEED", "3515"))
        self.max_aux_ratio = float(os.environ.get("FINAL_PRIOR_MAX_AUX_RATIO", ".15"))
        self.structural_weight = float(os.environ.get("FINAL_PRIOR_STRUCTURAL_WEIGHT", ".035"))
        self.preserve_weight = float(os.environ.get("FINAL_PRIOR_PRESERVE_WEIGHT", ".10"))
        self.aux_ramp_epochs = int(os.environ.get("FINAL_PRIOR_AUX_RAMP_EPOCHS", "5"))
        self.seam_threshold = float(os.environ.get("FINAL_PRIOR_SEAM_THRESHOLD", ".25"))
        self.output_residual_scale = float(os.environ.get("FINAL_PRIOR_OUTPUT_RESIDUAL_SCALE", "1.0"))
        self.overlap_context_scale = float(os.environ.get("FINAL_PRIOR_OVERLAP_CONTEXT_SCALE", "1.0"))
        self.audit_only = os.environ.get("FINAL_PRIOR_AUDIT_ONLY", "false").lower() == "true"
        self.use_autocast = os.environ.get("FINAL_PRIOR_AUTOCAST", "true").lower() == "true"
        self.teacher: nn.Module | None = None
        self._manifest_path: Path | None = None
        self._trainability: dict[str, Any] | None = None

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        if int(num_input_channels) != 2:
            raise ValueError(f"final TSRS native contract requires 2 input channels, got {num_input_channels}")
        base = nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        return FinalPriorNetwork(
            base,
            variant=os.environ.get("FINAL_PRIOR_VARIANT", "D"),
            output_residual_scale=float(os.environ.get("FINAL_PRIOR_OUTPUT_RESIDUAL_SCALE", "1.0")),
            overlap_context_scale=float(os.environ.get("FINAL_PRIOR_OVERLAP_CONTEXT_SCALE", "1.0")),
        )

    def _set_batch_size_and_oversample(self):
        super()._set_batch_size_and_oversample()
        # The teacher and full-resolution refiner make the matched runs
        # memory-bound.  A/B/C/D still share this exact batch contract.
        self.batch_size = min(self.batch_size, int(os.environ.get("FINAL_PRIOR_BATCH_SIZE", "1")))

    @staticmethod
    def _env_path(*names: str) -> Path | None:
        for name in names:
            value = os.environ.get(name)
            if value:
                return Path(value)
        return None

    def initialize(self):
        if self.was_initialized:
            return
        if os.environ.get("nnUNet_compile", "").lower() != "false":
            raise RuntimeError("nnUNetTrainerFinalPrior requires nnUNet_compile=false for native loading")

        super().initialize()
        if not isinstance(self.network, FinalPriorNetwork):
            raise TypeError(f"unexpected final network type: {type(self.network)!r}")

        native_path = self._env_path("FINAL_PRIOR_NATIVE_CHECKPOINT", "FINAL_PRIOR_TSRS_INIT")
        if native_path is None:
            raise RuntimeError("FINAL_PRIOR_NATIVE_CHECKPOINT is required")
        expected = os.environ.get("FINAL_PRIOR_NATIVE_SHA256", NATIVE_CHECKPOINT_SHA256)
        native_source = assert_native_checkpoint(native_path, expected_sha256=expected)
        state = load_state_dict_from_checkpoint(native_path, preferred="network_weights", map_location=self.device)
        self.network.backbone.load_state_dict(state, strict=True)

        # The fixed teacher is copied only after the native state is loaded and
        # always sees the same zero-prior input contract as the student.
        self.teacher = copy.deepcopy(self.network.backbone).eval().requires_grad_(False)
        self.network.set_training_mode(self.training_mode)
        self._trainability = {
            "training_mode": self.training_mode,
            "backbone_trainable": self.training_mode == "joint_finetune",
            "active_prior_parameter_names": self.network.plugin.active_parameter_names(),
        }

        overlap_path = self._env_path("FINAL_PRIOR_OVERLAP_CHECKPOINT", "R351_OVERLAP_CHECKPOINT")
        if overlap_path is None:
            # Dataset-native TSRS core has no RAM-trained overlap context.
            self.network.plugin.overlap_context_scale = 0.0
        if overlap_path is not None and (self.network.plugin.enable_input_prior or self.network.plugin.enable_output_prior):
            overlap_state = load_state_dict_from_checkpoint(
                overlap_path, preferred="model", map_location=self.device
            )
            self.network.overlap_prior.load_state_dict(overlap_state, strict=True)
            self.network.overlap_prior.eval().requires_grad_(False)

        refiner_path = self._env_path("FINAL_PRIOR_OUTPUT_REFINER_CHECKPOINT", "FINAL_PRIOR_RAM_INIT", "R351_RAM_INIT")
        refiner_report: dict[str, Any] | None = None
        if self.network.plugin.enable_output_prior and refiner_path is not None:
            refiner_report = load_r351_output_refiner(
                self.network.output_prior,
                refiner_path,
                identity=True,
                map_location="cpu",
            )

        output_folder = Path(self.output_folder)
        manifest = make_run_manifest(
            output_folder=output_folder,
            dataset=self.plans_manager.dataset_name,
            split="train/val (fixed split; test excluded from selection)",
            variant=self.variant,
            training_mode=self.training_mode,
            seed=self.seed,
            native_checkpoint=native_path,
            native_source=native_source,
            overlap_checkpoint=overlap_path,
            output_refiner_checkpoint=refiner_path,
            config={
                "epochs": self.num_epochs,
                "iterations_per_epoch": self.num_iterations_per_epoch,
                "val_iterations_per_epoch": self.num_val_iterations_per_epoch,
                "backbone_lr": self.backbone_lr,
                "prior_lr": self.prior_lr,
                "optimizer": "AdamW",
                "scheduler": "ratio-preserving polynomial",
                "output_residual_scale": self.output_residual_scale,
                "overlap_context_scale": self.overlap_context_scale,
                "tsrs_overlap_strong_loss": False,
                "teacher": "fixed native backbone; zero native prior channel",
                "trainability": self._trainability,
                "output_refiner_transfer": refiner_report,
            },
        )
        self._manifest_path = write_manifest_exclusive(output_folder, manifest)

        identity = self._identity_audit()
        (output_folder / "final_prior_initialization.json").write_text(
            json.dumps({
                "native_source": native_source,
                "identity": identity,
                "variant": self.variant,
                "training_mode": self.training_mode,
                "output_refiner_transfer": refiner_report,
                "overlap_strong_loss": "disabled: TSRS has no membership observations",
                "forbidden_backbone_sources": ["R350", "R332"],
            }, indent=2),
            encoding="utf-8",
        )
        if not identity["backbone_loaded_exactly"]:
            raise RuntimeError(identity)

    def _identity_audit(self) -> dict[str, Any]:
        """Check source loading and branch identity without a target tensor."""

        if self.teacher is None:
            raise RuntimeError("teacher must be initialized before identity audit")
        max_difference = 0.0
        for student, teacher in zip(self.network.backbone.parameters(), self.teacher.parameters()):
            max_difference = max(max_difference, float((student.detach() - teacher.detach()).abs().max()))
        return {
            "backbone_loaded_exactly": max_difference == 0.0,
            "backbone_teacher_max_abs_difference": max_difference,
            "native_prior_channel": "constant zero",
            "active_prior_parameters": self.network.plugin.active_parameter_names(),
            "variant": self.variant,
        }

    def configure_optimizers(self):
        if not isinstance(self.network, FinalPriorNetwork):
            raise TypeError("network has not been initialized as FinalPriorNetwork")
        if self._trainability is None:
            self.network.set_training_mode(self.training_mode)
        backbone = self.network.active_backbone_parameters()
        prior = self.network.active_prior_parameters()
        if not backbone and not prior:
            raise RuntimeError("final prior run has no trainable parameters")
        groups: list[dict[str, Any]] = []
        if backbone:
            groups.append({"params": backbone, "lr": self.backbone_lr, "name": "backbone"})
        if prior:
            groups.append({"params": prior, "lr": self.prior_lr, "name": "prior_adapter"})
        optimizer = torch.optim.AdamW(groups, weight_decay=self.weight_decay_final)
        scheduler = RatioPolyLRScheduler(
            optimizer,
            [float(group["lr"]) for group in groups],
            self.num_epochs,
        )
        return optimizer, scheduler

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.teacher is not None:
            self.teacher.eval()
        self.network.overlap_prior.eval()

    def _autocast_context(self):
        if not self.use_autocast or self.device.type != "cuda":
            return nullcontext()
        # bfloat16 avoids a GradScaler around the parameter-space route and is
        # stable for the full-resolution 1536x1024 native plan.
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    @staticmethod
    def _move_target(target: Any, device: torch.device) -> Any:
        if isinstance(target, (list, tuple)):
            return type(target)(value.to(device, non_blocking=True) for value in target)
        return target.to(device, non_blocking=True)

    def _forward_teacher(self, data: torch.Tensor) -> torch.Tensor:
        if self.teacher is None:
            raise RuntimeError("native teacher is not initialized")
        with torch.no_grad():
            teacher_output = self.teacher(
                zero_native_prior_channel(data[:, :1], data[:, 1:2])
            )
            return teacher_output[0] if isinstance(teacher_output, (list, tuple)) else teacher_output

    def train_step(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._move_target(batch["target"], self.device)
        high_target = target[0] if isinstance(target, (list, tuple)) else target
        valid = (high_target[:, :1] >= 0).float()
        observed = (high_target[:, :1] > 0).float()
        self.optimizer.zero_grad(set_to_none=True)

        with self._autocast_context():
            output = self.network(data)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            base = self.loss(output, target)
            with torch.no_grad():
                teacher_logits = self._forward_teacher(data)
                teacher_probability = torch.softmax(teacher_logits.float(), dim=1)[:, 1:2]
                overlap_probability = self.network.plugin.overlap_probability(
                    data[:, :1], data[:, 1:2]
                )

            margin = logits[:, 1:2].float() - logits[:, :1].float()
            terms, states = local_terms(
                margin,
                observed,
                teacher_probability,
                data[:, 1:2],
                overlap_probability,
                valid,
                membership_observed=False,
            )

        trainable = [parameter for parameter in self.network.parameters() if parameter.requires_grad]
        trainable_named = [(name, parameter) for name, parameter in self.network.named_parameters()
                           if parameter.requires_grad]
        routing = []
        routed_gradients: list[torch.Tensor | None] = []
        if self.variant == "A":
            # A is a real native_continue: every backbone parameter is updated
            # with the same base loss/schedule and no prior parameter is active.
            if self.network.active_prior_parameters():
                raise RuntimeError("variant A unexpectedly has active prior parameters")
            base.backward()
            routed_gradients = [parameter.grad for _, parameter in trainable_named]
            if not self.audit_only:
                torch.nn.utils.clip_grad_norm_(trainable, 12)
                self.optimizer.step()
        else:
            ramp = min(1.0, (self.current_epoch + 1) / float(self.aux_ramp_epochs))
            auxiliaries: list[tuple[str, torch.Tensor, float]] = []
            if self.network.plugin.enable_input_prior:
                auxiliaries.append(("separation", terms["separation"], self.structural_weight * ramp))
            # TSRS has no membership observations.  Keep the term out of the
            # route entirely so strong overlap supervision cannot be fabricated.
            if self.network.plugin.enable_input_prior or self.network.plugin.enable_output_prior:
                auxiliaries.append(("preserve", terms["preserve"], self.preserve_weight * ramp))
            if not auxiliaries:
                raise RuntimeError(f"variant {self.variant} has no routed prior losses")
            gradients, routing = route_parameter_gradients(
                self.network.parameters(),
                # Use the exact same nnU-Net base objective as native_continue;
                # local_terms supplies only the R351 auxiliary losses.
                base,
                auxiliaries,
                max_aux_ratio=self.max_aux_ratio,
            )
            routed_gradients = gradients
            if not self.audit_only:
                assign_gradients(self.network.parameters(), gradients)
                torch.nn.utils.clip_grad_norm_(trainable, 12)
                self.optimizer.step()

        identity_probability_max_abs = float(
            (torch.softmax(logits.detach().float(), dim=1)[:, 1:2]
             - teacher_probability.detach()).abs().max()
        )
        finite_gradients = bool(all(
            gradient is None or torch.isfinite(gradient).all()
            for gradient in routed_gradients
        ))
        gradient_norms = {
            name: float(gradient.detach().norm())
            for (name, _), gradient in zip(trainable_named, routed_gradients)
            if gradient is not None
        }
        active_gradient_norm = float(sum(gradient_norms.values()))
        branch_gradient_norms = {
            "backbone": float(sum(value for name, value in gradient_norms.items()
                                   if name.startswith("backbone."))),
            "input_prior": float(sum(value for name, value in gradient_norms.items()
                                      if ".input_adapter." in name)),
            "output_prior": float(sum(value for name, value in gradient_norms.items()
                                       if ".output_refiner." in name)),
        }
        result = {
            "loss": base.detach().cpu().numpy(),
            "base_loss": base.detach().cpu().numpy(),
            "separation_loss": terms["separation"].detach().cpu().numpy(),
            "overlap_loss": np.asarray(0.0, dtype=np.float32),
            "preserve_loss": terms["preserve"].detach().cpu().numpy(),
            "overlap_strong_loss_disabled": np.asarray(1.0, dtype=np.float32),
            "native_prior_channel_max_abs": np.asarray(0.0, dtype=np.float32),
            "identity_probability_max_abs": np.asarray(identity_probability_max_abs, dtype=np.float32),
            "finite_gradients": np.asarray(float(finite_gradients), dtype=np.float32),
            "active_gradient_norm": np.asarray(active_gradient_norm, dtype=np.float32),
            "branch_gradient_norms": branch_gradient_norms,
            **{key: np.asarray(value, dtype=np.float32) if isinstance(value, (int, float))
               else value for key, value in states.items()},
            "routing_rows": routing,
        }
        return result

    def on_train_epoch_end(self, outputs):
        # nnU-Net's collator accepts scalar/tensor values only. Keep routing
        # diagnostics in our JSONL history while giving the base logger the
        # scalar training fields it can aggregate.
        scalar_outputs = [
            {key: value for key, value in output.items()
             if not isinstance(value, (dict, list, tuple))}
            for output in outputs
        ]
        super().on_train_epoch_end(scalar_outputs)
        if not outputs:
            return
        row: dict[str, Any] = {"epoch": int(self.current_epoch), "variant": self.variant}
        for key in outputs[0]:
            if key in {"routing_rows", "branch_gradient_norms"}:
                continue
            values = [value[key] for value in outputs if key in value]
            row[key] = float(np.mean([float(np.asarray(value)) for value in values]))
        with (Path(self.output_folder) / "final_prior_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def on_epoch_end(self):
        super().on_epoch_end()
        if self._manifest_path is None:
            return
        metric = None
        try:
            metric = float(self.logger.get_value("ema_fg_dice", step=-1))
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        updates: dict[str, Any] = {
            "completed_epoch": int(self.current_epoch),
            "last_validation_ema_fg_dice": metric,
            "test_used_for_selection": False,
        }
        best = Path(self.output_folder) / "checkpoint_best.pth"
        if best.is_file():
            payload = torch.load(best, map_location="cpu", weights_only=False)
            selected_epoch = payload.get("current_epoch", self.current_epoch)
            updates.update({"selected_checkpoint": str(best), "selected_checkpoint_sha256": sha256_file(best),
                            "selected_epoch": int(selected_epoch)})
        update_manifest(self._manifest_path, updates)

    def on_train_end(self):
        super().on_train_end()
        if self._manifest_path is not None:
            update_manifest(self._manifest_path, {"status": "complete" if not self.audit_only else "audit_complete"})

    def run_training(self):
        if not self.audit_only:
            return super().run_training()
        # One real batch is enough to exercise source loading, no-GT forward,
        # zero native channel and R351 gradient routing before a long run.
        self.on_train_start()
        self.on_epoch_start()
        self.on_train_epoch_start()
        batch = next(iter(self.dataloader_train))
        result = self.train_step(batch)
        branch = result["branch_gradient_norms"]
        if self.variant == "A":
            branch_ok = branch["backbone"] > 0 and not self.network.plugin.active_parameter_names()
        elif self.variant == "B":
            branch_ok = branch["input_prior"] > 0 and branch["output_prior"] == 0
        elif self.variant == "C":
            branch_ok = branch["output_prior"] > 0 and branch["input_prior"] == 0
        else:
            branch_ok = branch["input_prior"] > 0 and branch["output_prior"] > 0
        audit = {
            "passed": bool(
                float(result["identity_probability_max_abs"]) <= 1e-6
                and bool(result["finite_gradients"])
                and float(result["active_gradient_norm"]) > 0
                and branch_ok
            ),
            "variant": self.variant,
            "training_mode": self.training_mode,
            "active_prior_parameters": self.network.plugin.active_parameter_names(),
            "routing_rows": [vars(row) for row in result["routing_rows"]],
            "native_prior_channel_max_abs": float(result["native_prior_channel_max_abs"]),
            "identity_probability_max_abs": float(result["identity_probability_max_abs"]),
            "finite_gradients": bool(result["finite_gradients"]),
            "active_gradient_norm": float(result["active_gradient_norm"]),
            "branch_gradient_norms": branch,
            "overlap_strong_loss_disabled": bool(result["overlap_strong_loss_disabled"]),
        }
        (Path(self.output_folder) / "final_prior_smoke.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        self.on_train_end()
        if not audit["passed"]:
            raise RuntimeError(audit)


__all__ = [
    "DATASET_NAME",
    "FinalPriorNetwork",
    "RatioPolyLRScheduler",
    "nnUNetTrainerFinalPrior",
]
