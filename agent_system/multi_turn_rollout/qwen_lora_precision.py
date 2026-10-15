"""Strict BF16-base/FP32-LoRA support for audited Qwen FSDP2 runs.

VERL's generic FSDP LoRA builder casts PEFT adapters to the frozen base
model's dtype.  That is necessary for FSDP1 flat parameters, but FSDP2 keeps
parameters as individual DTensors and supports FP32 trainable parameters in a
group whose frozen parameters are BF16.  This external module installs a
CUDA/FSDP2 language-model engine that preserves PEFT's FP32 adapters and
audits the precision and targeting assumptions at runtime.

The module is loaded only by the modern ALFWorld LoRA launcher through
``VERL_USE_EXTERNAL_MODULES``.  It deliberately fails for unaudited models or
adapter layouts instead of silently applying a partially compatible patch.
"""

from __future__ import annotations

import gc
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from verl.utils.fs import copy_to_local
from verl.utils.metric import Metric
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


@dataclass(frozen=True)
class _LoRAArchitectureProfile:
    display_name: str
    language_layers: int
    target_modules: int
    targets_per_rank: int
    target_kinds: Counter
    target_re: re.Pattern[str]


_ARCHITECTURE_PROFILES = {
    "qwen3": _LoRAArchitectureProfile(
        display_name="Qwen3",
        language_layers=36,
        target_modules=252,
        targets_per_rank=2_064_384,
        target_kinds=Counter(
            {
                "mlp.down_proj": 36,
                "mlp.gate_proj": 36,
                "mlp.up_proj": 36,
                "self_attn.k_proj": 36,
                "self_attn.o_proj": 36,
                "self_attn.q_proj": 36,
                "self_attn.v_proj": 36,
            }
        ),
        target_re=re.compile(
            r"\.model\.layers\.(?P<layer>\d+)\."
            r"(?P<kind>"
            r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
            r"mlp\.(?:gate_proj|up_proj|down_proj)"
            r")$"
        ),
    ),
    "qwen3_5": _LoRAArchitectureProfile(
        display_name="Qwen3.5",
        language_layers=32,
        target_modules=248,
        targets_per_rank=2_704_896,
        target_kinds=Counter(
            {
                "linear_attn.in_proj_a": 24,
                "linear_attn.in_proj_b": 24,
                "linear_attn.in_proj_qkv": 24,
                "linear_attn.in_proj_z": 24,
                "linear_attn.out_proj": 24,
                "mlp.down_proj": 32,
                "mlp.gate_proj": 32,
                "mlp.up_proj": 32,
                "self_attn.k_proj": 8,
                "self_attn.o_proj": 8,
                "self_attn.q_proj": 8,
                "self_attn.v_proj": 8,
            }
        ),
        target_re=re.compile(
            r"\.language_model\.layers\.(?P<layer>\d+)\."
            r"(?P<kind>"
            r"linear_attn\.(?:in_proj_a|in_proj_b|in_proj_qkv|in_proj_z|out_proj)|"
            r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
            r"mlp\.(?:gate_proj|up_proj|down_proj)"
            r")$"
        ),
    ),
}
_FORBIDDEN_TRAINABLE_FRAGMENTS = (
    ".visual.",
    ".mtp.",
    ".embed_tokens.",
    ".lm_head.",
)
_META_PPO_MINIBATCH_UPDATE = (
    os.environ.get("LAMER_META_PPO_MINIBATCH_UPDATE", "0") == "1"
)
_META_INTEGRATION_AUDIT_DIR = os.environ.get("LAMER_META_INTEGRATION_AUDIT_DIR")


def _local_parameter_tensor(parameter: torch.Tensor) -> torch.Tensor:
    """Return the rank-local tensor for ordinary tensors or DTensors."""
    value = parameter.detach()
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _json_numbers(value):
    """Convert nested metric containers to JSON-safe Python values."""
    if isinstance(value, Metric):
        return {
            "aggregation": value.aggregation.value,
            "values": [_json_numbers(item) for item in value.values],
            "aggregate": _json_numbers(value.aggregate()),
        }
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_numbers(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _nested_row_sums(value: torch.Tensor) -> list[float]:
    return [float(row.detach().float().sum().cpu()) for row in value.unbind()]


def _process_rss_gib() -> float | None:
    """Read the current process RSS without adding a psutil dependency."""
    try:
        with open("/proc/self/statm", encoding="utf-8") as stream:
            resident_pages = int(stream.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def _install_vllm_host_cache_release() -> None:
    """Return unused sleep-mode pinned buffers to the host after vLLM wakes.

    vLLM sleep level 1 must copy the frozen base weights to pinned host memory
    while the actor owns the GPU. ``CuMemAllocator.wake_up`` drops the backup
    tensor after restoring those weights, but PyTorch's pinned allocator may
    cache the freed block. On this 62-GiB host that cache competes with the
    resident ALFWorld environments during the next rollout.

    Emptying the unused host allocator cache after the synchronous wake copy
    preserves the level-1 LoRA protocol while trading a later pinned-allocation
    cost for bounded resident RAM. This is intentionally opt-in and pinned to
    the compatibility environment's audited private PyTorch API.
    """
    from vllm.device_allocator.sleep_mode_backend import CuMemBackend

    original_resume = CuMemBackend.resume
    if getattr(original_resume, "_lamer_releases_host_cache", False):
        return

    empty_host_cache = getattr(torch._C, "_accelerator_emptyHostCache", None)
    if not callable(empty_host_cache):
        raise RuntimeError(
            "RELEASE_VLLM_HOST_CACHE_AFTER_WAKE requires "
            "torch._C._accelerator_emptyHostCache in the pinned environment"
        )

    @wraps(original_resume)
    def resume_and_release(self, tags=None):
        result = original_resume(self, tags)
        if tags is None or "weights" in tags:
            # The allocator's wake-up copy is synchronous and has already
            # cleared every cpu_backup_tensor reference before returning.
            gc.collect()
            rss_before = _process_rss_gib()
            empty_host_cache()
            rss_after = _process_rss_gib()
            if rss_before is None or rss_after is None:
                print("Released unused vLLM pinned host cache after weight wake.", flush=True)
            else:
                print(
                    "Released unused vLLM pinned host cache after weight wake: "
                    f"worker RSS {rss_before:.3f} -> {rss_after:.3f} GiB "
                    f"({max(0.0, rss_before - rss_after):.3f} GiB returned).",
                    flush=True,
                )
        return result

    resume_and_release._lamer_releases_host_cache = True
    CuMemBackend.resume = resume_and_release


def _adapter_parent(parameter_name: str) -> str:
    for marker in (".lora_A.", ".lora_B."):
        if marker in parameter_name:
            return parameter_name.split(marker, 1)[0]
    raise RuntimeError(f"Unexpected trainable non-LoRA parameter: {parameter_name}")


class QwenFP32LoRAFSDPEngine(FSDPEngineWithLMHead):
    """FSDP2 engine that keeps only audited Qwen LoRA parameters in FP32."""

    _optimizer_precision_audited = False

    def train_mode(self, **kwargs):
        """Avoid redundant context-exit zeroing for each meta PPO minibatch.

        ``BaseEngine.train_batch`` already zeros once before backward. VERL's
        nested training contexts normally call ``optimizer_zero_grad`` again on
        each successful exit. In the meta protocol we suppress those cleanup
        calls and release gradients explicitly after each PPO-minibatch
        optimizer step. Exceptions retain the upstream emergency cleanup.
        """
        if _META_PPO_MINIBATCH_UPDATE:
            kwargs.setdefault("zero_grad_on_exit", False)
        return super().train_mode(**kwargs)

    def _build_lora_module(self, module):
        if self.engine_config.strategy != "fsdp2":
            raise RuntimeError("FP32 LoRA requires actor.strategy=fsdp2")
        model_type = self.model_config.hf_config.model_type
        if model_type not in _ARCHITECTURE_PROFILES:
            raise RuntimeError(
                "The FP32 LoRA engine only supports audited model types "
                f"{sorted(_ARCHITECTURE_PROFILES)}, got {model_type!r}"
            )
        self._lora_architecture_profile = _ARCHITECTURE_PROFILES[model_type]

        module.enable_input_require_grads()
        lora_adapter_path = getattr(self.model_config, "lora_adapter_path", None)
        if lora_adapter_path is not None:
            local_adapter_path = copy_to_local(
                lora_adapter_path,
                use_shm=self.model_config.use_shm,
            )
            module = PeftModel.from_pretrained(
                module,
                local_adapter_path,
                is_trainable=True,
                autocast_adapter_dtype=True,
            )
            peft_config = module.peft_config["default"]
            if isinstance(peft_config.task_type, str):
                peft_config.task_type = TaskType.CAUSAL_LM
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.model_config.lora_rank,
                lora_alpha=self.model_config.lora_alpha,
                target_modules=convert_to_regular_types(self.model_config.target_modules),
                target_parameters=convert_to_regular_types(self.model_config.target_parameters),
                exclude_modules=convert_to_regular_types(self.model_config.exclude_modules),
                lora_dropout=0.0,
                bias="none",
                init_lora_weights=True,
                use_rslora=False,
            )
            module = get_peft_model(
                module,
                lora_config,
                autocast_adapter_dtype=True,
            )

        self._audit_lora_model(module, stage="before FSDP2")
        return module

    def _build_fsdp_module(self, module):
        module = super()._build_fsdp_module(module)
        if self._is_lora:
            self._audit_lora_model(module, stage="after FSDP2")
        return module

    def _pilot_append(self, payload: dict) -> None:
        if not _META_INTEGRATION_AUDIT_DIR:
            return
        destination = Path(_META_INTEGRATION_AUDIT_DIR)
        destination.mkdir(parents=True, exist_ok=True)
        rank = int(getattr(self, "rank", 0))
        with (destination / f"worker_rank_{rank}.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(_json_numbers(payload), sort_keys=True) + "\n")

    def optimizer_zero_grad(self):
        if _META_INTEGRATION_AUDIT_DIR and hasattr(self, "_pilot_events"):
            self._pilot_events.append("zero_grad")
        return super().optimizer_zero_grad()

    def forward_step(self, micro_batch, loss_function, forward_only):
        if (
            _META_INTEGRATION_AUDIT_DIR
            and not forward_only
            and hasattr(self, "_pilot_events")
        ):
            fragment_index = len(self._pilot_fragments)
            self._pilot_events.append(f"forward_backward_fragment:{fragment_index + 1}")
            parameter_versions = [
                int(parameter._version)
                for parameter in self.module.parameters()
                if parameter.requires_grad
            ]
            adam_steps = []
            for parameter in self.module.parameters():
                if not parameter.requires_grad:
                    continue
                step = self.optimizer.state.get(parameter, {}).get("step", 0.0)
                adam_steps.append(float(step.detach().cpu()) if torch.is_tensor(step) else float(step))
            row_indices = (
                micro_batch["meta_record_row_index"].detach().cpu().tolist()
                if "meta_record_row_index" in micro_batch
                else []
            )
            input_ids = micro_batch.get("input_ids")
            if input_ids is None:
                input_token_count = 0
            elif input_ids.is_nested:
                input_token_count = int(input_ids.offsets().diff().sum().cpu())
            else:
                attention_mask = micro_batch.get("attention_mask")
                input_token_count = int(
                    attention_mask.sum().detach().cpu()
                    if attention_mask is not None
                    else input_ids.numel()
                )
            self._pilot_fragments.append(
                {
                    "fragment_index": fragment_index,
                    "row_indices": row_indices,
                    "input_token_count": input_token_count,
                    "parameter_version_min": min(parameter_versions, default=0),
                    "parameter_version_max": max(parameter_versions, default=0),
                    "adam_step_min": min(adam_steps, default=0.0),
                    "adam_step_max": max(adam_steps, default=0.0),
                }
            )
        return super().forward_step(micro_batch, loss_function, forward_only)

    def train_batch(self, data, loss_function):
        if not _META_INTEGRATION_AUDIT_DIR:
            return super().train_batch(data, loss_function)

        minibatch_index = int(getattr(self, "_pilot_minibatch_index", 0))
        self._pilot_events = []
        self._pilot_fragments = []
        self._pilot_scheduler_steps_this_minibatch = 0
        trainable = [
            parameter for parameter in self.module.parameters() if parameter.requires_grad
        ]
        before = [_local_parameter_tensor(parameter).clone() for parameter in trainable]
        row_indices = data["meta_record_row_index"].detach().cpu().tolist()
        attached_fields = {}
        for field in (
            "meta_trajectory_utility",
            "meta_base_advantage",
            "meta_final_unweighted_advantage",
            "meta_trajectory_weight",
            "meta_trainable_token_count",
            "meta_ppo_minibatch_scale",
        ):
            attached_fields[field] = data[field].detach().cpu().tolist()
        old_log_prob_sums = _nested_row_sums(data["old_log_probs"])
        response_token_counts = [
            int(row.detach().sum().cpu()) for row in data["response_mask"].unbind()
        ]
        memory_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()

        output = super().train_batch(data, loss_function)

        update_squared = 0.0
        for parameter, snapshot in zip(trainable, before, strict=True):
            difference = _local_parameter_tensor(parameter).float() - snapshot.float()
            update_squared += float(difference.square().sum().cpu())
        del before
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        grad_norm_values = output.get("metrics", {}).get("grad_norm", [])
        if isinstance(grad_norm_values, (int, float)):
            grad_norm = float(grad_norm_values)
        elif grad_norm_values:
            grad_norm = float(grad_norm_values[-1])
        else:
            grad_norm = float("nan")
        clip_threshold = float(self.optimizer_config.clip_grad)
        payload = {
            "kind": "ppo_minibatch",
            "minibatch_index": minibatch_index,
            "row_indices": row_indices,
            "response_token_counts": response_token_counts,
            "real_rows": sum(count > 0 for count in response_token_counts),
            "padding_rows": sum(count == 0 for count in response_token_counts),
            "attached_fields": attached_fields,
            "old_log_prob_sums": old_log_prob_sums,
            "loss": float(sum(output.get("loss", []))),
            "metrics": output.get("metrics", {}),
            "pre_clip_grad_norm": grad_norm,
            "grad_clip_threshold": clip_threshold,
            "grad_clipping_activated": grad_norm > clip_threshold,
            "parameter_update_norm": update_squared**0.5,
            "scheduler_steps": self._pilot_scheduler_steps_this_minibatch,
            "events": self._pilot_events,
            "fragments": self._pilot_fragments,
            "dynamic_batching": bool(self.engine_config.use_dynamic_bsz),
            "max_token_len_per_gpu": int(
                self.engine_config.max_token_len_per_gpu or 0
            ),
            "cuda_allocated_before": memory_before,
            "cuda_reserved_before": reserved_before,
            "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved": torch.cuda.max_memory_reserved(),
            "global_free_vram": free_bytes,
            "global_total_vram": total_bytes,
        }
        self._pilot_append(payload)
        self._pilot_minibatch_index = minibatch_index + 1
        return output

    def optimizer_step(self):
        if _META_INTEGRATION_AUDIT_DIR and hasattr(self, "_pilot_events"):
            self._pilot_events.append("global_clip")
        if self._is_lora and not self._optimizer_precision_audited:
            trainable_with_grad = [
                (name, parameter)
                for name, parameter in self.module.named_parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not trainable_with_grad:
                raise RuntimeError("No LoRA gradients were produced before the optimizer step")
            bad_grad_dtypes = {
                name: str(parameter.grad.dtype)
                for name, parameter in trainable_with_grad
                if parameter.grad.dtype != torch.float32
            }
            if bad_grad_dtypes:
                raise RuntimeError(
                    "LoRA optimizer-facing gradients are not FP32: "
                    f"{list(bad_grad_dtypes.items())[:5]}"
                )

        grad_norm = super().optimizer_step()

        if _META_INTEGRATION_AUDIT_DIR and hasattr(self, "_pilot_events"):
            self._pilot_events.append("optimizer_step")

        if _META_PPO_MINIBATCH_UPDATE:
            # Pinned VERL normally advances the scheduler only on the final PPO
            # minibatch of a rollout. Checkpoint 4B defines scheduler cadence at
            # the optimizer boundary instead. Advance immediately after every
            # Adam step, and let the worker's final callback only report this LR.
            self._meta_last_scheduler_lr = super().lr_scheduler_step()
            if _META_INTEGRATION_AUDIT_DIR and hasattr(self, "_pilot_events"):
                self._pilot_events.append("scheduler_step")
                self._pilot_scheduler_steps_this_minibatch += 1

        if self._is_lora and not self._optimizer_precision_audited:
            missing_state = []
            bad_state_dtypes = []
            for name, parameter in self.module.named_parameters():
                if not parameter.requires_grad:
                    continue
                state = self.optimizer.state.get(parameter, {})
                if "exp_avg" not in state or "exp_avg_sq" not in state:
                    missing_state.append(name)
                    continue
                for state_name in ("exp_avg", "exp_avg_sq"):
                    if state[state_name].dtype != torch.float32:
                        bad_state_dtypes.append(
                            (name, state_name, str(state[state_name].dtype))
                        )
            if missing_state:
                raise RuntimeError(
                    "Adam state was not created for all LoRA parameters: "
                    f"{missing_state[:5]}"
                )
            if bad_state_dtypes:
                raise RuntimeError(
                    "LoRA Adam moments are not FP32: "
                    f"{bad_state_dtypes[:5]}"
                )
            self._optimizer_precision_audited = True
            if self.rank == 0:
                print(
                    f"{self._lora_architecture_profile.display_name} LoRA optimizer "
                    "audit passed: FP32 gradients, "
                    "parameters, exp_avg, and exp_avg_sq."
                )

        if _META_PPO_MINIBATCH_UPDATE:
            # Release gradient storage without a second optimizer.zero_grad()
            # event. The next PPO minibatch still begins with BaseEngine's
            # single explicit zero_grad before any backward fragment.
            for parameter in self.module.parameters():
                parameter.grad = None

        return grad_norm

    def lr_scheduler_step(self):
        """Report the LR already advanced after the final PPO-minibatch step."""
        if not _META_PPO_MINIBATCH_UPDATE:
            return super().lr_scheduler_step()
        if not hasattr(self, "_meta_last_scheduler_lr"):
            raise RuntimeError(
                "Meta PPO scheduler reporting occurred before an optimizer step"
            )
        return self._meta_last_scheduler_lr

    def _audit_lora_model(self, module, *, stage: str) -> None:
        profile = self._lora_architecture_profile
        peft_config = getattr(module, "peft_config", {}).get("default")
        if peft_config is None:
            raise RuntimeError(f"No default PEFT configuration found {stage}")

        rank = int(peft_config.r)
        alpha = int(peft_config.lora_alpha)
        if rank != int(self.model_config.lora_rank):
            raise RuntimeError(
                f"PEFT rank changed from {self.model_config.lora_rank} to {rank} {stage}"
            )
        if alpha != int(self.model_config.lora_alpha):
            raise RuntimeError(
                f"PEFT alpha changed from {self.model_config.lora_alpha} to {alpha} {stage}"
            )
        if float(peft_config.lora_dropout) != 0.0:
            raise RuntimeError(f"LoRA dropout must be zero, got {peft_config.lora_dropout}")
        if peft_config.bias != "none":
            raise RuntimeError(f"LoRA bias must be 'none', got {peft_config.bias!r}")
        if bool(peft_config.use_rslora):
            raise RuntimeError("RS-LoRA is not part of this controlled experiment")

        trainable = [
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError(f"No trainable LoRA parameters found {stage}")

        bad_trainable_dtypes = {
            name: str(parameter.dtype)
            for name, parameter in trainable
            if parameter.dtype != torch.float32
        }
        if bad_trainable_dtypes:
            raise RuntimeError(
                f"Trainable LoRA parameters are not FP32 {stage}: "
                f"{list(bad_trainable_dtypes.items())[:5]}"
            )

        forbidden = [
            name
            for name, _ in trainable
            if any(fragment in name for fragment in _FORBIDDEN_TRAINABLE_FRAGMENTS)
        ]
        if forbidden:
            raise RuntimeError(f"Forbidden modules received LoRA adapters: {forbidden[:5]}")

        parents = [_adapter_parent(name) for name, _ in trainable]
        parent_counts = Counter(parents)
        bad_parent_counts = {
            name: count for name, count in parent_counts.items() if count != 2
        }
        if bad_parent_counts:
            raise RuntimeError(
                "Every target module must contain exactly lora_A and lora_B; got "
                f"{list(bad_parent_counts.items())[:5]}"
            )

        target_kinds = Counter()
        target_layers = set()
        unmatched_targets = []
        for parent in parent_counts:
            match = profile.target_re.search(parent)
            if match is None:
                unmatched_targets.append(parent)
                continue
            target_layers.add(int(match.group("layer")))
            target_kinds[match.group("kind")] += 1

        if unmatched_targets:
            raise RuntimeError(
                "LoRA targeted modules outside the intended language projections: "
                f"{unmatched_targets[:5]}"
            )
        if target_layers != set(range(profile.language_layers)):
            raise RuntimeError(
                f"LoRA did not cover exactly {profile.display_name} language layers "
                f"0..{profile.language_layers - 1}; got "
                f"{sorted(target_layers)}"
            )
        if target_kinds != profile.target_kinds:
            raise RuntimeError(
                f"LoRA target distribution differs from the audited "
                f"{profile.display_name} architecture: "
                f"{target_kinds}"
            )
        if len(parent_counts) != profile.target_modules:
            raise RuntimeError(
                f"Expected {profile.target_modules} LoRA modules, got {len(parent_counts)}"
            )

        trainable_elements = sum(parameter.numel() for _, parameter in trainable)
        expected_elements = rank * profile.targets_per_rank
        if trainable_elements != expected_elements:
            raise RuntimeError(
                f"Expected {expected_elements:,} trainable LoRA elements at rank {rank}, "
                f"got {trainable_elements:,}"
            )

        frozen_float_dtypes = {
            parameter.dtype
            for parameter in module.parameters()
            if not parameter.requires_grad and parameter.is_floating_point()
        }
        if frozen_float_dtypes != {torch.bfloat16}:
            raise RuntimeError(
                f"Frozen {profile.display_name} parameters must remain BF16 {stage}; got "
                f"{sorted(map(str, frozen_float_dtypes))}"
            )

        if self.rank == 0:
            print(
                f"{profile.display_name} LoRA audit passed {stage}: "
                f"{len(parent_counts)} modules, "
                f"{len(trainable)} tensors, {trainable_elements:,} FP32 trainable "
                "elements; frozen base BF16."
            )


_existing_engine = EngineRegistry._engines["language_model"]["fsdp2"]["cuda"]
if _existing_engine not in (FSDPEngineWithLMHead, QwenFP32LoRAFSDPEngine):
    raise RuntimeError(
        "Refusing to replace an unexpected CUDA/FSDP2 language-model engine: "
        f"{_existing_engine}"
    )
EngineRegistry._engines["language_model"]["fsdp2"]["cuda"] = (
    QwenFP32LoRAFSDPEngine
)

if os.environ.get("VERL_RELEASE_VLLM_HOST_CACHE_AFTER_WAKE", "0") == "1":
    _install_vllm_host_cache_release()
