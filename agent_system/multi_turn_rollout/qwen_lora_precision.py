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
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import wraps

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from verl.utils.fs import copy_to_local
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

    def optimizer_step(self):
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

        return grad_norm

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
