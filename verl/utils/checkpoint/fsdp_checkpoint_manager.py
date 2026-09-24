# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import warnings
from collections import OrderedDict
from typing import Optional, Union

import torch
import torch.distributed
from accelerate import init_empty_weights
from torch.distributed.fsdp import FullStateDictConfig, ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_state_ctx

from .checkpoint_manager import BaseCheckpointManager


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents (list[str], optional):
            Components to include; must contain 'model', 'optimizer', 'extra'.
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: Optional[list] = None,
        **kwargs,
    ):
        if checkpoint_contents is None:
            checkpoint_contents = ["model", "optimizer", "extra"]
        if processing_class is None:
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            warnings.warn("`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2)
            processing_class = kwargs.pop("tokenizer")
        assert "model" in checkpoint_contents and "optimizer" in checkpoint_contents and "extra" in checkpoint_contents, f"FSDPCheckpointManager must include ['model', 'optimizer', 'extra'], got {checkpoint_contents}"

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_contents=checkpoint_contents,
        )

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: str = None,
        del_local_after_load=False,
        load_training_state=True,
    ):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        # every rank download its own checkpoint
        remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        remote_extra_state_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        print(f"[rank-{self.rank}]: Loading from {remote_model_path} and {remote_optim_path} and {remote_extra_state_path}")
        local_model_path = copy_to_local(remote_model_path)
        local_optim_path = copy_to_local(remote_optim_path) if load_training_state else None
        local_extra_state_path = copy_to_local(remote_extra_state_path) if load_training_state else None

        model_state_dict = torch.load(local_model_path, weights_only=False)
        optimizer_state_dict = torch.load(local_optim_path, weights_only=False) if load_training_state else None
        extra_state_dict = torch.load(local_extra_state_path, weights_only=False) if load_training_state else None

        if del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if local_optim_path and is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if local_extra_state_path and is_non_local(local_extra_state_path) else None
            except Exception as e:
                print(f"[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored")

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            self.model.load_state_dict(model_state_dict)
            if load_training_state and self.optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
        # Training state is intentionally skipped for evaluation so every
        # checkpoint uses the same evaluation RNG rather than its saved RNG.
        if load_training_state and "rng" in extra_state_dict:
            # 'rng' may not exist for backward compatibility
            self.load_rng_state(extra_state_dict["rng"])

        if load_training_state and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path
        if max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0 and len(self.previous_saved_paths) >= max_ckpt_to_keep:
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_state_dict = self.model.state_dict()
                optimizer_state_dict = self.optimizer.state_dict() if self.optimizer is not None else None
                lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None

                extra_state_dict = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "rng": self.get_rng_state(),
                }
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}")
                print(f"[rank-{self.rank}]: Saving optim to {os.path.abspath(optim_path)}")
                print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}")
                torch.save(model_state_dict, model_path)
                torch.save(optimizer_state_dict, optim_path)  # TODO: address optimizer is None
                torch.save(extra_state_dict, extra_path)

        if self.rank == 0:
            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            model_config = unwrap_model.config
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                # Some model's name_or_path is empty if not initialized from pretrained,
                # in this cases, we don't save generation config.
                generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                generation_config.save_pretrained(local_path)
            else:
                generation_config = None

            model_config.save_pretrained(local_path)
            self.processing_class.save_pretrained(local_path)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if "hf_model" in self.checkpoint_contents:
            hf_local_path = os.path.join(local_path, "huggingface")
            os.makedirs(hf_local_path, exist_ok=True)

            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with get_fsdp_state_ctx(self.model, StateDictType.FULL_STATE_DICT, state_dict_config, None):
                state_dict = self.model.state_dict()

            if self.rank == 0:
                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    from transformers import AutoModelForVision2Seq

                    auto_model_cls = AutoModelForVision2Seq
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(model_config, torch_dtype=torch.bfloat16)
                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found in, using a generation config created from the model config when saving hf_model.")

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                self.processing_class.save_pretrained(hf_local_path)
                del state_dict
                del save_model

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        self.previous_saved_paths.append(local_path)


class FSDPLoraCheckpointManager(BaseCheckpointManager):
    """Save resumable FSDP LoRA checkpoints without copying the frozen model.

    The base model is reconstructed from ``base_model_path`` when a worker is
    created.  A checkpoint therefore only needs the trainable FSDP parameters,
    optimizer state, scheduler state, and RNG state.  Parameter names and
    shapes are checked strictly on load so this format cannot silently restore
    into a different FSDP/LoRA layout.

    This is an internal training checkpoint format.  Each rank writes its own
    trainable parameter shard and optimizer state, so the same world size and
    model configuration must be used when resuming.
    """

    format_version = 1

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_metadata: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_contents=["lora_model", "optimizer", "extra"],
        )
        self.checkpoint_metadata = checkpoint_metadata or {}

    def _model_path(self, local_path: str) -> str:
        return os.path.join(local_path, f"lora_model_world_size_{self.world_size}_rank_{self.rank}.pt")

    def _optim_path(self, local_path: str) -> str:
        return os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")

    def _extra_path(self, local_path: str) -> str:
        return os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

    def _trainable_parameters(self) -> OrderedDict:
        parameters = OrderedDict(
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )
        # FSDP1 with NO_SHARD exposes both each authoritative FlatParameter and
        # a same-storage unsharded parameter view after CPU offload. Saving both
        # doubles a LoRA checkpoint. Keep the flat tensors when they are present;
        # copying them also updates their views on restore.
        flat_parameters = OrderedDict(
            (name, parameter)
            for name, parameter in parameters.items()
            if name.endswith("._flat_param")
        )
        return flat_parameters or parameters

    def _load_legacy_model_state(self, local_path: str) -> None:
        """Load LoRA tensors from a pre-existing full FSDP checkpoint."""
        legacy_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        if not os.path.exists(legacy_path):
            raise FileNotFoundError(
                f"No LoRA-only or legacy model checkpoint found in {local_path}. "
                f"Expected {os.path.basename(self._model_path(local_path))}."
            )

        legacy_state = torch.load(legacy_path, map_location="cpu", mmap=True, weights_only=False)
        lora_state = OrderedDict((name, value) for name, value in legacy_state.items() if "lora_" in name)
        if not lora_state:
            raise RuntimeError(f"Legacy checkpoint {legacy_path} contains no LoRA parameters")

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            incompatible = self.model.load_state_dict(lora_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected LoRA keys while loading {legacy_path}: {incompatible.unexpected_keys}")
        print(f"[rank-{self.rank}]: Loaded {len(lora_state)} LoRA tensors from legacy full checkpoint {legacy_path}")

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: str = None,
        del_local_after_load=False,
        load_training_state=True,
    ):
        if local_path is None:
            return
        if hdfs_path is not None:
            raise NotImplementedError("HDFS LoRA-only checkpoint loading is not implemented")

        model_path = self._model_path(local_path)
        optim_path = self._optim_path(local_path)
        extra_path = self._extra_path(local_path)
        print(f"[rank-{self.rank}]: Loading LoRA checkpoint from {local_path}")

        if os.path.exists(model_path):
            payload = torch.load(model_path, map_location="cpu", weights_only=False)
            if payload.get("format_version") != self.format_version:
                raise RuntimeError(
                    f"Unsupported LoRA checkpoint format {payload.get('format_version')} in {model_path}"
                )
            saved_metadata = payload.get("metadata", {})
            for key in (
                "base_model_path",
                "model_type",
                "model_commit",
                "lora_rank",
                "lora_alpha",
                "target_modules",
            ):
                if saved_metadata.get(key) != self.checkpoint_metadata.get(key):
                    raise RuntimeError(
                        f"LoRA checkpoint {key} mismatch: checkpoint={saved_metadata.get(key)!r}, "
                        f"current={self.checkpoint_metadata.get(key)!r}"
                    )
            saved_parameters = payload["trainable_parameters"]
            # Version-1 checkpoints produced before duplicate FSDP views were
            # removed contain both FlatParameters and their weight views.
            saved_flat_parameters = OrderedDict(
                (name, parameter)
                for name, parameter in saved_parameters.items()
                if name.endswith("._flat_param")
            )
            if saved_flat_parameters:
                saved_parameters = saved_flat_parameters
            current_parameters = self._trainable_parameters()
            if set(saved_parameters) != set(current_parameters):
                missing = sorted(set(current_parameters) - set(saved_parameters))
                unexpected = sorted(set(saved_parameters) - set(current_parameters))
                raise RuntimeError(
                    "LoRA checkpoint parameter layout differs from the current model: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            with torch.no_grad():
                for name, parameter in current_parameters.items():
                    saved = saved_parameters[name]
                    if tuple(saved.shape) != tuple(parameter.shape):
                        raise RuntimeError(
                            f"Shape mismatch for {name}: checkpoint={tuple(saved.shape)}, "
                            f"current={tuple(parameter.shape)}"
                        )
                    parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))
            print(f"[rank-{self.rank}]: Restored {len(saved_parameters)} trainable LoRA/FSDP parameters")
        else:
            self._load_legacy_model_state(local_path)

        if load_training_state:
            optimizer_state_dict = torch.load(optim_path, map_location="cpu", weights_only=False)
            extra_state_dict = torch.load(extra_path, map_location="cpu", weights_only=False)
            if self.optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])
            if "rng" in extra_state_dict:
                self.load_rng_state(extra_state_dict["rng"])

        if del_local_after_load:
            state_paths = (model_path, optim_path, extra_path) if load_training_state else (model_path,)
            for path in state_paths:
                if os.path.exists(path) and is_non_local(path):
                    os.remove(path)

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        if local_path is None:
            return
        if hdfs_path is not None:
            raise NotImplementedError("HDFS LoRA-only checkpoint saving is not implemented")

        self.previous_global_step = global_step
        if max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0 and len(self.previous_saved_paths) >= max_ckpt_to_keep:
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        trainable_parameters = OrderedDict(
            (name, parameter.detach().to(device="cpu", copy=True))
            for name, parameter in self._trainable_parameters().items()
        )
        if not trainable_parameters:
            raise RuntimeError("LoRA-only checkpoint requested, but the FSDP model has no trainable parameters")

        model_payload = {
            "format_version": self.format_version,
            "world_size": self.world_size,
            "rank": self.rank,
            "metadata": self.checkpoint_metadata,
            "trainable_parameters": trainable_parameters,
        }
        optimizer_state_dict = self.optimizer.state_dict() if self.optimizer is not None else None
        extra_state_dict = {
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "rng": self.get_rng_state(),
        }

        model_path = self._model_path(local_path)
        optim_path = self._optim_path(local_path)
        extra_path = self._extra_path(local_path)
        print(f"[rank-{self.rank}]: Saving {len(trainable_parameters)} trainable parameters to {model_path}")
        torch.save(model_payload, model_path)
        torch.save(optimizer_state_dict, optim_path)
        torch.save(extra_state_dict, extra_path)

        if self.rank == 0:
            manifest = {
                "checkpoint_type": "fsdp_lora_only",
                "format_version": self.format_version,
                "world_size": self.world_size,
                **self.checkpoint_metadata,
            }
            with open(os.path.join(local_path, "lora_checkpoint.json"), "w", encoding="utf-8") as stream:
                json.dump(manifest, stream, ensure_ascii=False, indent=2)

        torch.distributed.barrier()
        self.previous_saved_paths.append(local_path)
