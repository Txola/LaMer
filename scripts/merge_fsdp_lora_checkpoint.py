#!/usr/bin/env python3
"""Export a single-GPU VERL FSDP LoRA checkpoint as a merged HF model."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


MARKER_NAME = ".lora_merge_complete.json"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a single-GPU, adapter-only VERL FSDP checkpoint into its "
            "frozen Hugging Face base model."
        )
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="VERL global_step_N directory or its actor subdirectory",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        help=(
            "Frozen Hugging Face base model. By default, read MODEL_PATH from "
            "the training run's training_parameters.txt."
        ),
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        help="Output directory (default: global_step_N/hf_model_merged)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing target only after a new merge succeeds",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom Transformers code from the local base-model directory",
    )
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> tuple[Path, Path, Path]:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"Checkpoint directory does not exist: {path}")
    if (path / "actor").is_dir():
        checkpoint_root = path
        actor = path / "actor"
    else:
        actor = path
        checkpoint_root = actor.parent
    if not (actor / "fsdp_config.json").is_file():
        raise SystemExit(f"Missing FSDP metadata: {actor / 'fsdp_config.json'}")
    return path, checkpoint_root, actor


def infer_base_model(actor: Path) -> Path:
    candidates = [
        actor.parents[2] / "training_parameters.txt",
        actor.parents[1] / "training_parameters.txt",
    ]
    for parameters in candidates:
        if not parameters.is_file():
            continue
        for line in parameters.read_text(encoding="utf-8").splitlines():
            if line.startswith("MODEL_PATH="):
                value = line.removeprefix("MODEL_PATH=").strip()
                if value:
                    base = Path(value).expanduser().resolve()
                    if base.is_dir():
                        log(f"Inferred base model from {parameters}: {base}")
                        return base
                    raise SystemExit(
                        f"MODEL_PATH from {parameters} does not exist: {base}"
                    )
    raise SystemExit(
        "Could not infer the frozen base model. Pass --base-model explicitly."
    )


def source_model_path(actor: Path, world_size: int) -> Path:
    ordinary = actor / f"model_world_size_{world_size}_rank_0.pt"
    lora_only = actor / f"lora_model_world_size_{world_size}_rank_0.pt"
    if ordinary.is_file() and lora_only.is_file():
        raise SystemExit(
            f"Ambiguous checkpoint: both {ordinary.name} and {lora_only.name} exist"
        )
    if ordinary.is_file():
        return ordinary
    if lora_only.is_file():
        return lora_only
    raise SystemExit(
        "No adapter checkpoint found; expected "
        f"{ordinary.name} or {lora_only.name} under {actor}"
    )


def load_adapter_state(actor: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    import torch

    fsdp = json.loads((actor / "fsdp_config.json").read_text(encoding="utf-8"))
    world_size = int(fsdp.get("world_size", 0))
    if world_size != 1:
        raise SystemExit(
            "This exporter currently supports single-GPU FSDP checkpoints only; "
            f"the checkpoint declares world_size={world_size}."
        )

    model_path = source_model_path(actor, world_size)
    raw = torch.load(
        model_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    payload_metadata: dict[str, Any] = {}
    if model_path.name.startswith("lora_model_"):
        if not isinstance(raw, dict) or "trainable_parameters" not in raw:
            raise SystemExit(f"Invalid LoRA-only checkpoint payload: {model_path}")
        payload_metadata = raw.get("metadata", {})
        raw = raw["trainable_parameters"]
    if not isinstance(raw, dict) or not raw:
        raise SystemExit(f"Checkpoint contains no adapter state: {model_path}")

    adapter_state: dict[str, Any] = {}
    for name, value in raw.items():
        if "lora_A" not in name and "lora_B" not in name:
            raise SystemExit(
                "Checkpoint is not adapter-only; found non-LoRA key "
                f"{name!r}. Use scripts/merge_fsdp_checkpoint.sh for a full checkpoint."
            )
        if name.endswith("._flat_param"):
            raise SystemExit(
                "This checkpoint stores flattened LoRA parameters without module names "
                "and cannot be exported safely. Resume it with the original trainer and "
                "save a named adapter checkpoint."
            )
        tensor = value._local_tensor if hasattr(value, "_local_tensor") else value
        if not isinstance(tensor, torch.Tensor):
            raise SystemExit(f"Adapter entry {name!r} is not a tensor")
        normalized = name.replace(".default.weight", ".weight")
        adapter_state[normalized] = tensor.detach().cpu().contiguous()

    meta_path = actor / "lora_train_meta.json"
    metadata = (
        json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.is_file()
        else payload_metadata
    )
    try:
        rank = int(metadata.get("r", metadata.get("lora_rank")))
        alpha = int(metadata.get("lora_alpha"))
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"Missing or invalid LoRA rank/alpha metadata under {actor}"
        ) from exc
    if rank <= 0 or alpha <= 0:
        raise SystemExit(f"Invalid LoRA metadata: rank={rank}, alpha={alpha}")
    metadata = {**metadata, "r": rank, "lora_alpha": alpha}
    return adapter_state, metadata, model_path


def exact_target_modules(adapter_state: dict[str, Any]) -> list[str]:
    targets = set()
    prefix = "base_model.model."
    for name in adapter_state:
        marker = ".lora_A." if ".lora_A." in name else ".lora_B."
        parent = name.split(marker, 1)[0]
        if parent.startswith(prefix):
            parent = parent[len(prefix) :]
        targets.add(parent)
    if not targets:
        raise SystemExit("Could not derive any adapted module paths")
    return sorted(targets)


def auto_model_class(config: Any) -> Any:
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    if type(config) in AutoModelForImageTextToText._model_mapping:
        return AutoModelForImageTextToText
    if type(config) in AutoModelForCausalLM._model_mapping:
        return AutoModelForCausalLM
    architecture = (getattr(config, "architectures", None) or [""])[0]
    if "ForConditionalGeneration" in architecture:
        return AutoModelForImageTextToText
    if "ForCausalLM" in architecture:
        return AutoModelForCausalLM
    raise SystemExit(
        f"Unsupported Transformers architecture: {architecture or type(config).__name__}"
    )


def complete_hf_model(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    if any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        return True
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if not index_path.is_file():
            continue
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        except (KeyError, OSError, json.JSONDecodeError):
            return False
        shards = set(weight_map.values())
        return bool(shards) and all((path / shard).is_file() for shard in shards)
    return False


def marker_matches(target: Path, actor: Path, base: Path, model_path: Path) -> bool:
    try:
        marker = json.loads((target / MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    stat = model_path.stat()
    return (
        marker.get("format") == "lamer_fsdp_lora_merged_hf_v2"
        and Path(marker.get("source_actor", "")).resolve() == actor
        and Path(marker.get("base_model", "")).resolve() == base
        and marker.get("source_size") == stat.st_size
        and marker.get("source_mtime_ns") == stat.st_mtime_ns
    )


def validate_safe_target(target: Path, protected: list[Path]) -> None:
    if target.name in {"", ".", ".."}:
        raise SystemExit(f"Unsafe target directory: {target}")
    resolved = target.resolve()
    forbidden = {Path("/"), Path.home().resolve(), *(path.resolve() for path in protected)}
    if resolved in forbidden:
        raise SystemExit(f"Refusing unsafe target directory: {resolved}")


def base_weight_bytes(base: Path) -> int:
    weights = list(base.glob("*.safetensors")) + list(base.glob("pytorch_model*.bin"))
    return sum(path.stat().st_size for path in weights)


def is_hf_weight_artifact(path: Path) -> bool:
    """Return whether a top-level HF repository entry stores model weights."""
    name = path.name
    return (
        name.endswith(".safetensors")
        or name.endswith(".safetensors.index.json")
        or name.startswith("pytorch_model")
        or name.startswith("tf_model")
        or name.startswith("flax_model")
        or name.startswith("adapter_model")
    )


def copy_base_metadata(base: Path, destination: Path) -> None:
    """Preserve the base model's non-weight files without version translation.

    LoRA merging changes model weights only. Loading and saving the tokenizer or
    processor with the merger's newer Transformers version can rewrite otherwise
    compatible metadata into a format that the evaluation stack cannot read.
    """
    copied = 0
    for source in base.iterdir():
        if is_hf_weight_artifact(source):
            continue
        target = destination / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        elif source.is_file():
            shutil.copy2(source, target, follow_symlinks=True)
        else:
            continue
        copied += 1
    if copied == 0:
        raise RuntimeError(f"No Hugging Face metadata found under base model: {base}")
    log(f"Copied {copied} base-model metadata entries without re-serialization.")


def merge(
    actor: Path,
    base: Path,
    destination: Path,
    adapter_state: dict[str, Any],
    metadata: dict[str, Any],
    model_path: Path,
    trust_remote_code: bool,
) -> None:
    import torch
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from transformers import AutoConfig

    actor_config_path = actor / "huggingface" / "config.json"
    actor_config = (
        json.loads(actor_config_path.read_text(encoding="utf-8"))
        if actor_config_path.is_file()
        else {}
    )
    config = AutoConfig.from_pretrained(base, trust_remote_code=trust_remote_code)
    actor_model_type = actor_config.get("model_type")
    if actor_model_type and actor_model_type != config.model_type:
        raise SystemExit(
            "Base/checkpoint model type mismatch: "
            f"checkpoint={actor_model_type}, base={config.model_type}"
        )

    targets = exact_target_modules(adapter_state)
    model_class = auto_model_class(config)
    log(
        f"Loading {config.model_type} base model in BF16 with "
        f"{len(targets)} exact LoRA target modules..."
    )
    model = model_class.from_pretrained(
        base,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=metadata["r"],
            lora_alpha=metadata["lora_alpha"],
            target_modules=targets,
            lora_dropout=0.0,
            bias="none",
            use_rslora=False,
        ),
        autocast_adapter_dtype=True,
    )

    load_result = set_peft_model_state_dict(model, adapter_state)
    if load_result.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys: {load_result.unexpected_keys}")
    loaded = get_peft_model_state_dict(model)
    if set(loaded) != set(adapter_state):
        missing = sorted(set(adapter_state) - set(loaded))
        unexpected = sorted(set(loaded) - set(adapter_state))
        raise RuntimeError(
            f"Adapter key mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for name, expected in adapter_state.items():
        torch.testing.assert_close(loaded[name].cpu(), expected, rtol=0, atol=0)
    log(f"Validated all {len(adapter_state)} saved LoRA tensors exactly.")

    log("Merging the FP32 LoRA delta into BF16 base weights...")
    model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(
        destination,
        safe_serialization=True,
        max_shard_size="4GB",
    )

    # Preserve config, tokenizer, processor, chat-template, and remote-code files
    # exactly as published by the base model. The adapter does not modify them.
    copy_base_metadata(base, destination)

    source_stat = model_path.stat()
    marker = {
        "format": "lamer_fsdp_lora_merged_hf_v2",
        "source_actor": str(actor),
        "source_checkpoint_file": str(model_path),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "base_model": str(base),
        "model_type": config.model_type,
        "lora_rank": metadata["r"],
        "lora_alpha": metadata["lora_alpha"],
        "adapter_tensor_count": len(adapter_state),
        "target_module_count": len(targets),
        "merge_dtype": "bfloat16",
    }
    (destination / MARKER_NAME).write_text(
        json.dumps(marker, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    _, checkpoint_root, actor = resolve_checkpoint(args.checkpoint)
    base = (
        args.base_model.expanduser().resolve()
        if args.base_model is not None
        else infer_base_model(actor)
    )
    if not base.is_dir() or not (base / "config.json").is_file():
        raise SystemExit(f"Invalid Hugging Face base model directory: {base}")

    target = (
        args.target_dir.expanduser().resolve()
        if args.target_dir is not None
        else (checkpoint_root / "hf_model_merged").resolve()
    )
    validate_safe_target(target, [actor, checkpoint_root, base])

    adapter_state, metadata, model_path = load_adapter_state(actor)
    if target.exists() and not args.force:
        if (
            target.is_dir()
            and complete_hf_model(target)
            and marker_matches(target, actor, base, model_path)
        ):
            log(f"Reusing completed LoRA merge: {target}")
            print(target)
            return
        raise SystemExit(
            f"Target exists but is incomplete or from another source: {target}. "
            "Pass --force to replace it after a successful new merge."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    required = base_weight_bytes(base) + 1024**3
    available = shutil.disk_usage(target.parent).free
    if available < required:
        raise SystemExit(
            f"Insufficient free disk: need about {required / 1024**3:.1f} GiB, "
            f"have {available / 1024**3:.1f} GiB"
        )

    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.merge-tmp.", dir=target.parent))
    try:
        # Keep stdout reserved for the final path so callers can use
        # MODEL_PATH=$(scripts/merge_fsdp_lora_checkpoint.sh ...).
        with contextlib.redirect_stdout(sys.stderr):
            merge(
                actor,
                base,
                temp,
                adapter_state,
                metadata,
                model_path,
                args.trust_remote_code,
            )
        if not complete_hf_model(temp):
            raise RuntimeError(f"Merge produced an incomplete Hugging Face model: {temp}")
        if target.exists():
            shutil.rmtree(target)
        temp.rename(target)
    finally:
        if temp.exists():
            shutil.rmtree(temp)

    log(f"LoRA merge complete: {target}")
    print(target)


if __name__ == "__main__":
    main()
