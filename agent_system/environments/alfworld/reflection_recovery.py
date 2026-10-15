"""Rollout-side orchestration for truncated ALFWorld reflection recovery."""

from __future__ import annotations

from typing import Any

from .projection import recover_truncated_reflection_memory


def default_reflection_recovery_metadata() -> dict[str, bool | str | None]:
    """Metadata present on every play and reflection generation record."""
    return {
        "reflection_recovered_from_truncation": False,
        "reflection_memory_override": None,
    }


def process_reflection_generation(
    manager: Any,
    reflection_output: Any,
    reflection_text: str,
    *,
    recovery_enabled: bool,
) -> list[dict[str, Any]]:
    """Parse a reflection and optionally override only its stored memory."""
    fields = reflection_output.extra_fields
    recovery = recover_truncated_reflection_memory(
        reflection_text,
        phase=str(fields["phase"]),
        hard_truncated=bool(fields["response_token_cap_hit"]),
        enabled=recovery_enabled,
    )
    memory_override = recovery.memory_override if recovery is not None else None
    if memory_override is None:
        _, _, _, reflection_infos = manager.step(
            [reflection_text],
            phase="reflect",
        )
    else:
        _, _, _, reflection_infos = manager.step(
            [reflection_text],
            phase="reflect",
            reflection_memory_overrides=[memory_override],
        )
    recovered = bool(
        reflection_infos[0].get(
            "reflection_recovered_from_truncation",
            False,
        )
    )
    fields["reflection_recovered_from_truncation"] = recovered
    fields["reflection_memory_override"] = reflection_infos[0].get(
        "reflection_memory_override"
    )
    return reflection_infos
