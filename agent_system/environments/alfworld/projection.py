from typing import List
from dataclasses import dataclass
import re
import copy


REFLECTION_MEMORY_MAX_CHARACTERS = 2000


def normalize_reflection_memory(text: str) -> str:
    """Apply the existing reflection-memory whitespace and size normalization."""
    return text.strip()[:REFLECTION_MEMORY_MAX_CHARACTERS]


@dataclass(frozen=True, slots=True)
class TruncatedReflectionRecovery:
    """Separately expose full extracted text and installed memory override."""

    extracted_text: str
    memory_override: str


def recover_truncated_reflection_memory(
    response: str,
    *,
    phase: str,
    hard_truncated: bool,
    enabled: bool,
) -> TruncatedReflectionRecovery | None:
    """Recover an unclosed reflection suffix without changing parser validity."""
    if phase not in {'play', 'reflect'}:
        raise ValueError(f"Unsupported ALFWorld generation phase: {phase!r}")
    if not enabled or phase != 'reflect' or not hard_truncated:
        return None

    start_tag = '<remark>'
    end_tag = '</remark>'
    start_idx = response.rfind(start_tag)
    if start_idx == -1:
        return None
    content_start = start_idx + len(start_tag)
    if response.find(end_tag, content_start) != -1:
        return None

    extracted_text = response[content_start:].strip()
    if not extracted_text:
        return None
    return TruncatedReflectionRecovery(
        extracted_text=extracted_text,
        memory_override=normalize_reflection_memory(extracted_text),
    )

def alfworld_projection(actions: List[str], action_pools: List[List[str]]=None, phase='play'):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """
    assert phase in ['play', 'reflect']
    if phase == 'play':
        valids = [0] * len(actions)
        actions = copy.deepcopy(actions)
        for i in range(len(actions)):
            original_str = actions[i]  # keep the original string
            actions[i] = actions[i].lower()

            # Attempt to extract the substring within <action>...</action>
            start_tag = "<action>"
            end_tag = "</action>"
            start_idx = actions[i].find(start_tag)
            end_idx = actions[i].find(end_tag)
            try:
                if start_idx == -1 or end_idx == -1:
                    # If we can't find a valid <action>...</action> block, mark as invalid
                    actions[i] = actions[i][-30:]  # 0 is invalid action for Sokoban
                    continue

                # Extract just the content between the tags
                extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
                
                actions[i] = extracted_action
                valids[i] = 1

            except:
                actions[i] = actions[i][-30:]

        return actions, valids
    else:
        valids = [0] * len(actions)
        reflections = [''] * len(actions)

        for i in range(len(actions)):
            action = actions[i]
            start_tag = "<remark>"
            start_idx = action.rfind(start_tag)
            end_tag = "</remark>"
            end_idx = action.rfind(end_tag)
            if start_idx == -1 or end_idx == -1:
                reflections[i] = ''
            else:
                reflections[i] = normalize_reflection_memory(
                    action[start_idx + len(start_tag):end_idx]
                )
                valids[i] = 1

        return reflections, valids


if __name__ == '__main__':
    print(alfworld_projection(['<reflection>gogogo</reflection>'], phase='reflect'))
