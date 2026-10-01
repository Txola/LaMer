ALFWORLD_PLAY_PROMPT = """
You are an expert agent operating in the ALFRED Embodied Environment.
{init_observation}{past_trajectories_reflections}{additional_context}{current_trajectory}

Your admissible actions of the current situation are: 
[{admissible_actions}]

Now it's your turn to take an action.

- Your response should first by step-by-step reasoning about the current situation.
- Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


ALFWORLD_REFLECT_PROMPT = """
You are an expert agent operating in the ALFRED Embodied Environment. 
{init_observation}

You will be given the history of a past experience.
Your job is to **reflect on the past sequence**, identify any **mistakes or inefficiencies**, and then devise a **concise, improved plan** starting from the original initial state.

Below are the actions you took and the corresponding observations:
{current_trajectory}{additional_context}
The task is NOT successfully completed.

Now it's your turn to reflect on the past experience and come up with a new plan of action.

- Your response should first be step-by-step reasoning about the strategy and path you took to attempt to complete the task. Identify where things went wrong or could be better.
- Then devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken.
- Finally, end the response with your reflection and improved plan inside <remark> </remark> tags, to guide the next trial.
"""

# Prompt templates for parsing past trajectories and reflections
PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE = '''

On trial #{traj_idx}, the actions you took and the corresponding observations are: 
{past_trajectory}
The task is NOT successfully completed. Your reflection is: 
{reflection}'''

HISTORY_ONLY_TEMPLATE = '''

On trial #{traj_idx}, the actions you took and the corresponding observations are: 
{past_trajectory}
The task is NOT successfully completed.
'''

REFLECTION_ONLY_TEMPLATE = '''

On trial #{traj_idx}, the task is NOT successfully completed. Your reflection is:
{reflection}'''


FAILED_PEER_REFLECTION_INSTRUCTION = (
    "These other attempts did not solve the task. Their reflections may contain "
    "useful observations, mistakes, or uncertain conclusions. Treat them as "
    "additional evidence rather than guaranteed facts."
)


def format_failed_peer_reflections(reflections):
    """Format failed-peer reflections without changing or merging their text."""
    if not reflections:
        return ""
    blocks = [f"\n\n{FAILED_PEER_REFLECTION_INSTRUCTION}"]
    for reflection in reflections:
        blocks.append(
            "\n\n[REFLECTION FROM ANOTHER UNSUCCESSFUL ATTEMPT "
            "ON THE SAME TASK]\n" + reflection
        )
    return "".join(blocks)


def format_unrelated_failed_reflections(reflections):
    """Format the negative-control context from failed unrelated tasks."""
    if not reflections:
        return ""
    blocks = [
        "\n\nThese reflections come from unsuccessful attempts on other tasks. "
        "They are an unrelated-information control and may not apply here."
    ]
    for reflection in reflections:
        blocks.append(
            "\n\n[REFLECTION FROM AN UNSUCCESSFUL ATTEMPT ON AN UNRELATED TASK]\n"
            + reflection
        )
    return "".join(blocks)


def format_failed_peer_histories(histories):
    """Format full failed same-task histories for one assisted reflection."""
    if not histories:
        return ""
    blocks = [
        "\n\nThe following are complete histories from other unsuccessful "
        "attempts on the exact same task. Use them together with the recipient "
        "attempt above as evidence when diagnosing mistakes and devising one "
        "improved plan."
    ]
    for index, history in enumerate(histories, start=1):
        blocks.append(
            f"\n\n[FAILED SAME-TASK PEER ATTEMPT {index}]\n{history}"
        )
    return "".join(blocks)


def format_reflection_merge_prompt(task, own_reflection, peer_reflections, peer_kind):
    """Build one consolidation prompt from an own reflection and peer reflections."""
    if peer_kind not in {"same_task", "unrelated"}:
        raise ValueError(f"Unsupported peer kind: {peer_kind}")
    if peer_kind == "same_task":
        source_description = (
            "unsuccessful attempts on the exact same task. They may contain useful "
            "observations, mistakes, or uncertain conclusions."
        )
        block_label = "FAILED SAME-TASK PEER REFLECTION"
    else:
        source_description = (
            "unsuccessful attempts on other tasks. They are an unrelated-information "
            "control and may not apply to the recipient's task."
        )
        block_label = "FAILED UNRELATED-TASK PEER REFLECTION"

    blocks = [
        "You are an expert agent operating in the ALFRED Embodied Environment.\n\n"
        f"Task: {task}\n\n"
        "Consolidate the recipient's normal reflection and the peer reflections "
        "below into one concise reflection and improved plan for the recipient. "
        "Do not merely concatenate the inputs, and do not treat failed-peer "
        "conclusions as guaranteed facts.\n\n"
        "[RECIPIENT'S OWN REFLECTION]\n"
        f"{own_reflection}\n\n"
        f"The peer reflections come from {source_description}"
    ]
    for index, reflection in enumerate(peer_reflections, start=1):
        blocks.append(
            f"\n\n[{block_label} {index}]\n{reflection}"
        )
    blocks.append(
        "\n\nProduce one consolidated reflection and improved plan. End the response "
        "with only the consolidated reflection and plan inside <remark> </remark> "
        "tags so it can guide the recipient's retry."
    )
    return "".join(blocks)

def parse_reflection(traj_idx, past_traj, reflection, reflection_type='reflection_only'):
    if traj_idx == 0 or len(reflection) == 0:
        return '\n'
    else:
        memories = []
        for _idx in range(traj_idx):
            if reflection_type == 'history_and_reflection':
                memory = PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                    reflection=reflection[_idx]
                )
            elif reflection_type == 'history_only':
                memory = HISTORY_ONLY_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                )
            elif reflection_type == 'reflection_only':
                memory = REFLECTION_ONLY_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    reflection=reflection[_idx]
                )
            memories.append(memory)
        return ''.join(memories)


CURR_TRAJ_AT_TRAJ1 = '''
Below are the actions you took and the corresponding observations: 
{current_trajectory}'''

CURR_TRAJ_AT_TRAJ2toN = '''

Currently you're on trial #{traj_idx}, below are the actions you took and the corresponding observations: 
{current_trajectory}'''

TRAJ_2toN_INIT = '''

Currently you're on trial #{traj_idx}, starting from the initial state.'''


def parse_current_trajectory(turn_idx, traj_idx, curr_traj):
    if traj_idx == 0:
        if turn_idx == 0:
            return ""
        else:
            return CURR_TRAJ_AT_TRAJ1.format(
                current_trajectory=curr_traj
            )
    else:
        if turn_idx == 0:
            return TRAJ_2toN_INIT.format(traj_idx=traj_idx + 1)
        else:
            return CURR_TRAJ_AT_TRAJ2toN.format(
                traj_idx=traj_idx + 1,
                current_trajectory=curr_traj
            )
        
def get_alfworld_prompt(phase: str = 'play',
                        turn_idx: int = 0,
                        traj_idx: int = 0,
                        init_observation: str = '',
                        curr_traj: str='',
                        past_traj: dict={},
                        admissible_actions: str='',
                        reflection: str='',
                        reflection_type: str='reflection_only',
                        additional_context: str='',
                        ):
    assert phase in ['play', 'reflect']
    if phase == 'play':
        past_trajectories_reflections = parse_reflection(traj_idx, past_traj, reflection, reflection_type)
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)

        prompt = ALFWORLD_PLAY_PROMPT.format(
            init_observation=init_observation,
            past_trajectories_reflections=past_trajectories_reflections,
            additional_context=additional_context,
            current_trajectory=current_trajectory,
            admissible_actions=admissible_actions,
        )

    else:
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = ALFWORLD_REFLECT_PROMPT.format(
            init_observation=init_observation,
            current_trajectory=current_trajectory,
            additional_context=additional_context,
        )

    return prompt.strip()
