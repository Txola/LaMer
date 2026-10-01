"""Modern VERL agent loop for LaMer's text ALFWorld protocol."""

from __future__ import annotations

import json
import os
from functools import lru_cache, partial
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from agent_system.environments.alfworld.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.alfworld.envs import (
    LocalAlfworldEnv,
    get_environment,
    load_config_file,
)
from agent_system.environments.alfworld.projection import alfworld_projection
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput


@lru_cache(maxsize=None)
def _ordered_evaluation_games(
    alf_config_path: str,
    eval_dataset: str,
    seed: int,
) -> tuple[str, ...]:
    """Return the exact seeded game order used by the legacy evaluator."""
    config = load_config_file(alf_config_path)
    env_type = config["env"]["type"]
    base_env = get_environment(env_type)(config, train_eval=eval_dataset)
    permutation = np.random.RandomState(seed).permutation(len(base_env.game_files))
    return tuple(base_env.game_files[index] for index in permutation)


def _task_type(gamefile: str) -> str:
    trajectory_path = gamefile.replace("game.tw-pddl", "traj_data.json")
    with open(trajectory_path, encoding="utf-8") as stream:
        return json.load(stream)["task_type"]


class AlfWorldAgentLoop(AgentLoopBase):
    """Run LaMer's per-action prompting protocol for validation.

    Every play turn and reflection is generated from a fresh, independently
    templated user prompt, matching LaMer's original rollout semantics.  The
    pinned v1 VERL worker accepts one output per agent-loop session, so validation
    returns the final model call with whole-episode outcome metadata.  Training
    remains disabled until the worker and GiGPO credit assignment are extended
    to retain every model call as its own policy sample.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length

    def _protocol(self, split: str) -> dict[str, Any]:
        env_config = self.config.env
        is_validation = split != "train"
        return {
            "is_validation": is_validation,
            "env_seed": int(
                env_config.get("val_seed", env_config.seed + 1000)
                if is_validation
                else env_config.seed
            ),
            "num_attempts": int(
                env_config.get("val_num_attempts", env_config.num_attempts)
                if is_validation
                else env_config.num_attempts
            ),
            "do_reflection": bool(
                env_config.get("val_do_reflection", env_config.do_reflection)
                if is_validation
                else env_config.do_reflection
            ),
            "max_turns": int(env_config.max_turns),
            "reflection_type": str(env_config.get("reflection_type", "reflection_only")),
            "eval_dataset": str(env_config.alfworld.get("eval_dataset", "eval_all")),
        }

    async def _generate_one(
        self,
        prompt: str,
        sampling_params: dict[str, Any],
        request_id: str,
        priority: int,
        phase: str,
        attempt_index: int,
        turn_index: int,
    ) -> tuple[AgentLoopOutput, str]:
        messages = [{"role": "user", "content": prompt}]
        # ALFWorld is text-only.  Build its prompt synchronously because this
        # machine does not reliably wake a reused asyncio executor thread after
        # its first job.  The same upstream Qwen3.5 processor-backed builder is
        # still used, so this changes scheduling rather than prompt tokens.
        # Tokenization takes milliseconds and is negligible beside generation.
        prompt_ids = self.continuous_token_builder.build_initial_tokens(messages)
        prompt_ids = self._cap_text_prompt_length(prompt_ids)

        metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", metrics):
            generated: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                priority=priority,
            )
        metrics["num_preempted"] = (
            generated.num_preempted if generated.num_preempted is not None else -1
        )

        merge_result = self.continuous_token_builder.merge_assistant_tokens(
            prompt_ids, generated.token_ids
        )
        response_mask, response_logprobs = (
            self.continuous_token_builder.align_response_metadata(
                merge_result,
                [],
                [] if generated.log_probs else None,
                assistant_logprobs=(
                    generated.log_probs if generated.log_probs else None
                ),
            )
        )
        response_ids = merge_result.token_ids[-len(response_mask) :] if response_mask else []
        aligned_prompt_ids = (
            merge_result.token_ids[: len(merge_result.token_ids) - len(response_mask)]
            if response_mask
            else merge_result.token_ids
        )
        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]
        if response_logprobs:
            response_logprobs = response_logprobs[: self.response_length]

        text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        output = AgentLoopOutput(
            prompt_ids=aligned_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            num_turns=2,
            metrics=metrics,
            extra_fields={
                "phase": phase,
                "attempt_index": attempt_index,
                "turn_index": turn_index,
                "prompt_text": prompt,
                "response_text": text,
                "turn_scores": [],
                "tool_rewards": [],
            },
        )
        return output, text

    @rollout_trace_op
    async def run(
        self,
        sampling_params: dict[str, Any],
        priority: int = 0,
        session_id: int = 0,
        **kwargs,
    ) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info") or {}
        split = str(extra_info.get("split", "test"))
        index = int(extra_info.get("index", kwargs.get("index", 0)))
        protocol = self._protocol(split)
        if not protocol["is_validation"]:
            raise NotImplementedError(
                "The Qwen3.5 ALFWorld integration is currently validation-only; "
                "GiGPO credit assignment must be ported before training is enabled."
            )

        config_path = Path(__file__).resolve().parents[1] / "environments" / "alfworld" / "configs" / "config_tw.yaml"
        games = _ordered_evaluation_games(
            str(config_path), protocol["eval_dataset"], protocol["env_seed"]
        )
        if index < 0 or index >= len(games):
            raise IndexError(
                f"Evaluation index {index} is outside the {len(games)} games in "
                f"{protocol['eval_dataset']}"
            )
        gamefile = games[index]
        recorded_gamefile = extra_info.get("gamefile")
        if recorded_gamefile and os.path.realpath(recorded_gamefile) != os.path.realpath(gamefile):
            raise ValueError(
                "Dataset game identity does not match the seeded ALFWorld order: "
                f"index={index}, dataset={recorded_gamefile!r}, expected={gamefile!r}"
            )
        task_type = _task_type(gamefile)

        local_env = LocalAlfworldEnv(
            str(config_path),
            gamefile,
            seed=protocol["env_seed"] + index,
            eval_dataset=protocol["eval_dataset"],
        )
        manager = AlfWorldEnvironmentManager(
            local_env,
            partial(alfworld_projection),
            protocol["num_attempts"],
            protocol["do_reflection"],
            self.config,
        )

        outputs: list[AgentLoopOutput] = []
        attempt_successes = [False] * protocol["num_attempts"]
        won = False
        request_id = (
            f"alfworld-{priority}-{session_id}"
            if getattr(self.rollout_config, "full_determinism", False)
            else uuid4().hex
        )

        try:
            observation, _ = manager.reset()
            for attempt_index in range(protocol["num_attempts"]):
                if won:
                    break

                if attempt_index > 0 and protocol["do_reflection"]:
                    observation, _ = manager.reflect()
                    reflection_output, reflection_text = await self._generate_one(
                        observation["text"][0],
                        sampling_params,
                        request_id,
                        int(priority),
                        "reflect",
                        attempt_index,
                        0,
                    )
                    _, _, _, reflection_infos = manager.step(
                        [reflection_text], phase="reflect"
                    )
                    reflection_output.extra_fields.update(
                        {
                            "gamefile": gamefile,
                            "task_type": task_type,
                            "parsed_action": reflection_infos[0].get(
                                "diagnostic_parsed_action"
                            ),
                            "is_action_valid": bool(
                                reflection_infos[0].get("is_action_valid", False)
                            ),
                        }
                    )
                    outputs.append(reflection_output)

                if attempt_index > 0:
                    observation, _ = manager.restart()

                for turn_index in range(protocol["max_turns"]):
                    play_output, play_text = await self._generate_one(
                        observation["text"][0],
                        sampling_params,
                        request_id,
                        int(priority),
                        "play",
                        attempt_index,
                        turn_index,
                    )
                    observation, rewards, dones, infos = manager.step(
                        [play_text], phase="play"
                    )
                    info = infos[0]
                    won = won or bool(info.get("won", False))
                    attempt_successes[attempt_index] = won
                    play_output.extra_fields.update(
                        {
                            "gamefile": gamefile,
                            "task_type": task_type,
                            "parsed_action": info.get("diagnostic_parsed_action"),
                            "is_action_valid": bool(info.get("is_action_valid", False)),
                            "action_is_effective": bool(
                                info.get("action_is_effective", False)
                            ),
                            "environment_reward": float(rewards[0]),
                            "won": won,
                        }
                    )
                    outputs.append(play_output)
                    if bool(dones[0]) or won:
                        break
        finally:
            local_env.close()

        cumulative_success = []
        seen_success = False
        for success in attempt_successes:
            seen_success = seen_success or success
            cumulative_success.append(float(seen_success))

        reward_extra_info: dict[str, float] = {
            "success_rate": float(won),
        }
        for attempt_index, success in enumerate(cumulative_success):
            reward_extra_info[f"success_rate[{attempt_index}]"] = success
            reward_extra_info[f"{task_type}|success_rate[{attempt_index}]"] = success

        outputs[-1].reward_score = 10.0 * float(won)
        outputs[-1].extra_fields["reward_extra_info"] = reward_extra_info
        outputs[-1].extra_fields["attempt_successes"] = cumulative_success
        outputs[-1].extra_fields["gamefile"] = gamefile
        outputs[-1].extra_fields["task_type"] = task_type
        return outputs[-1]
