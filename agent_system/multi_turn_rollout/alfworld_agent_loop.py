"""Modern VERL agent loop for LaMer's text ALFWorld protocol."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache, partial
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from agent_system.environments.alfworld.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.alfworld.envs import (
    get_environment,
    get_local_alfworld_env_pool,
    load_config_file,
)
from agent_system.environments.alfworld.projection import alfworld_projection
from agent_system.multi_turn_rollout.episode_credit import episode_scores_for_records
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


@lru_cache(maxsize=None)
def _task_type(gamefile: str) -> str:
    trajectory_path = gamefile.replace("game.tw-pddl", "traj_data.json")
    with open(trajectory_path, encoding="utf-8") as stream:
        return json.load(stream)["task_type"]


def _markdown_block(value: Any) -> str:
    """Render arbitrary text as an indented Markdown block."""
    text = str(value if value is not None else "")
    return "\n".join(f"    {line}" for line in text.splitlines()) or "    "


@lru_cache(maxsize=None)
def _selected_trace_games(
    games: tuple[str, ...],
    task_count: int,
    samples_per_task: int,
    sample_seed: int,
) -> frozenset[str]:
    """Select a stable, task-balanced subset of validation games to trace."""
    if samples_per_task <= 0:
        return frozenset()

    by_task: dict[str, list[str]] = {}
    for gamefile in games[:task_count]:
        by_task.setdefault(_task_type(gamefile), []).append(gamefile)

    selected: set[str] = set()
    for task_type in sorted(by_task):
        ranked = sorted(
            by_task[task_type],
            key=lambda gamefile: hashlib.sha256(
                f"{sample_seed}:{gamefile}".encode("utf-8")
            ).hexdigest(),
        )
        selected.update(ranked[:samples_per_task])
    return frozenset(selected)


class AlfWorldAgentLoop(AgentLoopBase):
    """Run LaMer's per-action prompting protocol with modern VERL.

    Every play turn and reflection is generated from a fresh, independently
    templated user prompt, matching LaMer's original rollout semantics.
    Validation returns the final model call with whole-episode outcome metadata.
    Training returns every model call so that LaMer's GiGPO trainer can treat
    each action and reflection as a separate policy sample.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.presence_penalty = float(self.config.env.get("presence_penalty", 0.0))
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                "env.presence_penalty must be between -2.0 and 2.0, got "
                f"{self.presence_penalty}"
            )
        self.brief_response_instruction = bool(
            self.config.env.get("brief_response_instruction", False)
        )

        self.validation_trace_dir = str(
            self.config.env.get("validation_trace_dir", "")
        )
        self.validation_trajectory_samples_per_task = int(
            self.config.env.get("validation_trajectory_samples_per_task", 0)
        )
        self.validation_trajectory_sample_seed = int(
            self.config.env.get("validation_trajectory_sample_seed", 0)
        )
        self.validation_task_count = int(
            self.config.env.get("validation_task_count", 0)
        )
        self.local_env_pool_size = int(
            self.config.env.alfworld.get("local_env_pool_size", 16)
        )
        if self.local_env_pool_size <= 0:
            raise ValueError("env.alfworld.local_env_pool_size must be positive")

    @staticmethod
    def _append_trace(trace_path: Path | None, text: str) -> None:
        """Append and flush a section so interrupted runs remain inspectable."""
        if trace_path is None:
            return
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()

    def _trace_path(
        self,
        games: tuple[str, ...],
        gamefile: str,
        task_type: str,
        index: int,
        session_id: int,
    ) -> Path | None:
        if not self.validation_trace_dir:
            return None
        task_count = self.validation_task_count or len(games)
        selected = _selected_trace_games(
            games,
            min(task_count, len(games)),
            self.validation_trajectory_samples_per_task,
            self.validation_trajectory_sample_seed,
        )
        if gamefile not in selected:
            return None

        trace_dir = Path(self.validation_trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        safe_task_type = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in task_type
        )
        return trace_dir / (
            f"task_{index:03d}_{safe_task_type}_session_{session_id}.md"
        )

    def _write_trace_step(
        self,
        trace_path: Path | None,
        output: AgentLoopOutput,
        *,
        parsed_action: Any,
        is_action_valid: bool,
        action_is_effective: bool,
        environment_reward: float,
        won: bool,
        next_observation: Any = None,
    ) -> None:
        if trace_path is None:
            return
        fields = output.extra_fields
        phase = fields["phase"]
        section = (
            f"## Attempt {int(fields['attempt_index']) + 1}, {phase}, "
            f"turn {int(fields['turn_index']) + 1}\n\n"
            "### Prompt sent to the model\n\n"
            f"{_markdown_block(fields['prompt_text'])}\n\n"
            "### Raw model response\n\n"
            f"{_markdown_block(fields['response_text'])}\n\n"
            "### Parsed action/reflection\n\n"
            f"{_markdown_block(parsed_action)}\n\n"
        )
        if phase == "play":
            section += (
                "### Resulting observation\n\n"
                f"{_markdown_block(next_observation)}\n\n"
            )
        section += (
            "### Outcome\n\n"
            f"- Generated tokens: `{fields['response_token_count']}`\n"
            f"- Reached response-token limit: "
            f"`{fields['response_token_cap_hit']}`\n"
            f"- Parse valid: `{is_action_valid}`\n"
            f"- Observation changed: `{action_is_effective}`\n"
            f"- Immediate environment reward: `{environment_reward}`\n"
            f"- Won by this point: `{won}`\n\n"
        )
        self._append_trace(trace_path, section)

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

    @staticmethod
    def _assign_lamer_credit(
        outputs: list[AgentLoopOutput],
        num_attempts: int,
        step_gamma: float,
        traj_gamma: float,
    ) -> None:
        """Attach the rewards used by LaMer's original GiGPO implementation.

        ``episode_reward`` retains original LaMer credit: it is the total
        environment reward in the corresponding play attempt, while reflections
        receive zero. ``future_episode_reward`` includes later-attempt rewards
        discounted by ``traj_gamma``; reflections receive the score of the retry
        they guide. The trainer selects between these episode fields.

        Step returns are unchanged: they propagate future play rewards backwards,
        using ``step_gamma`` inside an attempt and ``traj_gamma`` when crossing an
        attempt boundary. A reflection receives the future return available at
        its position.
        """
        attempt_rewards = [0.0] * num_attempts
        for output in outputs:
            fields = output.extra_fields
            if fields["phase"] == "play":
                attempt_rewards[int(fields["attempt_index"])] += float(
                    fields["environment_reward"]
                )

        running_return = 0.0
        current_attempt: int | None = None
        step_returns = [0.0] * len(outputs)
        for index in range(len(outputs) - 1, -1, -1):
            fields = outputs[index].extra_fields
            if fields["phase"] == "play":
                attempt_index = int(fields["attempt_index"])
                discount = (
                    step_gamma
                    if current_attempt is None or attempt_index == current_attempt
                    else traj_gamma
                )
                running_return = float(fields["environment_reward"]) + discount * running_return
                current_attempt = attempt_index
            step_returns[index] = running_return

        attempt_indices = [
            int(output.extra_fields["attempt_index"]) for output in outputs
        ]
        phases = [str(output.extra_fields["phase"]) for output in outputs]
        original_episode_scores = episode_scores_for_records(
            attempt_rewards,
            attempt_indices,
            phases,
            future_aware=False,
        )
        future_episode_scores = episode_scores_for_records(
            attempt_rewards,
            attempt_indices,
            phases,
            future_aware=True,
            step_returns=step_returns,
        )

        for output, episode_score, future_episode_score, step_return in zip(
            outputs,
            original_episode_scores,
            future_episode_scores,
            step_returns,
            strict=True,
        ):
            fields = output.extra_fields
            fields["episode_reward"] = episode_score
            fields["future_episode_reward"] = future_episode_score
            fields["step_return"] = step_return

    async def _generate_one(
        self,
        prompt: str,
        sampling_params: dict[str, Any],
        request_id: str,
        priority: int,
        phase: str,
        attempt_index: int,
        turn_index: int,
        capture_text: bool = False,
    ) -> tuple[AgentLoopOutput, str]:
        if self.brief_response_instruction:
            if phase == "play":
                prompt = (
                    f"{prompt.rstrip()}\n\n"
                    "Keep your reasoning brief. Always finish your response with "
                    "exactly one admissible action inside <action>...</action>.\n"
                )
            elif phase == "reflect":
                prompt = (
                    f"{prompt.rstrip()}\n\n"
                    "Keep your reasoning brief. Identify the main mistake without retelling "
                    "the full trajectory or being overly verbose. Always finish with exactly "
                    "one non-empty <remark>...</remark> block containing a concise reflection "
                    "and corrected plan. Stop immediately after </remark>.\n"
                )
            else:
                raise ValueError(f"Unsupported ALFWorld generation phase: {phase}")
        messages = [{"role": "user", "content": prompt}]
        # ALFWorld is text-only.  Build its prompt synchronously because this
        # machine does not reliably wake a reused asyncio executor thread after
        # its first job.  The same upstream Qwen3.5 processor-backed builder is
        # still used, so this changes scheduling rather than prompt tokens.
        # Tokenization takes milliseconds and is negligible beside generation.
        prompt_ids = self.continuous_token_builder.build_initial_tokens(messages)
        prompt_ids = self._cap_text_prompt_length(prompt_ids)

        metrics: dict[str, Any] = {}
        request_sampling_params = dict(sampling_params)
        request_sampling_params["presence_penalty"] = self.presence_penalty
        with simple_timer("generate_sequences", metrics):
            generated: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=request_sampling_params,
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
                # Preserve rollout-server metadata, especially the authoritative
                # min/max model-weight versions used by VERL's staleness metrics.
                # Constructing a fresh dictionary here previously discarded it.
                **generated.extra_fields,
                "phase": phase,
                "attempt_index": attempt_index,
                "turn_index": turn_index,
                "response_token_count": len(response_ids),
                "response_token_limit": self.response_length,
                "response_token_cap_hit": len(response_ids) >= self.response_length,
                **(
                    {"prompt_text": prompt, "response_text": text}
                    if capture_text
                    else {}
                ),
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
    ) -> AgentLoopOutput | list[AgentLoopOutput]:
        extra_info = kwargs.get("extra_info") or {}
        split = str(extra_info.get("split", "test"))
        index = int(extra_info.get("index", kwargs.get("index", 0)))
        protocol = self._protocol(split)

        config_path = Path(__file__).resolve().parents[1] / "environments" / "alfworld" / "configs" / "config_tw.yaml"
        recorded_gamefile = extra_info.get("gamefile")
        if protocol["is_validation"]:
            games = _ordered_evaluation_games(
                str(config_path), protocol["eval_dataset"], protocol["env_seed"]
            )
            if index < 0 or index >= len(games):
                raise IndexError(
                    f"Evaluation index {index} is outside the {len(games)} games in "
                    f"{protocol['eval_dataset']}"
                )
            gamefile = games[index]
            if recorded_gamefile and os.path.realpath(recorded_gamefile) != os.path.realpath(gamefile):
                raise ValueError(
                    "Dataset game identity does not match the seeded ALFWorld order: "
                    f"index={index}, dataset={recorded_gamefile!r}, expected={gamefile!r}"
                )
            environment_split = protocol["eval_dataset"]
        else:
            if not recorded_gamefile:
                raise ValueError("An explicit ALFWorld training gamefile is required")
            gamefile = str(recorded_gamefile)
            environment_split = "train"
        task_type = _task_type(gamefile)
        trace_path = (
            self._trace_path(games, gamefile, task_type, index, session_id)
            if protocol["is_validation"]
            else None
        )

        env_pool = get_local_alfworld_env_pool(
            str(config_path),
            environment_split,
            self.local_env_pool_size,
        )
        local_env = await env_pool.acquire(
            str(config_path),
            gamefile,
            seed=protocol["env_seed"]
            + int(extra_info.get("env_seed_offset", index)),
            eval_dataset=environment_split,
        )
        try:
            manager = AlfWorldEnvironmentManager(
                local_env,
                partial(alfworld_projection),
                protocol["num_attempts"],
                protocol["do_reflection"],
                self.config,
            )
        except BaseException:
            await env_pool.release(local_env)
            raise

        outputs: list[AgentLoopOutput] = []
        attempt_successes = [False] * protocol["num_attempts"]
        won = False
        task_uid = str(kwargs.get("uid", gamefile))
        trajectory_uid = f"{task_uid}:{session_id}"
        request_id = (
            f"alfworld-{task_uid}-{session_id}"
            if getattr(self.rollout_config, "full_determinism", False)
            else uuid4().hex
        )

        try:
            observation, _ = manager.reset()
            if trace_path is not None:
                trace_path.write_text(
                    "# ALFWorld validation trajectory\n\n"
                    f"- Dataset index: `{index}`\n"
                    f"- Session: `{session_id}`\n"
                    f"- Task type: `{task_type}`\n"
                    f"- Game: `{gamefile}`\n"
                    f"- Sampling seed: `{self.rollout_config.seed}`\n"
                    f"- Presence penalty: `{self.presence_penalty}`\n\n"
                    "## Initial observation\n\n"
                    f"{_markdown_block(observation['anchor'][0])}\n\n",
                    encoding="utf-8",
                )
                print(f"Writing ALFWorld validation trace to {trace_path}")
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
                        capture_text=trace_path is not None,
                    )
                    _, _, _, reflection_infos = manager.step(
                        [reflection_text], phase="reflect"
                    )
                    reflection_output.extra_fields.update(
                        {
                            "task_uid": task_uid,
                            "traj_uid": trajectory_uid,
                            "gamefile": gamefile,
                            "task_type": task_type,
                            "anchor_obs": observation["anchor"][0],
                            "parsed_action": reflection_infos[0].get(
                                "diagnostic_parsed_action"
                            ),
                            "is_action_valid": bool(
                                reflection_infos[0].get("is_action_valid", False)
                            ),
                            "action_is_effective": bool(
                                reflection_infos[0].get("action_is_effective", False)
                            ),
                            "environment_reward": 0.0,
                            "won": False,
                        }
                    )
                    outputs.append(reflection_output)
                    self._write_trace_step(
                        trace_path,
                        reflection_output,
                        parsed_action=reflection_infos[0].get(
                            "diagnostic_parsed_action"
                        ),
                        is_action_valid=bool(
                            reflection_infos[0].get("is_action_valid", False)
                        ),
                        action_is_effective=bool(
                            reflection_infos[0].get("action_is_effective", False)
                        ),
                        environment_reward=0.0,
                        won=False,
                    )

                if attempt_index > 0:
                    observation, _ = manager.restart()

                for turn_index in range(protocol["max_turns"]):
                    anchor_obs = observation["anchor"][0]
                    play_output, play_text = await self._generate_one(
                        observation["text"][0],
                        sampling_params,
                        request_id,
                        int(priority),
                        "play",
                        attempt_index,
                        turn_index,
                        capture_text=trace_path is not None,
                    )
                    observation, rewards, dones, infos = manager.step(
                        [play_text], phase="play"
                    )
                    info = infos[0]
                    won = won or bool(info.get("won", False))
                    attempt_successes[attempt_index] = won
                    play_output.extra_fields.update(
                        {
                            "task_uid": task_uid,
                            "traj_uid": trajectory_uid,
                            "gamefile": gamefile,
                            "task_type": task_type,
                            "anchor_obs": anchor_obs,
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
                    self._write_trace_step(
                        trace_path,
                        play_output,
                        parsed_action=info.get("diagnostic_parsed_action"),
                        is_action_valid=bool(info.get("is_action_valid", False)),
                        action_is_effective=bool(
                            info.get("action_is_effective", False)
                        ),
                        environment_reward=float(rewards[0]),
                        won=won,
                        next_observation=observation["anchor"][0],
                    )
                    if bool(dones[0]) or won:
                        break
        except Exception as exc:
            self._append_trace(
                trace_path,
                "## Trajectory interrupted\n\n"
                f"`{type(exc).__name__}: {exc}`\n",
            )
            raise
        finally:
            await env_pool.release(local_env)
        # The policy records below no longer need the trajectory-local history
        # manager. Release its observation/action strings before queue packing.
        del manager, local_env

        cumulative_success = []
        seen_success = False
        for success in attempt_successes:
            seen_success = seen_success or success
            cumulative_success.append(float(seen_success))

        algorithm_config = self.config.algorithm
        self._assign_lamer_credit(
            outputs,
            protocol["num_attempts"],
            float(algorithm_config.get("step_gamma", algorithm_config.gamma)),
            float(algorithm_config.get("traj_gamma", algorithm_config.gamma)),
        )
        # Raw text is useful while writing selected Markdown traces, but the
        # policy update consumes token IDs and the GiGPO metadata below. Do not
        # duplicate prompt/response strings in TransferQueue.
        for output in outputs:
            output.extra_fields.pop("prompt_text", None)
            output.extra_fields.pop("response_text", None)

        reward_extra_info: dict[str, float] = {
            "success_rate": float(won),
        }
        for attempt_index, success in enumerate(cumulative_success):
            reward_extra_info[f"success_rate[{attempt_index}]"] = success
            reward_extra_info[f"{task_type}|success_rate[{attempt_index}]"] = success

        for phase in ("play", "reflect"):
            phase_outputs = [
                output
                for output in outputs
                if output.extra_fields["phase"] == phase
            ]
            if not phase_outputs:
                continue
            prefix = "play" if phase == "play" else "reflection"
            valid_count = sum(
                bool(output.extra_fields["is_action_valid"])
                for output in phase_outputs
            )
            response_token_count = sum(
                int(output.extra_fields["response_token_count"])
                for output in phase_outputs
            )
            cap_hit_count = sum(
                bool(output.extra_fields["response_token_cap_hit"])
                for output in phase_outputs
            )
            record_count = len(phase_outputs)
            reward_extra_info.update(
                {
                    f"diagnostics/{prefix}_record_count": float(record_count),
                    f"diagnostics/{prefix}_valid_count": float(valid_count),
                    f"diagnostics/{prefix}_response_token_count": float(
                        response_token_count
                    ),
                    f"diagnostics/{prefix}_token_cap_hit_count": float(
                        cap_hit_count
                    ),
                    (
                        "diagnostics/play_valid_action_ratio"
                        if phase == "play"
                        else "diagnostics/reflection_parse_valid_ratio"
                    ): (
                        valid_count / record_count
                    ),
                    f"diagnostics/{prefix}_mean_response_tokens": (
                        response_token_count / record_count
                    ),
                    f"diagnostics/{prefix}_token_cap_hit_rate": (
                        cap_hit_count / record_count
                    ),
                }
            )

        total_response_tokens = sum(
            int(output.extra_fields["response_token_count"])
            for output in outputs
        )
        total_cap_hits = sum(
            bool(output.extra_fields["response_token_cap_hit"])
            for output in outputs
        )
        reward_extra_info.update(
            {
                "diagnostics/response_record_count": float(len(outputs)),
                "diagnostics/response_token_count": float(total_response_tokens),
                "diagnostics/token_cap_hit_count": float(total_cap_hits),
                "diagnostics/mean_response_tokens": (
                    total_response_tokens / len(outputs)
                ),
                "diagnostics/token_cap_hit_rate": total_cap_hits / len(outputs),
            }
        )

        self._append_trace(
            trace_path,
            "## Final result\n\n"
            f"- Attempt success (cumulative): `{cumulative_success}`\n"
            f"- Overall success: `{won}`\n",
        )

        outputs[-1].reward_score = 10.0 * float(won)
        outputs[-1].extra_fields["reward_extra_info"] = reward_extra_info
        outputs[-1].extra_fields["attempt_successes"] = cumulative_success
        outputs[-1].extra_fields["gamefile"] = gamefile
        outputs[-1].extra_fields["task_type"] = task_type
        return outputs[-1] if protocol["is_validation"] else outputs
