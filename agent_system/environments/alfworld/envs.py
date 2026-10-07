import asyncio
import os
import resource
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray
from contextlib import asynccontextmanager
from copy import copy
from functools import lru_cache
from itertools import cycle

from .alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward


def repeated_shuffled_game_cycle(gamefiles, repeats, seed):
    """Yield each shuffled game ``repeats`` times before advancing.

    TextWorld normally selects one new game per batch slot. GiGPO instead needs
    every slot in a rollout group to start from the same game, with a new game
    selected for the group on its next reset.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    gamefiles = list(gamefiles)
    if not gamefiles:
        raise ValueError("gamefiles must not be empty")
    rng = np.random.RandomState(seed)
    while True:
        rng.shuffle(gamefiles)
        for gamefile in gamefiles:
            for _ in range(repeats):
                yield gamefile

@ray.remote
class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(
        self,
        config,
        seed,
        base_env,
        game_files=None,
        batch_size=1,
        repeat_game_within_batch=False,
        preserve_game_order=False,
    ):
        # Evaluation workers receive a fixed game pool whose size equals their
        # internal batch. Every game is therefore loaded exactly once, while a
        # small number of Ray processes can host the complete evaluation set.
        if game_files is not None and preserve_game_order:
            base_env.game_files = list(game_files)
            base_env.num_games = len(game_files)
            batch_size = len(game_files)
        self.batch_size = batch_size
        self.env = base_env.init_env(batch_size=batch_size)
        self.env.seed(seed)
        if game_files is not None:
            # Evaluation supplies an explicit ordered game list. TextWorld's
            # seed() shuffles that list before the first reset, which breaks
            # same-task diagnostic groups and makes row order incidental.
            self.env._gamefiles_iterator = cycle(list(game_files))
        elif repeat_game_within_batch:
            self.env._gamefiles_iterator = repeated_shuffled_game_cycle(
                self.env.gamefiles, repeats=batch_size, seed=seed
            )
    
    def step(self, actions):
        """Execute a step in the environment"""
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos

    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image
    
    def restart(self):
        '''Get back to init state of the game'''
        self.env.last_commands = [None] * self.batch_size
        self.env.obs, infos = self.env.batch_env.reset()

        obs = self.env.obs
        infos['observation_text'] = obs
        return obs, infos

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed=0, env_num=1, group_n=1, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.num_cpus_per_worker = float(env_kwargs.get('num_cpus_per_worker', 0.1))
        self.num_gpus_per_worker = float(env_kwargs.get('num_gpus_per_worker', 0))
        self.games_per_worker = int(env_kwargs.get('games_per_worker', 32))
        self.preserve_game_order = bool(env_kwargs.get('preserve_game_order', False))
        if self.games_per_worker <= 0:
            raise ValueError("games_per_worker must be positive")

        evaluation_games = None
        if not is_train:
            if env_num > len(base_env.game_files):
                raise ValueError(
                    f"Requested {env_num} distinct evaluation games, but "
                    f"the selected pool contains only {len(base_env.game_files)}"
                )
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(base_env.game_files))[:env_num]
            evaluation_games = [base_env.game_files[index] for index in indices]
            # A diagnostic may request independent rollouts of each exact
            # evaluation game. Standard validation uses group_n=1 and is
            # therefore unchanged.
            evaluation_games = [
                gamefile
                for gamefile in evaluation_games
                for _ in range(group_n)
            ]

        # Training uses one Ray actor per task group. Each actor hosts the
        # group's rollouts as a synchronous TextWorld batch and repeats the
        # selected game across every batch slot. This preserves GiGPO grouping
        # without paying for one Python/Ray process per rollout.
        if evaluation_games is None:
            config['env']['textworld_asynchronous'] = False
            worker_specs = [
                (None, self.group_n, True) for _ in range(env_num)
            ]
        else:
            config['env']['textworld_asynchronous'] = False
            worker_specs = [
                (evaluation_games[start:start + self.games_per_worker], None, False)
                for start in range(0, len(evaluation_games), self.games_per_worker)
            ]

        self.workers = []
        self.worker_sizes = []
        game_offset = 0
        for worker_index, (game_files, batch_size, repeat_game) in enumerate(worker_specs):
            worker_size = len(game_files) if game_files is not None else batch_size
            worker_seed = seed + (game_offset if game_files is not None else worker_index)
            worker = AlfworldWorker.options(
                num_cpus=self.num_cpus_per_worker,
                num_gpus=self.num_gpus_per_worker,
            ).remote(
                config,
                worker_seed,
                base_env,
                game_files,
                worker_size,
                repeat_game,
                self.preserve_game_order,
            )
            self.workers.append(worker)
            self.worker_sizes.append(worker_size)
            game_offset += worker_size

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        action_offset = 0
        for worker, worker_size in zip(self.workers, self.worker_sizes):
            worker_actions = actions[action_offset:action_offset + worker_size]
            future = worker.step.remote(worker_actions)
            futures.append(future)
            action_offset += worker_size

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        env_index = 0
        for obs, scores, dones, infos in results:
            for local_index in range(len(obs)):
                info = {key: values[local_index] for key, values in infos.items()}
                text_obs_list.append(obs[local_index])
                dones_list.append(dones[local_index])
                info_list.append(info)
                self.prev_admissible_commands[env_index] = info['admissible_commands']
                rewards_list.append(compute_reward(info, self.multi_modal))
                env_index += 1

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        env_index = 0
        for obs, infos in results:
            for local_index in range(len(obs)):
                info = {key: values[local_index] for key, values in infos.items()}
                text_obs_list.append(obs[local_index])
                self.prev_admissible_commands[env_index] = info['admissible_commands']
                info_list.append(info)
                env_index += 1

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def restart(self):
        '''Get back to init state of the game'''
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.restart.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        env_index = 0
        for obs, infos in results:
            for local_index in range(len(obs)):
                info = {key: values[local_index] for key, values in infos.items()}
                text_obs_list.append(obs[local_index])
                self.prev_admissible_commands[env_index] = info['admissible_commands']
                info_list.append(info)
                env_index += 1

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list
    
    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        image_batches = ray.get(futures)
        return [image for batch in image_batches for image in batch]

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)


class LocalAlfworldEnv(gym.Env):
    """One in-process text ALFWorld game for an asynchronous agent session.

    The legacy rollout path batches games behind Ray actors.  Modern VERL
    already distributes agent sessions across Ray workers, so creating another
    Ray actor per session would add nested scheduling and would block the
    worker's asyncio event loop on ``ray.get``.  This wrapper keeps the same
    TextWorld and reward behavior while giving one agent-loop coroutine its own
    isolated game state.
    """

    def __init__(self, alf_config_path, gamefile, seed=0, eval_dataset="eval_all"):
        super().__init__()
        self.alf_config_path = os.path.realpath(alf_config_path)
        self.environment_signature = _local_alfworld_environment_signature(
            self.alf_config_path, eval_dataset
        )
        base_env = copy(
            _local_alfworld_base_environment(self.alf_config_path, eval_dataset)
        )
        if gamefile not in base_env.game_files:
            raise ValueError(
                f"Game {gamefile!r} is not part of ALFWorld split {eval_dataset!r}"
            )

        base_env.game_files = [gamefile]
        base_env.num_games = 1
        self.multi_modal = base_env.config["env"]["type"] == "AlfredThorEnv"
        if self.multi_modal:
            raise ValueError("LocalAlfworldEnv currently supports text ALFWorld only")

        self.env = base_env.init_env(batch_size=1)
        self.num_processes = 1
        self.prev_admissible_commands = [None]
        self.configure(gamefile, seed=seed, eval_dataset=eval_dataset)

    def configure(self, gamefile, seed=0, eval_dataset="eval_all"):
        """Select the next isolated game without rebuilding the TextWorld stack."""
        signature = _local_alfworld_environment_signature(
            self.alf_config_path, eval_dataset
        )
        if signature != self.environment_signature:
            raise ValueError(
                "Cannot reuse an ALFWorld environment across incompatible wrapper "
                f"configurations: {self.environment_signature!r} != {signature!r}"
            )

        base_env = _local_alfworld_base_environment(
            self.alf_config_path, eval_dataset
        )
        if gamefile not in base_env.game_files:
            raise ValueError(
                f"Game {gamefile!r} is not part of ALFWorld split {eval_dataset!r}"
            )

        # TextworldBatchGymEnv.reset() consumes this iterator, closes the
        # previously loaded interpreter, loads this exact game and resets all
        # mutable game state.  Keeping the outer wrapper avoids repeatedly
        # registering environments and constructing wrapper/interpreter stacks.
        self.env.gamefiles = [gamefile]
        self.env.seed(seed)
        self.env._gamefiles_iterator = cycle([gamefile])
        self.prev_admissible_commands = [None]

    def prepare_for_pool(self):
        """Drop references to trajectory data before returning this slot."""
        self.prev_admissible_commands = [None]
        self.env.last_commands = [None]
        self.env.obs = None
        # SyncBatchEnv caches the complete result of the last step to implement
        # terminal-state behavior. The next reset always overwrites this field,
        # so retaining it while idle only keeps stale observations/infos alive.
        if self.env.batch_env is not None:
            self.env.batch_env.last = [None]

    @staticmethod
    def _split_infos(infos):
        return [{key: values[0] for key, values in infos.items()}]

    def reset(self):
        observations, infos = self.env.reset()
        split_infos = self._split_infos(infos)
        self.prev_admissible_commands[0] = split_infos[0]["admissible_commands"]
        return list(observations), None, split_infos

    def step(self, actions):
        if len(actions) != 1:
            raise ValueError("LocalAlfworldEnv accepts exactly one action")
        observations, _, dones, infos = self.env.step(actions)
        split_infos = self._split_infos(infos)
        self.prev_admissible_commands[0] = split_infos[0]["admissible_commands"]
        rewards = [compute_reward(split_infos[0], self.multi_modal)]
        return list(observations), None, rewards, list(dones), split_infos

    def restart(self):
        self.env.last_commands = [None]
        self.env.obs, infos = self.env.batch_env.reset()
        split_infos = self._split_infos(infos)
        self.prev_admissible_commands[0] = split_infos[0]["admissible_commands"]
        return list(self.env.obs), None, split_infos

    @property
    def get_admissible_commands(self):
        return self.prev_admissible_commands

    def close(self):
        close = getattr(self.env, "close", None)
        if close is not None:
            close()


@lru_cache(maxsize=None)
def _local_alfworld_base_environment(alf_config_path, eval_dataset):
    """Scan and cache one immutable game catalogue per split and process."""
    config = load_config_file(alf_config_path)
    config["env"]["textworld_asynchronous"] = False
    env_type = config["env"]["type"]
    return get_environment(env_type)(config, train_eval=eval_dataset)


@lru_cache(maxsize=None)
def _local_alfworld_environment_signature(alf_config_path, eval_dataset):
    """Describe settings that affect the TextWorld wrapper construction."""
    base_env = _local_alfworld_base_environment(alf_config_path, eval_dataset)
    domain_randomization = bool(
        base_env.config["env"].get("domain_randomization", False)
        if eval_dataset == "train"
        else False
    )
    return (
        base_env.config["env"]["type"],
        domain_randomization,
        base_env.config["general"]["training_method"],
    )


def _process_rss_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes. ALFWorld training runs on Linux,
    # but retaining the distinction keeps local diagnostics meaningful.
    divisor = 1024**2 if os.uname().sysname == "Linux" else 1024**3
    return rss / divisor


class LocalAlfworldEnvPool:
    """Bound the number of reusable TextWorld environments in one Ray worker."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("ALFWorld environment pool capacity must be positive")
        self.capacity = int(capacity)
        self._idle = asyncio.LifoQueue(maxsize=self.capacity)
        self._allocation_lock = asyncio.Lock()
        self._size = 0
        self._leases = 0

    async def acquire(self, alf_config_path, gamefile, seed=0, eval_dataset="eval_all"):
        try:
            local_env = self._idle.get_nowait()
        except asyncio.QueueEmpty:
            create = False
            async with self._allocation_lock:
                if self._size < self.capacity:
                    self._size += 1
                    create = True
            if create:
                try:
                    local_env = LocalAlfworldEnv(
                        alf_config_path,
                        gamefile,
                        seed=seed,
                        eval_dataset=eval_dataset,
                    )
                except BaseException:
                    async with self._allocation_lock:
                        self._size -= 1
                    raise
                print(
                    "ALFWorld environment pool allocated "
                    f"slot {self._size}/{self.capacity}; "
                    f"worker peak RSS={_process_rss_gb():.3f} GiB"
                )
                return local_env
            local_env = await self._idle.get()

        try:
            local_env.configure(
                gamefile,
                seed=seed,
                eval_dataset=eval_dataset,
            )
        except BaseException:
            local_env.close()
            async with self._allocation_lock:
                self._size -= 1
            raise
        return local_env

    async def release(self, local_env):
        local_env.prepare_for_pool()
        self._leases += 1
        self._idle.put_nowait(local_env)
        if self._leases % self.capacity == 0:
            print(
                "ALFWorld environment pool reused "
                f"{self._leases} leases across {self._size} slots; "
                f"worker peak RSS={_process_rss_gb():.3f} GiB"
            )

    @asynccontextmanager
    async def lease(self, alf_config_path, gamefile, seed=0, eval_dataset="eval_all"):
        local_env = await self.acquire(
            alf_config_path,
            gamefile,
            seed=seed,
            eval_dataset=eval_dataset,
        )
        try:
            yield local_env
        finally:
            await self.release(local_env)


_LOCAL_ALFWORLD_ENV_POOLS = {}


def get_local_alfworld_env_pool(
    alf_config_path,
    eval_dataset,
    capacity,
):
    """Return the process-local pool compatible with this ALFWorld split."""
    config_path = os.path.realpath(alf_config_path)
    signature = _local_alfworld_environment_signature(config_path, eval_dataset)
    # asyncio synchronization primitives belong to the event loop that drives
    # them. Ray uses one persistent loop per async actor; including it in the
    # key also keeps standalone callers that use several asyncio.run() calls
    # from accidentally reusing a queue owned by an already-closed loop.
    event_loop = asyncio.get_running_loop()
    key = (config_path, signature, int(capacity), id(event_loop))
    pool = _LOCAL_ALFWORLD_ENV_POOLS.get(key)
    if pool is None:
        pool = LocalAlfworldEnvPool(capacity)
        _LOCAL_ALFWORLD_ENV_POOLS[key] = pool
    return pool


def build_alfworld_envs(alf_config_path, seed, env_num, group_n, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, is_train, env_kwargs)
