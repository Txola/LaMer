import ray
import gym
import numpy as np
from typing import Dict, Any, Tuple, List

from .game.env import MineSweeper

@ray.remote(num_cpus=0.1)
class MineSweeperWorker:
    """
    Ray remote actor for MineSweeper environments.
    Each worker holds its own MineSweeper environment instance.
    """

    def __init__(self, env_kwargs: Dict[str, Any] = None):
        """Initialize the MineSweeper environment in this worker"""
        self.env = MineSweeper(**env_kwargs)
        ## Code for MetaRL ## To allow restart for MetaRL, we add an env_copy to allow returning to init_state
        self.env_copy = self.env.copy()

    def step(self, action):
        """Execute a step in the environment."""
        act = "L"
        x, y = action
        obs, reward, done, info = self.env.step(act, x, y)
        return obs, reward, done, info
    
    def reset(self, seed_for_reset):
        """Reset the environment with a new episode."""
        obs, info = self.env.reset(seed=seed_for_reset)
        ## Code for MetaRL ##
        self.env_copy = self.env.copy()
        return obs, info

    def render(self, mode_for_render='board'):
        """Render the environment."""
        if mode_for_render == "board":
            return self.env.to_board_str_repr()
        else:
            raise ValueError(f"Invalid render mode: {mode_for_render}. Must be one of 'table', 'board', or 'coord'.")
    
    ## Code for MetaRL ##
    def restart(self):
        '''Get back to init state of the game'''
        self.env = self.env_copy.copy()
        obs = self.render(self.env.board_type) 
        info = {
            'won': False,
        }
        return obs, info
    
    def get_board_mine(self):
        return self.env.board_mine
        

class MineSweeperMultiProcessEnv(gym.Env):
    """
    Ray-based wrapper for the MineSweeper environment.
    Each Ray actor creates an independent MineSweeperEnv instance.
    The main process communicates with Ray actors to collect step/reset results.
    """

    def __init__(self,
                 seed=0, 
                 env_num=1, 
                 group_n=1, 
                 is_train=True,
                 env_kwargs=None):
        """
        - env_num: Number of different environments
        - group_n: Number of same environments in each group (for GRPO and GiGPO)
        - env_kwargs: Dictionary of parameters for initializing MineSweeperEnv
        - seed: Random seed for reproducibility
        """
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        np.random.seed(seed)

        if env_kwargs is None:
            env_kwargs = {}

        # Create Ray remote actors instead of processes
        self.workers = []
        for i in range(self.num_processes):
            worker = MineSweeperWorker.remote(env_kwargs)
            self.workers.append(worker)

    def step(self, actions):
        """
        Perform step in parallel.
        :param actions: list[int], length must match self.num_processes
        :return:
            obs_list, reward_list, done_list, info_list
            Each is a list of length self.num_processes
        """
        assert len(actions) == self.num_processes

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self.workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        Perform reset in parallel.
        :return: obs_list and info_list, the initial observations for each environment
        """
        # randomly generate self.env_num seeds
        if self.is_train:
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)

        # repeat the seeds for each group
        seeds = np.repeat(seeds, self.group_n)
        seeds = seeds.tolist()

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote(seeds[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list = []
        info_list = []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def render(self, mode='table', env_idx=None):
        """
        Request rendering from Ray actor environments.
        Can specify env_idx to get render result from a specific environment,
        otherwise returns a list from all environments.
        """
        if env_idx is not None:
            future = self.workers[env_idx].render.remote(mode)
            return ray.get(future)
        else:
            futures = []
            for worker in self.workers:
                future = worker.render.remote(mode)
                futures.append(future)
            results = ray.get(futures)
            return results
    
    ## Code for MetaRL ##
    def restart(self):
        '''Get back to init state of the game'''
        futures = [worker.restart.remote() for worker in self.workers]
        results = ray.get(futures)
        obs_list = []
        info_list = []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def close(self):
        """
        Close all Ray actors
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

    def __del__(self):
        self.close()


class MineSweeperLocalVectorEnv(gym.Env):
    """In-process vector backend with the same logical layout as the Ray backend."""

    def __init__(self, seed=0, env_num=1, group_n=1, is_train=True, env_kwargs=None):
        super().__init__()
        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.seed_rng = np.random.RandomState(seed)
        env_kwargs = env_kwargs or {}

        # Each logical trajectory owns independent game and RNG state. Keeping
        # these objects in one process avoids the large per-Ray-worker overhead.
        self.envs = [MineSweeper(**env_kwargs) for _ in range(self.num_processes)]
        self.env_copies = [env.copy() for env in self.envs]
        self.worker_rng_states = [
            np.random.RandomState(seed + worker_idx + 1).get_state()
            for worker_idx in range(self.num_processes)
        ]

    def _reset_one(self, env_idx, seed_for_reset):
        # MineSweeper currently uses NumPy's module-level RNG to choose the
        # initial revealed cell. Swap state around the call to emulate the
        # independent RNG owned by each Ray worker without coupling workers.
        process_rng_state = np.random.get_state()
        np.random.set_state(self.worker_rng_states[env_idx])
        try:
            result = self.envs[env_idx].reset(seed=seed_for_reset)
            self.worker_rng_states[env_idx] = np.random.get_state()
        finally:
            np.random.set_state(process_rng_state)
        return result

    def step(self, actions):
        assert len(actions) == self.num_processes
        results = []
        for env, action in zip(self.envs, actions):
            x, y = action
            results.append(env.step("L", x, y))
        return tuple([result[field] for result in results] for field in range(4))

    def reset(self):
        if self.is_train:
            seeds = self.seed_rng.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = self.seed_rng.randint(2**16, 2**32 - 1, size=self.env_num)
        seeds = np.repeat(seeds, self.group_n).tolist()

        results = [self._reset_one(i, seed) for i, seed in enumerate(seeds)]
        self.env_copies = [env.copy() for env in self.envs]
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def render(self, mode='table', env_idx=None):
        if mode != "board":
            raise ValueError(
                f"Invalid render mode: {mode}. Must be one of 'table', 'board', or 'coord'."
            )
        if env_idx is not None:
            return self.envs[env_idx].to_board_str_repr()
        return [env.to_board_str_repr() for env in self.envs]

    def restart(self):
        self.envs = [env.copy() for env in self.env_copies]
        obs_list = [env.to_board_str_repr() for env in self.envs]
        info_list = [{"won": False} for _ in self.envs]
        return obs_list, info_list

    def get_board_mine(self):
        return [env.board_mine for env in self.envs]

    def close(self):
        pass
        
        
def build_minesweeper_envs(
        seed=0,
        env_num=1,
        group_n=1,
        is_train=True,
        env_kwargs=None,
        execution_backend="ray"):
    if execution_backend == "ray":
        return MineSweeperMultiProcessEnv(seed, env_num, group_n, is_train, env_kwargs=env_kwargs)
    if execution_backend == "local":
        return MineSweeperLocalVectorEnv(seed, env_num, group_n, is_train, env_kwargs=env_kwargs)
    raise ValueError(
        f"Unsupported Minesweeper execution backend: {execution_backend}. "
        "Expected 'ray' or 'local'."
    )
