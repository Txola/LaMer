import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

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
    ):
        # Evaluation workers receive a fixed game pool whose size equals their
        # internal batch. Every game is therefore loaded exactly once, while a
        # small number of Ray processes can host the complete evaluation set.
        if game_files is not None:
            base_env.game_files = list(game_files)
            base_env.num_games = len(game_files)
            batch_size = len(game_files)
        self.batch_size = batch_size
        self.env = base_env.init_env(batch_size=batch_size)
        self.env.seed(seed)
        if repeat_game_within_batch:
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

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, is_train, env_kwargs)
