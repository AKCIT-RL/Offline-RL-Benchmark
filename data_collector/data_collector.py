# ------------------- DATA COLLECTOR -------------------
# This file is a modified version of the DataCollector class from Minari
# The modifications include:
# - Support for mujoco playground environments
# - Custom step data callback for tensor conversion
#
# https://github.com/minari-dataset/minari/blob/main/minari/data_collector/data_collector.py
# https://github.com/AKCIT-RL/mujoco_playground

import os
import shutil
import tempfile
import warnings
import secrets
from typing import Any, Callable, Dict, Optional, SupportsFloat, Type
from concurrent.futures import ThreadPoolExecutor
import threading
import dataclasses

import torch
import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.envs.registration import EnvSpec

from minari.dataset.step_data import StepData
from minari.data_collector.episode_buffer import EpisodeBuffer
from minari.data_collector.callbacks import EpisodeMetadataCallback, StepDataCallback
from minari.dataset.minari_storage import MinariStorage
from minari.dataset.minari_dataset import MinariDataset, parse_dataset_id
from minari.namespace import create_namespace, list_local_namespaces
from minari.utils import _generate_dataset_metadata, _generate_dataset_path

from mujoco_playground import wrapper_torch

AUTOSEED_BIT_SIZE = 64


class CustomStepDataCallback(StepDataCallback):
    def __call__(
        self,
        env: gym.Env,
        obs: Any,
        info: Dict[str, Any],
        action: Optional[Any] = None,
        rew: Optional[Any] = None,
        terminated: Optional[bool] = None,
        truncated: Optional[bool] = None,
    ) -> StepData:

        _info = {
            k: (np.squeeze(v.cpu().numpy()) if isinstance(v, torch.Tensor) else v)
            for k, v in info.items()
        }
        step_data: StepData = {
            "action": (
                np.squeeze(action.cpu().numpy())
                if isinstance(action, torch.Tensor)
                else action
            ),
            "observation": (
                np.squeeze(obs.cpu().numpy()) if isinstance(obs, torch.Tensor) else obs
            ),
            "reward": (
                np.squeeze(rew.cpu().numpy()) if isinstance(rew, torch.Tensor) else rew
            ),
            "terminated": (
                np.squeeze(terminated.cpu().numpy()).astype(bool)
                if isinstance(terminated, torch.Tensor)
                else terminated
            ),
            "truncated": (
                np.squeeze(truncated.cpu().numpy()).astype(bool)
                if isinstance(truncated, torch.Tensor)
                else truncated
            ),
            "info": _info,
        }

        return step_data


class DataCollector:
    r"""Gymnasium environment wrapper that collects step data.

    This wrapper is meant to work as a temporary buffer of the environment data before creating a Minari dataset. The creation of the buffers
    that will be convert to a Minari dataset is agnostic to the user:

    .. code::

        import minari
        import gymnasium as gym

        env = minari.DataCollector(gym.make('EnvID'))

        env.reset()

        for _ in range(num_steps):
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step()

            if terminated or truncated:
                env.reset()

        dataset = env.create_dataset(dataset_id="env_name/dataset_name-v(version)", **kwargs)

    Some of the characteristics of this wrapper:

        * The step data is stored per episode in dictionaries. This dictionaries are then stored in-memory in a global list buffer. The
          episode dictionaries contain items with list buffers as values for the main episode step datasets `observations`, `actions`,
          `terminations`, and `truncations`, the `infos` key can be a list or another nested dictionary with extra datasets.

        * A new episode dictionary buffer is created if the env.step(action) call returns `truncated` or `terminated`, or if the environment calls
          env.reset(). If calling reset and the previous episode was not `truncated` or `terminated`, this will automatically be `truncated`.

    """

    def __init__(
        self,
        env,
        step_data_callback: Type[StepDataCallback] = StepDataCallback,
        episode_metadata_callback: Type[
            EpisodeMetadataCallback
        ] = EpisodeMetadataCallback,
        record_infos: bool = False,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        data_format: Optional[str] = None,
    ):
        """Initialize the data collector attributes and create the temporary directory for caching.

        Args:
            env (gym.Env): Gymnasium environment
            step_data_callback (type[StepDataCallback], optional): Callback class to edit/update step databefore storing to buffer. Defaults to StepDataCallback.
            episode_metadata_callback (type[EpisodeMetadataCallback], optional): Callback class to add custom metadata to episode group in HDF5 file. Defaults to EpisodeMetadataCallback.
            record_infos (bool, optional): If True record the info return key of each step. Defaults to False.
            observation_space (gym.Space): Observation space of the dataset. The default value is the environment observation space.
            action_space (gym.Space): Action space of the dataset. The default value is the environment action space.
            data_format (str, optional): Data format to store the data in the Minari dataset. If None (defaults), it will use the default format of MinariStorage.
        """
        self.env = env
        self._step_data_callback = step_data_callback()
        self._episode_metadata_callback = episode_metadata_callback()

        self.datasets_path = os.environ.get("MINARI_DATASETS_PATH")
        if self.datasets_path is None:
            self.datasets_path = os.path.join(
                os.path.expanduser("~"), ".minari", "datasets"
            )
        if not os.path.exists(self.datasets_path):
            os.makedirs(self.datasets_path)
        self.data_format = data_format

        if observation_space is None:
            observation_space = env.observation_space
        self._observation_space = observation_space
        if action_space is None:
            action_space = env.action_space
        self._action_space = action_space

        self._record_infos = record_infos
        # Optional minimum episodic return required to persist an episode.
        # When set, completed episodes whose summed reward is below this value
        # are dropped at store time (used for the expert P90 filter).
        self._episode_return_threshold: Optional[float] = None
        self._buffer: Optional[EpisodeBuffer] = None
        self._episode_id = 0
        self._list_terminated = np.array([0 for _ in range(self.env.num_envs)])
        self._last_observation = [None for _ in range(self.env.num_envs)]
        self._last_info = [None for _ in range(self.env.num_envs)]
        self._timesteps = 0
        self._timesteps_lock = threading.Lock()
        self._storage_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=self.env.num_envs)
        self._reset_storage()

    def _reset_storage(self):
        self._episode_id = 0
        self._tmp_dir = tempfile.TemporaryDirectory(dir=self.datasets_path)
        data_format_kwarg = (
            {"data_format": self.data_format} if self.data_format is not None else {}
        )
        self._storage = MinariStorage.new(
            self._tmp_dir.name,
            observation_space=self._observation_space,
            action_space=self._action_space,
            **data_format_kwarg,
        )

    def _process_env_step(self, env_idx, obs, action, rew, terminated, truncated, info):
        """Process step data for a single environment (parallelizable)."""
        if self._list_terminated[env_idx] == 0:
            env_obs = (
                obs[env_idx] if isinstance(obs, (np.ndarray, torch.Tensor)) else obs
            )
            env_info = {
                k: v[env_idx] if isinstance(v, (np.ndarray, torch.Tensor)) else v
                for k, v in info.items()
            }
            env_action = (
                action[env_idx]
                if isinstance(action, (np.ndarray, torch.Tensor))
                else action
            )
            env_rew = (
                rew[env_idx] if isinstance(rew, (np.ndarray, torch.Tensor)) else rew
            )
            env_terminated = (
                terminated[env_idx]
                if isinstance(terminated, (np.ndarray, torch.Tensor))
                else terminated
            )
            env_truncated = (
                truncated[env_idx]
                if isinstance(truncated, (np.ndarray, torch.Tensor))
                else truncated
            )

            step_data = self._step_data_callback(
                env=self.env,
                obs=env_obs,
                info=env_info,
                action=env_action,
                rew=env_rew,
                terminated=env_terminated,
                truncated=env_truncated,
            )

            # Space validation warnings
            if not self._storage.observation_space.contains(
                step_data["observation"]
            ):
                warnings.warn(
                    f"Observation for env {env_idx} is not in observation space.\n"
                    f"Observation: {step_data['observation']}\nObservation type: {type(step_data['observation'])}\n"
                    f"Observation shape: {step_data['observation'].shape}\nSpace: {self._storage.observation_space}"
                )
            if not self._storage.action_space.contains(step_data["action"]):
                warnings.warn(
                    f"Action for env {env_idx} is not in action space.\n"
                    f"Action: {step_data['action']}\nSpace: {self._storage.action_space}",
                )

            if not self._record_infos:
                step_data["info"] = {}

            # Update buffer with new step data
            self._buffer[env_idx] = self._buffer[env_idx].add_step_data(step_data)
            
            # Thread-safe timestep increment
            with self._timesteps_lock:
                self._timesteps += 1
            
            # Handle episode termination
            if step_data["terminated"] or step_data["truncated"]:
                self._list_terminated[env_idx] = 1
                self._last_observation[env_idx] = step_data["observation"]
                self._last_info[env_idx] = step_data["info"]

    def _prepare_new_buffer(self, env_idx, new_episode_id):
        """Prepare a new episode buffer for a single environment (parallelizable)."""
        last_info = self._last_info[env_idx]
        return EpisodeBuffer(
            id=new_episode_id + env_idx,
            observations=self._last_observation[env_idx],
            infos=last_info if (self._record_infos and last_info) else None,
        )

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Gymnasium step method supporting batched environments."""
        obs, rew, terminated, truncated, info = self.env.step(action)
        action = np.asarray([action]) if len(action.shape) < 2  else np.asarray(action)
 
        # Parallel processing of step data for each environment
        futures = []
        for env_idx in range(self.env.num_envs):
            future = self._executor.submit(
                self._process_env_step,
                env_idx, obs, action, rew, terminated, truncated, info
            )
            futures.append(future)
        
        # Wait for all parallel tasks to complete
        for future in futures:
            future.result()

        # Process step data for each environment
        if self._list_terminated.sum() == len(self._list_terminated):
            # Prepare new episode ID
            new_episode_id = self._episode_id + self.env.num_envs
            
            # Parallel preparation of new buffers
            futures = []
            for env_idx in range(self.env.num_envs):
                future = self._executor.submit(
                    self._prepare_new_buffer, env_idx, new_episode_id
                )
                futures.append(future)
            
            # Collect new buffers in order
            new_buffers = []
            for future in futures:
                new_buffers.append(future.result())
            
            # Sequential storage updates (required for MinariStorage)
            # Episodes dropped by the return-threshold gate must not leave holes
            # in the id sequence, so reassign ids from total_episodes on store.
            for env_idx in range(self.env.num_envs):
                if self._passes_return_threshold(self._buffer[env_idx]):
                    buf = dataclasses.replace(
                        self._buffer[env_idx], id=self._storage.total_episodes
                    )
                    self._storage.update_episodes([buf])
            
            # Update buffers and episode ID
            self._buffer = new_buffers
            self._episode_id = new_episode_id
                
        return (
            obs,
            rew,
            self._list_terminated,
            truncated,
            info,
            {"env_state": self.env.env_state},
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Gymnasium environment reset."""
        self._flush_to_storage()

        autoseed_enabled = (not options) or options.get("minari_autoseed", True)
        if seed is None and autoseed_enabled:
            seed = secrets.randbits(AUTOSEED_BIT_SIZE)

        self._list_terminated = np.array([0 for _ in range(self.env.num_envs)])

        self.reset_timesteps()

        obs = self.env.reset()

        # Initialize buffer list for each environment
        self._buffer = []
        for i in range(self.env.num_envs):
            env_obs = obs[i] if isinstance(obs, (np.ndarray, torch.Tensor)) else obs
            step_data = self._step_data_callback(env=self.env, obs=env_obs, info={})
            self._buffer.append(
                EpisodeBuffer(
                    id=self._episode_id + i,
                    # seed=seed,
                    options=options,
                    observations=step_data["observation"],
                    # Pass None if infos is empty to let the first step establish the structure
                    infos=step_data["info"] if (self._record_infos and step_data["info"]) else None,
                )
            )
        return obs, {"env_state": self.env.env_state}

    def add_to_dataset(self, dataset: MinariDataset):
        """Add extra data to Minari dataset from collector environment buffers (DataCollector).

        Args:
            dataset (MinariDataset): Dataset to add the data
        """
        self._flush_to_storage()

        first_id = dataset.storage.total_episodes
        dataset.storage.update_from_storage(self._storage)
        if dataset.episode_indices is not None:
            new_ids = first_id + np.arange(self._storage.total_episodes)
            dataset.episode_indices = np.append(dataset.episode_indices, new_ids)

        self._reset_storage()

    def create_dataset(
        self,
        dataset_id: str,
        eval_env: Optional[str | gym.Env | EnvSpec] = None,
        algorithm_name: Optional[str] = None,
        author: Optional[str | set] = None,
        author_email: Optional[str | set] = None,
        code_permalink: Optional[str] = None,
        ref_min_score: Optional[float] = None,
        ref_max_score: Optional[float] = None,
        expert_policy: Optional[Callable[[ObsType], ActType]] = None,
        num_episodes_average_score: int = 100,
        description: Optional[str] = None,
        requirements: Optional[list] = None,
    ):
        """Create a Minari dataset using the data collected from stepping with a Gymnasium environment wrapped with a `DataCollector` Minari wrapper.

        The ``dataset_id`` parameter corresponds to the name of the dataset, with the syntax as follows:
        ``(namespace/)(env_name/)dataset_name(-v[version])`` where ``env_name`` identifies the name of the environment used to generate the dataset. The `namespace` is optional.
        This ``dataset_id`` is used to load the Minari datasets with :meth:`minari.load_dataset`.

        Args:
            dataset_id (str): name id to identify Minari dataset
            eval_env (str | gym.Env | EnvSpec, optional): Gymnasium environment(gym.Env)/environment id(str)/environment spec(EnvSpec) to use for evaluation with the dataset. After loading the dataset, the environment can be recovered as follows: `MinariDataset.recover_environment(eval_env=True).
                                                    If None the `env` used to collect the buffer data should be used for evaluation.
            algorithm_name (str, optional): name of the algorithm used to collect the data. Defaults to None.
            author (str | set, optional): name of the author(s) that generated the dataset. Defaults to None.
            author_email (str | set, optional): email(s) of the author(s) that generated the dataset. Defaults to None.
            code_permalink (str, optional): link to relevant code used to generate the dataset. Defaults to None.
            ref_min_score(float, optional): minimum reference score from the average returns of a random policy. This value is later used to normalize a score with :meth:`minari.get_normalized_score`. If default None the value will be estimated with a default random policy.
            ref_max_score (float, optional): maximum reference score from the average returns of a hypothetical expert policy. This value is used in :meth:`minari.get_normalized_score`. Default None.
            expert_policy (Callable[[ObsType], ActType], optional): policy to compute `ref_max_score` by averaging the returns over a number of episodes equal to  `num_episodes_average_score`.
                                                                            `ref_max_score` and `expert_policy` can't be passed at the same time. Default to None
            num_episodes_average_score (int): number of episodes to average over the returns to compute `ref_min_score` and `ref_max_score`. Default to 100.
            description (str, optional): description of the dataset being created. Defaults to None.
            requirements (list of str, optional): list of requirements in pip-style to load the environment and reproduce the dataset. For example, `mujoco>=3.1.0,<3.2.0`, which indicate the supported version range for mujoco package. Defaults to None.

        Returns:
            MinariDataset
        """
        namespace = parse_dataset_id(dataset_id)[0]

        if namespace is not None and namespace not in list_local_namespaces():
            create_namespace(namespace)

        dataset_path = _generate_dataset_path(dataset_id)
        metadata: Dict[str, Any] = _generate_dataset_metadata(
            dataset_id,
            None,
            eval_env,
            algorithm_name,
            author,
            author_email,
            code_permalink,
            ref_min_score,
            ref_max_score,
            expert_policy,
            num_episodes_average_score,
            description,
            requirements,
        )

        self._save_to_disk(dataset_path, metadata)
        return MinariDataset(dataset_path)

    def _passes_return_threshold(self, buffer) -> bool:
        """Whether a completed episode buffer meets the minimum return to be stored."""
        if self._episode_return_threshold is None:
            return True
        if len(buffer) == 0:
            return False
        return float(np.sum(buffer.rewards)) >= self._episode_return_threshold

    def _flush_to_storage(self):
        """Flush all buffers to storage."""
        if self._buffer is not None:
            for buffer in self._buffer:
                if len(buffer) > 0:
                    if not buffer.terminations[-1]:
                        buffer.truncations[-1] = True
                    if self._passes_return_threshold(buffer):
                        buf = dataclasses.replace(
                            buffer, id=self._storage.total_episodes
                        )
                        self._storage.update_episodes([buf])
        self._buffer = None

    def _save_to_disk(
        self, path: str | os.PathLike, dataset_metadata: Dict[str, Any] = {}
    ):
        """Save all in-memory buffer data and move temporary files to a permanent location in disk.

        Args:
            path (str): path to store the dataset, e.g.: '/home/foo/datasets/data'
            dataset_metadata (Dict, optional): additional metadata to add to the dataset file. Defaults to {}.
        """
        self._flush_to_storage()

        assert (
            "observation_space" not in dataset_metadata.keys()
        ), "'observation_space' is not allowed as an optional key."
        assert (
            "action_space" not in dataset_metadata.keys()
        ), "'action_space' is not allowed as an optional key."
        assert (
            "env_spec" not in dataset_metadata.keys()
        ), "'env_spec' is not allowed as an optional key."
        self._storage.update_metadata(dataset_metadata)

        episode_metadata = self._storage.apply(self._episode_metadata_callback)
        self._storage.update_episode_metadata(episode_metadata)

        files = os.listdir(self._storage.data_path)
        for file in files:
            shutil.move(
                os.path.join(self._storage.data_path, file),
                os.path.join(path, file),
            )

        self._reset_storage()

    def close(self):
        """Close the DataCollector.

        Clear buffer and close temporary directory.
        """
        super().close()
        self._buffer = None
        self._executor.shutdown(wait=True)
        shutil.rmtree(self._tmp_dir.name)

    def action_sample(self, num_envs: int = 1):
        return torch.randn((num_envs, self.env.num_actions), device=self.env.device)

    def get_timesteps(self):
        return self._timesteps

    def reset_timesteps(self):
        self._timesteps = 0

    def set_episode_return_threshold(self, threshold: Optional[float]):
        """Set (or clear with None) the minimum episodic return for an episode to be stored."""
        self._episode_return_threshold = threshold

    def get_stored_steps(self) -> int:
        """Total number of transitions already flushed to storage."""
        return self._storage.total_steps
