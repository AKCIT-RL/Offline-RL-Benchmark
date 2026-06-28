import torch
import numpy as np
import jax
import jax.numpy as jp
import gymnasium as gym
from collections.abc import Mapping
try:
    from flax.core import frozen_dict
except ImportError:
    frozen_dict = None

from mujoco_playground import wrapper, wrapper_torch
from mujoco_playground import registry

from .data_collector import DataCollector, CustomStepDataCallback


class WrapperJax():
    def __init__(
        self,
        env,
        env_cfg,
        num_actors,
        seed,
        command_type=None,
        device: str | torch.device | None = None,
    ):
        self.command_type = command_type
        self.env = env
        self.device = device
        self._batched_reset = jax.jit(jax.vmap(self.env.reset))
        self._batched_step = jax.jit(jax.vmap(self.env.step))
        self.rng = jax.random.PRNGKey(seed)

        # Curriculum support: Go2RoughCurriculum needs score-based respawning
        # between episodes (level up/down by terrain difficulty) to match the
        # training-time state distribution. Detected via the env API.
        base = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        self.curriculum = hasattr(base, "reset_to") and hasattr(base, "compute_next_tile")
        if self.curriculum:
            self._curr_base = base
            self._batched_reset_to = jax.jit(jax.vmap(base.reset_to))
            self._terrain_level = None
            self._terrain_col = None
            self.curriculum_num_rows = base.num_rows
            self.curriculum_num_cols = base.num_cols

        self.episode_length = env_cfg.episode_length
        self.num_actions = self.env.action_size
        if isinstance(self.env.observation_size, dict):
            self.num_obs = self.env.observation_size["state"]
        else:
            self.num_obs = self.env.observation_size
        self.num_envs = num_actors

        self.timesteps = 0
        # Optional integer tag injected into per-step infos to mark the source
        # policy of each transition (used by the medium-expert dataset).
        self.collection_source = None

    def _maybe_unfreeze(self, tree):
        if frozen_dict and isinstance(tree, frozen_dict.FrozenDict):
            return tree.unfreeze()
        if isinstance(tree, Mapping):
            return dict(tree)
        return tree

    def _tree_to_numpy(self, tree):
        if isinstance(tree, Mapping):
            return {k: self._tree_to_numpy(v) for k, v in tree.items()}
        if isinstance(tree, (list, tuple)):
            return type(tree)(self._tree_to_numpy(v) for v in tree)
        return np.asarray(tree)

    def _apply_command_override(self, env_state):
        if self.command_type is None or "command" not in env_state.info:
            return env_state

        commands = env_state.info["command"]
        zeros = jp.zeros_like(commands)

        if self.command_type == "fowardbackward":
            command = zeros.at[..., 0].set(commands[..., 0])
        elif self.command_type == "foward":
            command = zeros.at[..., 0].set(jp.abs(commands[..., 0]))
        elif self.command_type == "fowardfixed":
            command = zeros.at[..., 0].set(1.0)
        else:
            return env_state

        obs = self._maybe_unfreeze(env_state.obs)
        if isinstance(obs, dict):
            obs["state"] = obs["state"].at[..., -3:].set(command)
        else:
            obs = obs.at[..., -3:].set(command)

        info = self._maybe_unfreeze(env_state.info)
        info["command"] = command

        return env_state.replace(obs=obs, info=info)

    def reset(self):
        self.rng, reset_rng = jax.random.split(self.rng)
        reset_keys = jax.random.split(reset_rng, self.num_envs)
        if self.curriculum:
            self.env_state = self._curriculum_reset(reset_keys)
        else:
            self.env_state = self._batched_reset(reset_keys)
        self.env_state = self._apply_command_override(self.env_state)
        self.timesteps = 0
        obs_field = self.env_state.obs
        if isinstance(obs_field, dict):
            obs = np.asarray(obs_field["state"])
        else:
            obs = np.asarray(obs_field)
        return obs

    def _curriculum_reset(self, reset_keys):
        """Respawn each env on its curriculum tile (score-based level up/down).

        First episode: level 0 on a random column. Later episodes: promote/regress
        the level based on the finished episode's progress and whether it timed out,
        matching ``CurriculumAutoResetWrapper`` used during training.
        """
        if self._terrain_level is None:
            cols = jax.random.randint(
                reset_keys[0], (self.num_envs,), 0, self._curr_base.num_cols
            )
            levels = jp.zeros((self.num_envs,), dtype=jp.int32)
        else:
            max_progress = self.env_state.info["max_progress"]
            truncation = jp.asarray(
                [1.0 if self.timesteps >= self.episode_length else 0.0] * self.num_envs
            )
            levels, cols = self._curr_base.compute_next_tile(
                self._terrain_level, self._terrain_col, max_progress, truncation
            )
        self._terrain_level = levels
        self._terrain_col = cols
        return self._batched_reset_to(reset_keys, levels, cols)


    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = jp.asarray(action)

        self.env_state = self._batched_step(self.env_state, action)
        self.env_state = self._apply_command_override(self.env_state)
        self.timesteps += 1
        obs_field = self.env_state.obs
        if isinstance(obs_field, dict):
            obs = np.asarray(obs_field["state"])
        else:
            obs = np.asarray(obs_field)
        rew = np.asarray(self.env_state.reward)
        done = np.asarray(self.env_state.done)
        truncated = np.asarray([self.timesteps >= self.episode_length for _ in range(self.num_envs)])
        info = self._tree_to_numpy(self.env_state.info)
        if self.collection_source is not None:
            info["source"] = np.full(self.num_envs, self.collection_source, dtype=np.int64)
        return obs, rew, done, truncated, info

def wrapper_fn(
    env_name: str,
    num_actors: int,
    seed: int,
    action_repeat: int,
    device: str,
    command_type: str,
):
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)

    env_wrapped = WrapperJax(
        env,
        env_cfg,
        num_actors,
        seed,
        command_type=command_type,
        device=device,
    )

    return env_wrapped


def wrapper_collector(
    env_name: str, num_envs: int, seed: int, action_repeat: int, device: str, command_type: str = None
):
    env = wrapper_fn(env_name, num_envs, seed, action_repeat, device, command_type)
    obs_shape = (env.num_obs,) if type(env.num_obs) == int else tuple(env.num_obs)
    return DataCollector(
        env,
        step_data_callback=CustomStepDataCallback,
        # gym.spaces.Box is serialized natively by Minari, so loading a dataset
        # does not require importing this package first (unlike a custom space).
        observation_space=gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        ),
        action_space=gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(env.num_actions,), dtype=np.float32
        ),
        record_infos=True
    )
