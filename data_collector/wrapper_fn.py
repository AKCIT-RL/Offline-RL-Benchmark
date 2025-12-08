import torch
import numpy as np
import jax
import jax.numpy as jp
from collections.abc import Mapping
try:
    from flax.core import frozen_dict
except ImportError:
    frozen_dict = None

from mujoco_playground import wrapper, wrapper_torch
from mujoco_playground import registry

from .data_collector import DataCollector, CustomStepDataCallback
from .space import NumpySpace


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

        self.episode_length = env_cfg.episode_length
        self.num_actions = self.env.action_size
        if isinstance(self.env.observation_size, dict):
            self.num_obs = self.env.observation_size["state"]
        else:
            self.num_obs = self.env.observation_size
        self.num_envs = num_actors

        self.timesteps = 0

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
        self.env_state = self._batched_reset(reset_keys)
        self.env_state = self._apply_command_override(self.env_state)
        self.timesteps = 0
        obs_field = self.env_state.obs
        if isinstance(obs_field, dict):
            obs = np.asarray(obs_field["state"])
        else:
            obs = np.asarray(obs_field)
        return obs

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
    return DataCollector(
        env,
        step_data_callback=CustomStepDataCallback,
        observation_space=NumpySpace(
            shape=(env.num_obs,) if type(env.num_obs) == int else env.num_obs,
            dtype=np.float32,
        ),
        action_space=NumpySpace(shape=(env.num_actions,), dtype=np.float32),
        record_infos=True
    )
