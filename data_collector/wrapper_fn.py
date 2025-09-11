import torch
import numpy as np
import jax
import jax.numpy as jp

from mujoco_playground import wrapper, wrapper_torch
from mujoco_playground import registry

from .data_collector import DataCollector, CustomStepDataCallback
from .space import NumpySpace


class WrapperTorch(wrapper_torch.RSLRLBraxWrapper):
    def __init__(
      self,
      env,
      num_actors,
      seed,
      episode_length,
      action_repeat,
      randomization_fn=None,
      render_callback=None,
      device_rank=None,
      command_type=None,
  ):
        super().__init__(env, num_actors, seed, episode_length, action_repeat, randomization_fn, render_callback, device_rank)
        self.command_type = command_type

    def reset(self):
        self.key, key_reset = jax.random.split(self.key)
        _key_reset = jax.random.split(key_reset, self.batch_size)
        self.env_state = self.reset_fn(_key_reset)

        if self.command_type == "fowardbackward":
            command = jp.concatenate([
                self.env_state.info["command"][:, [0]],  # shape (batch, 1)
                jp.zeros((self.env_state.info["command"].shape[0], 2), dtype=self.env_state.info["command"].dtype)
            ], axis=1)
            self.env_state.info["command"] = command
        elif self.command_type == "foward":
            command = jp.concatenate([
                jp.abs(self.env_state.info["command"][:, [0]]),  # shape (batch, 1)
                jp.zeros((self.env_state.info["command"].shape[0], 2), dtype=self.env_state.info["command"].dtype)
            ], axis=1)
            self.env_state.info["command"] = command
        elif self.command_type == "fowardfixed":
            command = jp.array([1.5, 0, 0])
            self.env_state.info["command"] = command

        if self.asymmetric_obs:
            obs = wrapper_torch._jax_to_torch(self.env_state.obs["state"])
        # critic_obs = jax_to_torch(self.env_state.obs["privileged_state"])
        else:
            obs = wrapper_torch._jax_to_torch(self.env_state.obs)
        return obs

    def reset_with_critic_obs(self):
        self.key, key_reset = jax.random.split(self.key)
        _key_reset = jax.random.split(key_reset, self.batch_size)
        self.env_state = self.reset_fn(_key_reset)
        obs = wrapper_torch._jax_to_torch(self.env_state.obs["state"])
        critic_obs = wrapper_torch._jax_to_torch(self.env_state.obs["privileged_state"])
        return obs, critic_obs




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
    randomizer = registry.get_domain_randomizer(env_name)

    env_wrapped = WrapperTorch(
        env,
        num_actors,
        seed,
        env_cfg.episode_length,
        action_repeat,
        randomization_fn=randomizer,
        device_rank=int(device.split(":")[-1]) if "cuda:" in device else 0,
        command_type=command_type
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
    )
