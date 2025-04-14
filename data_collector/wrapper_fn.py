import torch
import numpy as np

from mujoco_playground import wrapper, wrapper_torch
from mujoco_playground import registry

from .data_collector import DataCollector, CustomStepDataCallback
from .space import NumpySpace


def wrapper_fn(
    env_name: str,
    num_actors: int,
    seed: int,
    action_repeat: int,
    device: str,
):
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)
    randomizer = registry.get_domain_randomizer(env_name)
    render_trajectory = []

    def render_callback(_, state):
        render_trajectory.append(state)

    env_wrapped = wrapper_torch.RSLRLBraxWrapper(
        env,
        num_actors,
        seed,
        env_cfg.episode_length,
        action_repeat,
        render_callback=render_callback,
        randomization_fn=randomizer,
        device_rank=int(device.split(":")[-1]) if "cuda" in device else 0,
    )

    return env_wrapped


def wrapper_collector(
    env_name: str, num_envs: int, seed: int, action_repeat: int, device: str
):
    env = wrapper_fn(env_name, num_envs, seed, action_repeat, device)
    return DataCollector(
        env,
        step_data_callback=CustomStepDataCallback,
        observation_space=NumpySpace(shape=env.num_obs, dtype=np.float32),
        action_space=NumpySpace(shape=(env.num_actions,), dtype=np.float32),
    )
