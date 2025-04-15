import pyrallis
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, SupportsFloat, Type
import functools
from etils import epath

from tqdm import tqdm
import torch
import minari
import jax
import numpy as np
import mujoco
from jax import numpy as jp
import mediapy as media

from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground.config import locomotion_params

from mujoco_playground import registry
from mujoco_playground import wrapper, wrapper_torch

from data_collector import wrapper_collector


@dataclass
class Config:
    env_name: str = None  # The name of the environment to collect data from.
    num_samples: int = (
        10**6
    )  # The number of samples refers to the number of environment transitions recorded in the dataset.
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    num_envs: int = 1
    action_repeat: int = 1
    model_checkpoint: Optional[str] = None
    dataset_version: int = 0
    difficulty: str = "random"
    algorithm_name: str = "random_policy"
    author: str = "Luana Martins"
    author_email: str = "luanagbmartins@gmail.com"
    code_permalink: str = "https://github.com/AKCIT-RL/mujoco_playground"
    description: Optional[str] = None


def get_checkpoint_path(model_checkpoint: str):
    ckpt_path = str(epath.Path(model_checkpoint).resolve())
    FINETUNE_PATH = epath.Path(ckpt_path)
    latest_ckpts = list(FINETUNE_PATH.glob("*"))
    latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
    latest_ckpts.sort(key=lambda x: int(x.name))
    return latest_ckpts


def get_inference_fn(restore_checkpoint_path: str, env_name: str):
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    ppo_training_params = dict(ppo_params)
    ppo_training_params["num_timesteps"] = 0

    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.network_factory
        )

    randomizer = registry.get_domain_randomizer(env_name)

    train_fn = functools.partial(
        ppo.train,
        **dict(ppo_training_params),
        network_factory=network_factory,
        randomization_fn=randomizer,
    )

    make_inference_fn, params, metrics = train_fn(
        environment=registry.load(env_name),
        eval_env=registry.load(env_name),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        restore_checkpoint_path=restore_checkpoint_path,  # restore from the checkpoint!
        seed=1,
    )

    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    return jit_inference_fn


@pyrallis.wrap()
def main(config: Config):
    print(config)

    env = wrapper_collector(
        config.env_name,
        config.num_envs,
        config.seed,
        config.action_repeat,
        config.device,
    )

    if config.difficulty == "random":
        env.reset()
        timesteps = 0
        pbar = tqdm(total=config.num_samples, desc="Collecting samples")
        while timesteps < config.num_samples:
            terminated = torch.zeros(config.num_envs, device=config.device).bool()
            truncated = torch.zeros(config.num_envs, device=config.device).bool()
            while not terminated.all():
                action = env.action_sample(config.num_envs)
                _, _, terminated, truncated, _, _ = env.step(action)
                pbar.update(env.get_timesteps() - timesteps)
                timesteps = env.get_timesteps()
            env.reset()
        pbar.close()

    elif config.difficulty == "medium":
        restore_checkpoint_path = get_checkpoint_path(config.model_checkpoint)
        samples_per_ckpt = config.num_samples // len(restore_checkpoint_path)
        for ckpt in restore_checkpoint_path:
            inference_fn = get_inference_fn(ckpt, config.env_name)

            env = wrapper_collector(
                config.env_name,
                config.num_envs,
                config.seed,
                config.action_repeat,
                config.device,
            )

            rng = jax.random.PRNGKey(0)

            obs, info = env.reset()
            env_state = info["env_state"].obs

            timesteps = 0
            pbar = tqdm(total=samples_per_ckpt, desc="Collecting samples")
            while timesteps < samples_per_ckpt:
                terminated = torch.zeros(config.num_envs, device=config.device).bool()
                while not terminated.all():
                    action, _ = inference_fn(info["env_state"].obs, rng)
                    _, _, terminated, _, _, info = env.step(
                        wrapper_torch._jax_to_torch(action)
                    )
                    env_state = info["env_state"].obs
                    pbar.update(env.get_timesteps() - timesteps)
                    timesteps = env.get_timesteps()
                _, info = env.reset()
                env_state = info["env_state"].obs
            pbar.close()
    elif config.difficulty == "expert":
        restore_checkpoint_path = get_checkpoint_path(config.model_checkpoint)
        inference_fn = get_inference_fn(restore_checkpoint_path[-1], config.env_name)

        env = wrapper_collector(
            config.env_name,
            config.num_envs,
            config.seed,
            config.action_repeat,
            config.device,
        )

        rng = jax.random.PRNGKey(0)

        obs, info = env.reset()
        env_state = info["env_state"].obs

        timesteps = 0
        pbar = tqdm(total=config.num_samples, desc="Collecting samples")
        while timesteps < config.num_samples:
            terminated = torch.zeros(config.num_envs, device=config.device).bool()
            while not terminated.all():
                action, _ = inference_fn(env_state.obs, rng)
                _, _, terminated, _, _, info = env.step(
                    wrapper_torch._jax_to_torch(action)
                )
                env_state = info["env_state"].obs
                pbar.update(env.get_timesteps() - timesteps)
                timesteps = env.get_timesteps()
            _, info = env.reset()
            env_state = info["env_state"].obs
        pbar.close()

    else:
        raise ValueError(f"Difficulty {config.difficulty} not supported")

    dataset = env.create_dataset(
        dataset_id=f"playground/{config.env_name}-{config.difficulty}-v{config.dataset_version}",
        algorithm_name=config.algorithm_name,
        author=config.author,
        author_email=config.author_email,
        code_permalink=config.code_permalink,
        description=config.description,
    )


if __name__ == "__main__":
    main()
