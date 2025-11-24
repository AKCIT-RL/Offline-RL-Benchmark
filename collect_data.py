from string import printable
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
import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


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
    model_checkpoint: Optional[list[str]] = None
    dataset_version: int = 0
    difficulty: str = "random"
    algorithm_name: str = "random_policy"
    author: str = "Luana Martins"
    author_email: str = "luanagbmartins@gmail.com"
    code_permalink: str = "https://github.com/AKCIT-RL/mujoco_playground"
    description: Optional[str] = None
    minari_dataset_path: str = "/home/luana/OfflineRL/CORL/datasets"
    command_type: str = None


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
        nf = ppo_params.network_factory
        nf["value_obs_key"] = "state"
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **nf
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
        restore_checkpoint_path=restore_checkpoint_path,  # restore from the checkpoint!
        seed=1,
    )

    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    return jit_inference_fn


@pyrallis.wrap()
def main(config: Config):
    print(config)

    os.environ["MINARI_DATASETS_PATH"] = config.minari_dataset_path

    env = wrapper_collector(
        config.env_name,
        config.num_envs,
        config.seed,
        config.action_repeat,
        config.device,
        config.command_type,
    )
    rng = jax.random.PRNGKey(0)

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
        samples_per_ckpt_model = config.num_samples // len(config.model_checkpoint)
        print(f"Collecting {samples_per_ckpt_model} samples per checkpoint.\nTotal checkpoints: {len(config.model_checkpoint)}")

        for i, model_checkpoint in enumerate(config.model_checkpoint):
            print("-"*100)
            print(f"Checkpoint {i+1} of {len(config.model_checkpoint)}")
            print(f"Collecting samples from checkpoint: {model_checkpoint}")

            restore_checkpoint_path = get_checkpoint_path(model_checkpoint)
            samples_per_ckpt = samples_per_ckpt_model // len(restore_checkpoint_path)
            print(f"Collecting {samples_per_ckpt} samples per checkpoint.")
            print(f"Total checkpoints: {len(restore_checkpoint_path)}")
            for ckpt in restore_checkpoint_path:
                print(f"Collecting samples from checkpoint: {ckpt}")
                inference_fn = get_inference_fn(ckpt, config.env_name)
                env.reset_timesteps()
                _, info = env.reset()
                env_state = info["env_state"].obs

                timesteps = 0
                pbar = tqdm(total=samples_per_ckpt, desc="Collecting samples")
                while timesteps < samples_per_ckpt:
                    terminated = torch.zeros(config.num_envs, device=config.device).bool()
                    while not terminated.all():
                        action, _ = inference_fn(env_state, rng)
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
        samples_per_ckpt = config.num_samples // len(config.model_checkpoint)
        print(f"Collecting {samples_per_ckpt} samples per checkpoint.\nTotal checkpoints: {len(config.model_checkpoint)}")
        
        for i, ckpt in enumerate(config.model_checkpoint):
            print("-"*100)
            print(f"Checkpoint {i+1} of {len(config.model_checkpoint)}")
            print(f"Collecting samples from checkpoint: {ckpt}")
            restore_checkpoint_path = get_checkpoint_path(ckpt)
            inference_fn = get_inference_fn(restore_checkpoint_path[-1], config.env_name)
            env.reset_timesteps()
            _, info = env.reset()
            env_state = info["env_state"].obs
            timesteps = 0
            pbar = tqdm(total=samples_per_ckpt, desc="Collecting samples")
            while timesteps < samples_per_ckpt:
                terminated = torch.zeros(config.num_envs, device=config.device).bool()
                while not terminated.all():
                    action, _ = inference_fn(env_state, rng)
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
    
    if config.command_type is not None:
        dataset_name = f"{config.env_name}-{config.command_type}-{config.difficulty}-v{config.dataset_version}"
    else:
        dataset_name = f"{config.env_name}-{config.difficulty}-v{config.dataset_version}"

    print(
        f"Creating dataset {dataset_name}"
    )
    dataset = env.create_dataset(
        dataset_id=f"playground/{dataset_name}",
        algorithm_name=config.algorithm_name,
        author=config.author,
        author_email=config.author_email,
        code_permalink=config.code_permalink,
        description=config.description,
    )
    print(
        f"Dataset created with {dataset.total_steps} timesteps and {dataset.total_episodes} episodes"
    )


if __name__ == "__main__":
    main()
