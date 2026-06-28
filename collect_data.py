import pyrallis
import re
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, SupportsFloat, Type
import functools
from etils import epath
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from mujoco_playground.config import locomotion_params, manipulation_params

from mujoco_playground import registry
from mujoco_playground import wrapper, wrapper_torch

from data_collector import wrapper_collector
import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# Difficulty (dataset type) -> collection semantics. See collect_data_refactor_plan.md (Item 3).
VALID_DIFFICULTIES = ("expert", "medium", "medium-replay", "medium-expert")

# Integer codes recorded in info["source"] for the medium-expert dataset.
SOURCE_EXPERT = 0
SOURCE_MEDIUM = 1

# Registry env name -> benchmark task-id slug (see collect_data_refactor_plan.md, section 0).
# Go2JoystickFlatTerrain is resolved separately because the same env serves two
# tasks depending on the joystick command (fixed vs variable).
ENV_TO_TASK_ID = {
    "H1JoystickGaitTracking": "h1-gait-tracking",
    "G1JoystickFlatTerrain": "g1-joystick-direction",
    "Go2Footstand": "go2-footstand",
    "Go2Handstand": "go2-handstand",
    "Go2Getup": "go2-getup",
    "Go2GetupWalk": "go2-getup-walk",
    "H1Getup": "h1-getup-stand",
    "Go2RoughCurriculum": "go2-rough-terrain",
    "Go2PushRecovery": "go2-push-recovery",
}


def _camel_to_kebab(name: str) -> str:
    """Fallback slug for envs not in ENV_TO_TASK_ID (e.g. new envs being trained)."""
    return re.sub(r"(?<!^)(?=[A-Z0-9])", "-", name).lower()


def resolve_task_id(env_name: str, command_type: Optional[str]) -> str:
    """Map a registry env name (+ joystick command) to the benchmark task-id slug.

    Go2JoystickFlatTerrain backs two benchmark tasks: a fixed-command Tier 1 task
    (``go2-flat-forward``) and the variable-command Tier 2 task
    (``go2-joystick-direction``). Any other command override is kept distinguishable
    so datasets never silently collide.
    """
    if env_name == "Go2JoystickFlatTerrain":
        if command_type == "fowardfixed":
            return "go2-flat-forward"
        if command_type is None:
            return "go2-joystick-direction"
        return f"go2-joystick-{command_type}"
    return ENV_TO_TASK_ID.get(env_name, _camel_to_kebab(env_name))


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
    minari_dataset_path: str = "/home/luana/Documents/OfflineRL/Benchmark/CORL/datasets"
    command_type: str = None
    num_eval_episodes: int = 20  # Number of episodes to run when evaluating each checkpoint to find peak
    peak_margin_percent: float = 2.0  # Margin percentage for peak selection (e.g., 2.0 means within 2% of max)
    max_eval_workers: int = 4  # Number of parallel workers for checkpoint evaluation
    num_tiers: int = 5  # Number of performance tiers for categorizing checkpoints (samples distributed evenly across tiers)
    # ---- Item 3 additions ----
    medium_low: float = 0.4  # Lower bound (relative performance) of the medium band
    medium_high: float = 0.6  # Upper bound (relative performance) of the medium band
    p90_eval_episodes: int = 100  # Episodes used to estimate the expert P90 return threshold
    p90_percentile: float = 90.0  # Percentile threshold for the expert episode filter


def get_checkpoint_path(model_checkpoint: str):
    ckpt_path = str(epath.Path(model_checkpoint).resolve())
    FINETUNE_PATH = epath.Path(ckpt_path)
    latest_ckpts = list(FINETUNE_PATH.glob("*"))
    latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
    latest_ckpts.sort(key=lambda x: int(x.name))
    return latest_ckpts


def get_inference_fn(restore_checkpoint_path: str, env_name: str, deterministic: bool = True):
    """Build a jitted inference function from a checkpoint.

    Args:
        restore_checkpoint_path: Path to the checkpoint to restore.
        env_name: Environment name.
        deterministic: If True, use the policy mean ``mu(s)``; if False, sample
            from the policy distribution ``pi(a|s)`` (requires a fresh RNG key
            per call).
    """
    try:
        ppo_params = locomotion_params.brax_ppo_config(env_name)
    except:
        ppo_params = manipulation_params.brax_ppo_config(env_name)

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
    # Item G: collection is run WITHOUT domain randomization (DR is reserved for the
    # shifted eval datasets, plan item 9). DR only perturbs env physics, not the policy
    # network, so randomization_fn=None restores the exact checkpoint topology
    # (verified: deterministic return identical with vs without DR). Passing None also
    # avoids the registry's informational "no domain randomizer" print for envs lacking one.
    train_fn = functools.partial(
        ppo.train,
        **dict(ppo_training_params),
        network_factory=network_factory,
        randomization_fn=None,
    )

    make_inference_fn, params, metrics = train_fn(
        environment=registry.load(env_name),
        eval_env=registry.load(env_name),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        restore_checkpoint_path=restore_checkpoint_path,  
        seed=1,
    )

    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=deterministic))

    return jit_inference_fn


def rollout_episode_returns(
    inference_fn, env_name: str, num_episodes: int, seed: int = 42
) -> list:
    """Roll out a policy and return the list of episodic returns.

    Used both to rank checkpoints (mean) and to estimate the expert P90
    return threshold.
    """
    env = registry.load(env_name)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)

    returns = []
    for _ in range(num_episodes):
        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)

        episode_return = 0.0
        for _ in range(env._config.episode_length):
            rng, act_rng = jax.random.split(rng)
            action, _ = inference_fn(state.obs, act_rng)
            state = jit_step(state, action)
            episode_return += float(state.reward)
            if bool(state.done):
                break
        returns.append(episode_return)

    return returns


def evaluate_checkpoint(ckpt_path: str, env_name: str, num_eval_episodes: int = 20) -> float:
    """Evaluate a checkpoint deterministically and return its mean episodic return."""
    inference_fn = get_inference_fn(ckpt_path, env_name, deterministic=True)
    returns = rollout_episode_returns(inference_fn, env_name, num_eval_episodes)
    return float(np.mean(returns))


def find_peak_checkpoint(checkpoints: list, env_name: str, num_eval_episodes: int = 20, margin_percent: float = 2.0, max_workers: int = 4) -> int:
    """
    Evaluate all checkpoints and find the one with highest performance.
    Uses a margin-based approach: selects the earliest checkpoint that is within
    margin_percent of the maximum reward, to avoid selecting overfitted later checkpoints.
    
    Args:
        checkpoints: List of checkpoint paths
        env_name: Environment name
        num_eval_episodes: Number of episodes for evaluation
        margin_percent: Margin percentage (e.g., 2.0 means within 2% of max reward)
        max_workers: Number of parallel workers for evaluation
    
    Returns:
        Tuple of (peak_index, list of rewards)
    """
    print(f"Evaluating {len(checkpoints)} checkpoints to find peak performance (using {max_workers} workers)...")
    
    def eval_single_checkpoint(args):
        idx, ckpt = args
        avg_reward = evaluate_checkpoint(str(ckpt), env_name, num_eval_episodes)
        return idx, ckpt.name, avg_reward
    
    # Parallel evaluation of checkpoints
    rewards = [None] * len(checkpoints)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(eval_single_checkpoint, (i, ckpt)): i 
                   for i, ckpt in enumerate(checkpoints)}
        
        for future in tqdm(as_completed(futures), total=len(checkpoints), desc="Evaluating checkpoints"):
            idx, ckpt_name, avg_reward = future.result()
            rewards[idx] = avg_reward
            print(f"  Checkpoint {idx} ({ckpt_name}): avg reward = {avg_reward:.2f}")
    
    max_reward = max(rewards)
    max_idx = int(np.argmax(rewards))
    
    # Calculate threshold: accept checkpoints within margin_percent of max
    threshold = max_reward * (1 - margin_percent / 100.0)
    
    # Find the earliest checkpoint that meets or exceeds the threshold
    peak_idx = max_idx  # default to max if none found earlier
    for i, r in enumerate(rewards):
        if r >= threshold:
            peak_idx = i
            break
    
    print(f"Max reward: {max_reward:.2f} at checkpoint {max_idx} ({checkpoints[max_idx].name})")
    print(f"Threshold (within {margin_percent}%): {threshold:.2f}")
    print(f"Peak checkpoint selected at index {peak_idx} ({checkpoints[peak_idx].name}) with reward {rewards[peak_idx]:.2f}")
    
    return peak_idx, rewards


def relative_performance(rewards: list) -> list:
    """Map raw rewards to [0, 1] relative performance (robust to negative returns)."""
    min_reward = min(rewards)
    max_reward = max(rewards)
    reward_range = max_reward - min_reward
    if reward_range == 0:
        return [1.0 for _ in rewards]
    return [(r - min_reward) / reward_range for r in rewards]


def select_medium_checkpoint(checkpoints: list, rewards: list, low: float = 0.4, high: float = 0.6) -> int:
    """Select a single 'medium' checkpoint (~40-60% of relative performance).

    Picks the earliest (in training order) checkpoint whose relative performance
    falls inside [low, high]. If none qualifies, falls back to the checkpoint
    closest to the midpoint of the band.

    Returns:
        Index into ``checkpoints`` of the selected medium checkpoint.
    """
    rel = relative_performance(rewards)
    candidates = [i for i, rp in enumerate(rel) if low <= rp <= high]
    if candidates:
        medium_idx = candidates[0]
    else:
        mid = (low + high) / 2.0
        medium_idx = int(np.argmin([abs(rp - mid) for rp in rel]))

    print(
        f"Medium checkpoint selected at index {medium_idx} "
        f"({checkpoints[medium_idx].name}): reward={rewards[medium_idx]:.2f}, "
        f"relative_perf={rel[medium_idx]:.2f}"
    )
    return medium_idx


def analyze_peak(checkpoints: list, rewards: Optional[list], peak_idx: int, medium_idx: Optional[int]) -> dict:
    """Summarize the training curve for the dataset metadata (plan item 7.2).

    Derives the peak/medium iteration positions, their mean returns, ``peak_ratio``
    (medium/peak) and a coarse ``curve_shape``. ``iter_peak``/``iter_medium`` are
    normalized positions in [0, 1] across the ordered step-checkpoints. When only a
    single checkpoint is available the analysis collapses to that point.
    """
    n = len(checkpoints)
    if rewards is None or n <= 1:
        return {
            "iter_peak": 0.0,
            "iter_medium": None,
            "peak_return_mean": float(rewards[0]) if rewards else None,
            "medium_return_mean": None,
            "peak_ratio": None,
            "curve_shape": "single_checkpoint",
        }

    denom = n - 1
    iter_peak = peak_idx / denom
    iter_medium = (medium_idx / denom) if medium_idx is not None else None
    peak_return = float(rewards[peak_idx])
    medium_return = float(rewards[medium_idx]) if medium_idx is not None else None
    peak_ratio = (medium_return / peak_return) if (medium_return is not None and peak_return != 0) else None

    # Coarse curve shape: where the peak lands plus volatility of the curve.
    rel = relative_performance(rewards)
    volatility = float(np.std(np.diff(rel))) if n > 2 else 0.0
    if volatility > 0.25:
        curve_shape = "unstable"
    elif iter_peak < 0.3:
        curve_shape = "early_plateau"
    elif iter_peak > 0.7:
        curve_shape = "late_plateau"
    else:
        curve_shape = "linear"

    return {
        "iter_peak": iter_peak,
        "iter_medium": iter_medium,
        "peak_return_mean": peak_return,
        "medium_return_mean": medium_return,
        "peak_ratio": peak_ratio,
        "curve_shape": curve_shape,
    }


def categorize_checkpoints_into_tiers(checkpoints: list, rewards: list, num_tiers: int) -> dict:
    """
    Categorize checkpoints into performance tiers based on their relative performance.
    
    Args:
        checkpoints: List of checkpoint paths
        rewards: List of average rewards for each checkpoint
        num_tiers: Number of performance tiers to create
    
    Returns:
        Dictionary with tier info:
        - 'tiers': List of lists, where each inner list contains (checkpoint_idx, checkpoint_path, reward) tuples
        - 'tier_boundaries': List of (min_perf, max_perf) tuples for each tier
    """
    if len(checkpoints) == 0:
        return {'tiers': [], 'tier_boundaries': []}
    
    min_reward = min(rewards)
    max_reward = max(rewards)
    reward_range = max_reward - min_reward
    
    # Handle edge case where all rewards are the same
    if reward_range == 0:
        # All checkpoints go into the top tier
        tier_assignments = [(i, checkpoints[i], rewards[i]) for i in range(len(checkpoints))]
        return {
            'tiers': [[] for _ in range(num_tiers - 1)] + [tier_assignments],
            'tier_boundaries': [(min_reward, max_reward) for _ in range(num_tiers)]
        }
    
    # Calculate relative performance percentages (0-100%)
    relative_perfs = [(r - min_reward) / reward_range * 100 for r in rewards]
    
    # Create tier boundaries based on performance percentages
    # Tier 0: 0-20%, Tier 1: 20-40%, ..., Tier 4: 80-100% (for num_tiers=5)
    tier_size = 100.0 / num_tiers
    tier_boundaries = []
    for t in range(num_tiers):
        min_perf = t * tier_size
        max_perf = (t + 1) * tier_size
        tier_boundaries.append((min_perf, max_perf))
    
    # Assign checkpoints to tiers
    tiers = [[] for _ in range(num_tiers)]
    for i, (ckpt, reward, rel_perf) in enumerate(zip(checkpoints, rewards, relative_perfs)):
        # Find which tier this checkpoint belongs to
        tier_idx = min(int(rel_perf / tier_size), num_tiers - 1)  # Clamp to last tier for 100%
        tiers[tier_idx].append((i, ckpt, reward))
    
    # Print tier assignments
    print(f"\nCheckpoint categorization into {num_tiers} performance tiers:")
    print(f"Performance range: {min_reward:.2f} to {max_reward:.2f}")
    for t in range(num_tiers):
        perf_min, perf_max = tier_boundaries[t]
        reward_min = min_reward + (perf_min / 100) * reward_range
        reward_max = min_reward + (perf_max / 100) * reward_range
        print(f"  Tier {t} ({perf_min:.0f}%-{perf_max:.0f}%, reward {reward_min:.2f}-{reward_max:.2f}): "
              f"{len(tiers[t])} checkpoints")
        for idx, ckpt, reward in tiers[t]:
            print(f"    - Checkpoint {idx} ({ckpt.name}): reward = {reward:.2f}")
    
    return {
        'tiers': tiers,
        'tier_boundaries': tier_boundaries
    }


def compute_tier_based_distribution(checkpoints: list, rewards: list, total_samples: int, num_tiers: int) -> list:
    """
    Compute sample distribution based on performance tiers.
    
    Distributes total_samples evenly across tiers, then within each tier,
    distributes samples evenly among the checkpoints in that tier.
    
    Args:
        checkpoints: List of checkpoint paths
        rewards: List of average rewards for each checkpoint
        total_samples: Total number of samples to collect
        num_tiers: Number of performance tiers
    
    Returns:
        List of sample counts for each checkpoint (same order as input checkpoints)
    """
    tier_info = categorize_checkpoints_into_tiers(checkpoints, rewards, num_tiers)
    tiers = tier_info['tiers']
    
    # Count non-empty tiers
    non_empty_tiers = [t for t in tiers if len(t) > 0]
    num_non_empty_tiers = len(non_empty_tiers)
    
    if num_non_empty_tiers == 0:
        return []
    
    # Samples per tier (evenly distributed across non-empty tiers)
    samples_per_tier = total_samples // num_non_empty_tiers
    
    # Initialize sample counts for each checkpoint
    samples_distribution = [0] * len(checkpoints)
    
    total_assigned = 0
    for tier_idx, tier in enumerate(tiers):
        if len(tier) == 0:
            continue
        
        # Distribute samples evenly among checkpoints in this tier
        samples_per_ckpt_in_tier = samples_per_tier // len(tier)
        remainder = samples_per_tier % len(tier)
        
        for i, (ckpt_idx, ckpt_path, reward) in enumerate(tier):
            # Add one extra sample to first few checkpoints to handle remainder
            extra = 1 if i < remainder else 0
            samples_distribution[ckpt_idx] = samples_per_ckpt_in_tier + extra
            total_assigned += samples_distribution[ckpt_idx]
    
    # Handle any remaining samples due to rounding (add to first checkpoint)
    remaining = total_samples - total_assigned
    if remaining > 0 and len(checkpoints) > 0:
        samples_distribution[0] += remaining
    
    # Print distribution summary
    print(f"\nTier-based sample distribution:")
    print(f"  Total samples: {total_samples}")
    print(f"  Non-empty tiers: {num_non_empty_tiers}")
    print(f"  Samples per tier: ~{samples_per_tier}")
    for tier_idx, tier in enumerate(tiers):
        if len(tier) > 0:
            tier_samples = sum(samples_distribution[ckpt_idx] for ckpt_idx, _, _ in tier)
            print(f"  Tier {tier_idx}: {tier_samples} samples across {len(tier)} checkpoints")
    
    return samples_distribution


def collect_from_checkpoint(
    env,
    inference_fn,
    num_samples: int,
    rng,
    num_envs: int,
    source_code: Optional[int] = None,
    return_threshold: Optional[float] = None,
):
    """Collect ``num_samples`` stored transitions from a single policy.

    Centralizes the rollout loop shared by every difficulty. Handles:
      - per-step RNG splitting (required for stochastic ``pi(a|s)`` sampling);
      - optional ``source_code`` tagging in info (medium-expert);
      - optional P90 ``return_threshold`` gating (expert): episodes below the
        threshold are dropped by the collector and do not count toward the budget,
        which is measured by transitions actually persisted to storage.

    Returns:
        Tuple ``(rng, episode_returns)`` where ``episode_returns`` is the list of
        episodic returns observed during this phase (item 7.1). It feeds the
        ``return_min`` estimate and the dataset return histogram in metadata.
    """
    # Configure the collector / wrapper for this phase.
    env.env.collection_source = source_code
    env.set_episode_return_threshold(return_threshold)

    env.reset_timesteps()
    _, info = env.reset()
    obs = info["env_state"].obs

    start_steps = env.get_stored_steps()
    target_steps = start_steps + num_samples

    episode_returns = []
    pbar = tqdm(total=num_samples, desc="Collecting samples")
    last_stored = start_steps
    while env.get_stored_steps() < target_steps:
        terminated = jp.zeros(num_envs, dtype=bool)
        truncated = jp.zeros(num_envs, dtype=bool)
        episode_return = 0.0
        while not (terminated.all() or truncated.all()):
            rng, act_rng = jax.random.split(rng)
            action, _ = inference_fn(obs, act_rng)
            _, rew, terminated, truncated, _, info = env.step(action)
            obs = info["env_state"].obs
            episode_return += float(np.sum(np.asarray(rew)))
        episode_returns.append(episode_return)
        # Episode finished: flush to storage (drops sub-threshold episodes).
        _, info = env.reset()
        obs = info["env_state"].obs

        stored = env.get_stored_steps()
        pbar.update(max(0, stored - last_stored))
        last_stored = stored
    pbar.close()

    # Clear phase-specific configuration so it does not leak to the next call.
    env.env.collection_source = None
    env.set_episode_return_threshold(None)

    return rng, episode_returns


def resolve_step_checkpoints(model_checkpoints: list) -> list:
    """Flatten the provided run directories into a single ordered list of step-checkpoints."""
    all_ckpts = []
    for model_checkpoint in model_checkpoints:
        all_ckpts.extend(get_checkpoint_path(model_checkpoint))
    return all_ckpts


@pyrallis.wrap()
def main(config: Config):
    print(config)

    if config.difficulty not in VALID_DIFFICULTIES:
        raise ValueError(
            f"Difficulty '{config.difficulty}' not supported. "
            f"Valid options: {VALID_DIFFICULTIES}"
        )
    if config.model_checkpoint is None or len(config.model_checkpoint) == 0:
        raise ValueError("config.model_checkpoint must point to at least one training run directory.")

    # Item 6.1: collection is sequential (one episode per buffer flush). The P90
    # filter and the transition budget are only well-defined with a single env,
    # so force num_envs=1 regardless of what was requested.
    if config.num_envs != 1:
        print(
            f"[warning] Data collection runs sequentially with num_envs=1; "
            f"overriding requested num_envs={config.num_envs} -> 1."
        )
        config.num_envs = 1

    os.environ["MINARI_DATASETS_PATH"] = config.minari_dataset_path

    env = wrapper_collector(
        config.env_name,
        config.num_envs,
        config.seed,
        config.action_repeat,
        config.device,
        config.command_type,
    )
    rng = jax.random.PRNGKey(config.seed)

    # All run directories are merged into one ordered list of step-checkpoints.
    checkpoints = resolve_step_checkpoints(config.model_checkpoint)
    print(f"Resolved {len(checkpoints)} step-checkpoints from {len(config.model_checkpoint)} run(s).")

    # Evaluate checkpoints once (shared by selection-based difficulties).
    checkpoint_rewards = None
    if len(checkpoints) > 1:
        _, checkpoint_rewards = find_peak_checkpoint(
            checkpoints,
            config.env_name,
            num_eval_episodes=config.num_eval_episodes,
            margin_percent=config.peak_margin_percent,
            max_workers=config.max_eval_workers,
        )

    def get_peak_checkpoint():
        """The expert (peak) checkpoint: earliest within margin% of the max return."""
        if len(checkpoints) == 1:
            return checkpoints[0]
        max_reward = max(checkpoint_rewards)
        thresh = max_reward * (1 - config.peak_margin_percent / 100.0)
        peak_idx = next(i for i, r in enumerate(checkpoint_rewards) if r >= thresh)
        print(
            f"Peak checkpoint: index {peak_idx} ({checkpoints[peak_idx].name}), "
            f"reward={checkpoint_rewards[peak_idx]:.2f}"
        )
        return checkpoints[peak_idx]

    def estimate_p90_threshold(expert_inference_fn):
        returns = rollout_episode_returns(
            expert_inference_fn, config.env_name, config.p90_eval_episodes, seed=config.seed
        )
        threshold = float(np.percentile(returns, config.p90_percentile))
        print(
            f"Expert P{config.p90_percentile:.0f} threshold = {threshold:.2f} "
            f"(from {config.p90_eval_episodes} episodes; "
            f"mean={np.mean(returns):.2f}, max={np.max(returns):.2f})"
        )
        return threshold

    # Item 7: tracking shared across difficulties for the dataset metadata.
    episode_returns: list = []
    peak_idx = None
    medium_idx = None
    if checkpoint_rewards is not None:
        max_reward = max(checkpoint_rewards)
        thresh = max_reward * (1 - config.peak_margin_percent / 100.0)
        peak_idx = next(i for i, r in enumerate(checkpoint_rewards) if r >= thresh)

    if config.difficulty == "expert":
        # Single peak checkpoint, deterministic policy, episodes filtered by P90.
        peak_ckpt = get_peak_checkpoint()
        print(f"[expert] peak checkpoint: {peak_ckpt}")
        inference_fn = get_inference_fn(peak_ckpt, config.env_name, deterministic=True)
        threshold = estimate_p90_threshold(inference_fn)
        rng, episode_returns = collect_from_checkpoint(
            env, inference_fn, config.num_samples, rng, config.num_envs,
            return_threshold=threshold,
        )

    elif config.difficulty == "medium":
        # Single ~40-60% checkpoint, stochastic policy, no filtering.
        if len(checkpoints) == 1:
            medium_ckpt = checkpoints[0]
            print("[medium] only one checkpoint available; using it as the medium policy.")
        else:
            medium_idx = select_medium_checkpoint(
                checkpoints, checkpoint_rewards, config.medium_low, config.medium_high
            )
            medium_ckpt = checkpoints[medium_idx]
        print(f"[medium] checkpoint: {medium_ckpt}")
        inference_fn = get_inference_fn(medium_ckpt, config.env_name, deterministic=False)
        rng, episode_returns = collect_from_checkpoint(env, inference_fn, config.num_samples, rng, config.num_envs)

    elif config.difficulty == "medium-replay":
        # Checkpoints in the window [0, iter_medium], stochastic, tier-based budget.
        if len(checkpoints) == 1:
            raise ValueError(
                "medium-replay requires multiple step-checkpoints spanning random -> medium."
            )
        medium_idx = select_medium_checkpoint(
            checkpoints, checkpoint_rewards, config.medium_low, config.medium_high
        )
        window_ckpts = checkpoints[: medium_idx + 1]
        window_rewards = checkpoint_rewards[: medium_idx + 1]
        print(
            f"[medium-replay] using {len(window_ckpts)} checkpoints in window "
            f"[0, {medium_idx}] (random -> medium)."
        )
        samples_distribution = compute_tier_based_distribution(
            window_ckpts, window_rewards, config.num_samples, config.num_tiers
        )
        for j, ckpt in enumerate(window_ckpts):
            n = samples_distribution[j]
            if n <= 0:
                continue
            print(f"[medium-replay] collecting {n} samples from {ckpt.name}")
            inference_fn = get_inference_fn(ckpt, config.env_name, deterministic=False)
            rng, returns = collect_from_checkpoint(env, inference_fn, n, rng, config.num_envs)
            episode_returns.extend(returns)

    elif config.difficulty == "medium-expert":
        # 50% expert (deterministic, P90-filtered) + 50% medium (stochastic), one dataset.
        half = config.num_samples // 2

        peak_ckpt = get_peak_checkpoint()
        print(f"[medium-expert] expert checkpoint: {peak_ckpt}")
        expert_inference = get_inference_fn(peak_ckpt, config.env_name, deterministic=True)
        threshold = estimate_p90_threshold(expert_inference)
        rng, returns = collect_from_checkpoint(
            env, expert_inference, half, rng, config.num_envs,
            source_code=SOURCE_EXPERT, return_threshold=threshold,
        )
        episode_returns.extend(returns)

        if len(checkpoints) == 1:
            raise ValueError(
                "medium-expert requires multiple step-checkpoints to select a medium policy."
            )
        medium_idx = select_medium_checkpoint(
            checkpoints, checkpoint_rewards, config.medium_low, config.medium_high
        )
        medium_ckpt = checkpoints[medium_idx]
        print(f"[medium-expert] medium checkpoint: {medium_ckpt}")
        medium_inference = get_inference_fn(medium_ckpt, config.env_name, deterministic=False)
        rng, returns = collect_from_checkpoint(
            env, medium_inference, config.num_samples - half, rng, config.num_envs,
            source_code=SOURCE_MEDIUM,
        )
        episode_returns.extend(returns)

    # dataset_id convention: playground/<task-id>/<difficulty>-v<version> (plan Item 5).
    task_id = resolve_task_id(config.env_name, config.command_type)
    dataset_id = f"playground/{task_id}/{config.difficulty}-v{config.dataset_version}"

    # ---- Item 7: peak analysis, reference scores and normalized returns ----
    peak_info = analyze_peak(checkpoints, checkpoint_rewards, peak_idx or 0, medium_idx)

    if checkpoint_rewards is not None:
        return_expert = float(max(checkpoint_rewards))
        return_min = float(min(checkpoint_rewards))
    else:
        return_expert = float(np.max(episode_returns)) if episode_returns else None
        return_min = float(np.min(episode_returns)) if episode_returns else None

    # 7.3: warn when the peak lands too early in training (curriculum may be off).
    if peak_info["iter_peak"] is not None and peak_info["iter_peak"] < 0.2 and len(checkpoints) > 1:
        print(
            f"[warning] Peak reached at iter_peak={peak_info['iter_peak']:.2f} (<0.2). "
            f"Curve looks like an early plateau; consider noise injection for the medium policy."
        )

    span = (return_expert - return_min) if (return_expert is not None and return_min is not None) else None
    if span:
        normalized = [100.0 * (r - return_min) / span for r in episode_returns]
        normalized_score_mean = float(np.mean(normalized)) if normalized else None
    else:
        normalized_score_mean = None

    return_mean = float(np.mean(episode_returns)) if episode_returns else None
    return_std = float(np.std(episode_returns)) if episode_returns else None

    print(f"Creating dataset {dataset_id}")
    dataset = env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name=config.algorithm_name,
        author=config.author,
        author_email=config.author_email,
        code_permalink=config.code_permalink,
        description=config.description,
        ref_min_score=return_min,
        ref_max_score=return_expert,
    )

    # 7.5: attach the free-form metadata block (env, seeds, checkpoints, peak analysis).
    metadata = {
        "env_name": config.env_name,
        "task_id": task_id,
        "difficulty": config.difficulty,
        "command_type": config.command_type,
        "deterministic": config.difficulty in ("expert", "medium-expert"),
        "domain_randomization": False,
        "policy_checkpoint": str(checkpoints[peak_idx]) if peak_idx is not None else None,
        "medium_checkpoint": str(checkpoints[medium_idx]) if medium_idx is not None else None,
        "num_checkpoints": len(checkpoints),
        "seeds": {"env_seed": config.seed, "sampling_seed": config.seed},
        "collection_config": {
            "num_samples": config.num_samples,
            "num_envs": config.num_envs,
            "p90_eval_episodes": config.p90_eval_episodes,
            "p90_percentile": config.p90_percentile,
            "medium_low": config.medium_low,
            "medium_high": config.medium_high,
        },
        "peak_analysis": peak_info,
        "return_min": return_min,
        "return_expert": return_expert,
        "return_mean": return_mean,
        "return_std": return_std,
        "normalized_score_mean": normalized_score_mean,
        "num_episodes": len(episode_returns),
    }
    # C.3: record curriculum usage and the level scheme for Go2RoughCurriculum.
    if getattr(env.env, "curriculum", False):
        metadata["curriculum_wrapper"] = True
        metadata["curriculum_scheme"] = {
            "num_rows": env.env.curriculum_num_rows,
            "num_cols": env.env.curriculum_num_cols,
        }
    else:
        metadata["curriculum_wrapper"] = False
    dataset.storage.update_metadata(metadata)

    # 7.6: human-readable metadata.json next to the dataset.
    dataset_dir = os.path.join(config.minari_dataset_path, *dataset_id.split("/"))
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata written to {metadata_path}")

    print(
        f"Dataset created with {dataset.total_steps} timesteps and {dataset.total_episodes} episodes"
    )



if __name__ == "__main__":
    main()
