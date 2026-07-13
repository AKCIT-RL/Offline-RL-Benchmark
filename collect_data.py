import pyrallis
import re
import json
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, SupportsFloat, Type
import functools
from etils import epath
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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
from data_collector.wrapper_fn import apply_command_override
import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Persistent XLA compilation cache: checkpoint evaluation spawns one fresh process
# per checkpoint (VRAM hygiene), and without this every process re-JITs the same
# graphs (~most of the per-checkpoint startup cost). This module is re-imported by
# spawned workers, so the cache is enabled there too.
jax.config.update(
    "jax_compilation_cache_dir",
    os.environ.get("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jax_compilation")),
)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)


# Difficulty (dataset type) -> collection semantics. See collect_data_refactor_plan.md (Item 3).
VALID_DIFFICULTIES = ("expert", "medium", "medium-replay", "medium-expert")

# Integer codes recorded in info["source"] for the medium-expert dataset.
SOURCE_EXPERT = 0
SOURCE_MEDIUM = 1

# Go2RoughCurriculum defaults to the Warp collision backend (impl="warp"). On the
# procedural heightfield that path is unreliable for offline collection: its CCD
# (convex/EPA) contact buffer grows a large per-step allocation that OOMs the GPU
# (shared with JAX), while shrinking that buffer (naccdmax) instead overflows and
# silently drops contacts. The JAX/MJX backend uses fixed compile-time contact
# buffers (no mempool growth, no overflow), simulates the heightfield collision
# with valid physics, runs ~3x faster here, and emits none of the Warp ccd
# warnings. It is also the standard brax-training backend, so dynamics match well.

# ``njmax`` is the per-world cap on active constraint rows (nefc). When the solver
# needs more rows than this, mujoco_warp DROPS the excess contacts and prints
# ``nefc overflow - please increase njmax to <N>``, silently degrading the physics
# recorded into the dataset. The upstream defaults are sized for the nominal gait
# (G1JoystickFlatTerrain: 29*2+8*4 = 90, H1JoystickGaitTracking: 19+8*4 = 51), but
# occasional multi-contact frames during collection overflow them (observed peaks:
# G1 -> 96, H1 -> 55). Bump them with generous headroom so no contact is ever
# dropped. The cost is a few extra (nworld, njmax) arrays, negligible at collection
# env counts.
COLLECTION_NJMAX = {
    "G1JoystickFlatTerrain": 160,
    "H1JoystickGaitTracking": 128,
}


def env_config_overrides(env_name: str) -> Optional[dict]:
    """Per-env registry.load overrides applied during collection (see above)."""
    if env_name == "Go2RoughCurriculum":
        return {"impl": "jax"}
    if env_name in COLLECTION_NJMAX:
        return {"njmax": COLLECTION_NJMAX[env_name]}
    return None

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
    num_envs: int = 16  # Parallel envs for data collection (batched on GPU; ~linear speedup)
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
    eval_tasks_per_child: int = 1  # Checkpoints evaluated per worker process before respawn (>1 amortizes startup; VRAM leaks ~linearly per ckpt, keep small)
    num_tiers: int = 5  # Number of performance tiers for categorizing checkpoints (samples distributed evenly across tiers)
    # ---- Item 3 additions ----
    medium_low: float = 0.4  # Lower bound (relative performance) of the medium band
    medium_high: float = 0.6  # Upper bound (relative performance) of the medium band
    p90_eval_episodes: int = 100  # Episodes used to estimate the expert P90 return threshold
    p90_percentile: float = 90.0  # Percentile threshold for the expert episode filter
    curriculum_all_levels: bool = True  # Curriculum envs: spread collection across ALL difficulty levels (per-level P90 gating for expert)
    overwrite: bool = False  # If True, delete an existing dataset dir and regenerate it instead of skipping.


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
        environment=registry.load(env_name, config_overrides=env_config_overrides(env_name)),
        eval_env=registry.load(env_name, config_overrides=env_config_overrides(env_name)),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        restore_checkpoint_path=restore_checkpoint_path,  
        seed=1,
    )

    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=deterministic))

    return jit_inference_fn


def rollout_episode_returns(
    inference_fn, env_name: str, num_episodes: int, seed: int = 42, command_type: Optional[str] = None
) -> list:
    """Roll out a policy and return the list of episodic returns.

    Used both to rank checkpoints (mean) and to estimate the expert P90
    return threshold. All episodes run in PARALLEL (vmap over ``num_episodes``)
    and the whole rollout is one jitted ``lax.scan``, so there is a single
    device round-trip instead of ``num_episodes * episode_length`` python steps.
    Rewards are masked after each episode's first ``done``.

    ``command_type`` applies the same joystick command override used by the
    collection wrapper. The P90 threshold MUST be estimated under the same
    command regime as the collection, otherwise the absolute gate is
    meaningless (e.g. fowardfixed returns below the random-command P90 would
    drop every episode).
    """
    env = registry.load(env_name, config_overrides=env_config_overrides(env_name))
    episode_length = env._config.episode_length

    def run(reset_keys, rng):
        state = jax.vmap(env.reset)(reset_keys)
        state = apply_command_override(state, command_type)

        def step_fn(carry, _):
            state, rng, returns, done = carry
            rng, act_rng = jax.random.split(rng)
            action, _ = inference_fn(state.obs, act_rng)
            state = jax.vmap(env.step)(state, action)
            state = apply_command_override(state, command_type)
            # jp.where (not reward * (1-done)) so post-done NaNs cannot poison returns.
            returns = returns + jp.where(done > 0.5, 0.0, state.reward)
            done = jp.maximum(done, state.done)
            return (state, rng, returns, done), None

        init = (state, rng, jp.zeros(num_episodes), jp.zeros(num_episodes))
        (_, _, returns, _), _ = jax.lax.scan(step_fn, init, length=episode_length)
        return returns

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    reset_keys = jax.random.split(reset_rng, num_episodes)
    returns = jax.jit(run)(reset_keys, rng)
    return [float(r) for r in np.asarray(returns)]


def rollout_returns_by_level(
    inference_fn,
    env_name: str,
    num_rows: int,
    num_cols: int,
    episodes_per_level: int,
    seed: int = 42,
    max_parallel: Optional[int] = None,
):
    """Roll out a policy spread across ALL curriculum levels.

    Spawns ``episodes_per_level`` episodes on each of the ``num_rows`` difficulty
    levels (random columns) via the env's ``reset_to``. Returns
    ``(returns, levels)`` numpy arrays aligned per episode, so the caller can
    estimate a P90 threshold PER level. Used only for curriculum envs, where a
    single global threshold would drop every harder (lower-return) level.

    Episodes are rolled out in chunks of at most ``max_parallel`` (default: all
    at once). Chunking bounds the warp collision mempool to the collection's
    footprint: the procedural-terrain heightfield collision (convex/EPA) is very
    VRAM-hungry and warp does NOT release its mempool in-process, so a large
    single-shot rollout here would OOM the subsequent collection.
    """
    env = registry.load(env_name, config_overrides=env_config_overrides(env_name))
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    episode_length = env._config.episode_length

    rng_np = np.random.default_rng(seed)
    levels_np = np.repeat(np.arange(num_rows), episodes_per_level).astype(np.int32)
    cols_np = rng_np.integers(0, num_cols, size=levels_np.shape[0]).astype(np.int32)
    total = int(levels_np.shape[0])
    if not max_parallel or max_parallel <= 0:
        max_parallel = total

    def run(reset_keys, levels, cols, rng):
        n = levels.shape[0]
        state = jax.vmap(base.reset_to)(reset_keys, levels, cols)

        def step_fn(carry, _):
            state, rng, returns, done = carry
            rng, act_rng = jax.random.split(rng)
            action, _ = inference_fn(state.obs, act_rng)
            state = jax.vmap(base.step)(state, action)
            returns = returns + jp.where(done > 0.5, 0.0, state.reward)
            done = jp.maximum(done, state.done)
            return (state, rng, returns, done), None

        init = (state, rng, jp.zeros(n), jp.zeros(n))
        (_, _, returns, _), _ = jax.lax.scan(step_fn, init, length=episode_length)
        return returns

    run_jit = jax.jit(run)
    rng = jax.random.PRNGKey(seed)
    returns_chunks = []
    for start in range(0, total, max_parallel):
        end = min(start + max_parallel, total)
        levels_c = jp.asarray(levels_np[start:end])
        cols_c = jp.asarray(cols_np[start:end])
        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, end - start)
        returns_chunks.append(np.asarray(run_jit(reset_keys, levels_c, cols_c, rng)))
    returns = np.concatenate(returns_chunks) if returns_chunks else np.array([])
    return returns, levels_np


def evaluate_checkpoint(ckpt_path: str, env_name: str, num_eval_episodes: int = 20) -> float:
    """Evaluate a checkpoint deterministically and return its mean episodic return."""
    inference_fn = get_inference_fn(ckpt_path, env_name, deterministic=True)
    returns = rollout_episode_returns(inference_fn, env_name, num_eval_episodes)
    return float(np.mean(returns))


def _eval_worker(args):
    """Process-pool entry point: evaluate one checkpoint in a fresh interpreter.

    Each checkpoint builds a new env + PPO net + jitted graphs. jax.clear_caches()
    frees the XLA cache but NOT the warp mempool, so VRAM still accumulates and OOMs
    after ~20 checkpoints. Running each evaluation in its own short-lived process
    (``eval_tasks_per_child`` tasks per process, default 1) guarantees jax and warp
    release all device memory on exit.
    """
    idx, ckpt, env_name, num_eval_episodes = args
    return idx, ckpt.name, evaluate_checkpoint(str(ckpt), env_name, num_eval_episodes)


EVAL_CACHE_FILENAME = "eval_cache.json"


def _eval_cache_path(ckpt) -> str:
    """Cache file lives next to the step-checkpoints of the run that owns them."""
    return str(epath.Path(ckpt).parent / EVAL_CACHE_FILENAME)


def _eval_cache_key(env_name: str, num_eval_episodes: int) -> str:
    return f"{env_name}|ep{num_eval_episodes}"


def _load_cached_reward(ckpt, env_name: str, num_eval_episodes: int) -> Optional[float]:
    path = _eval_cache_path(ckpt)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            cache = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    value = cache.get(_eval_cache_key(env_name, num_eval_episodes), {}).get(ckpt.name)
    return float(value) if value is not None else None


def _store_cached_reward(ckpt, env_name: str, num_eval_episodes: int, reward: float) -> None:
    """Incrementally persist one evaluation result (crash-safe write)."""
    path = _eval_cache_path(ckpt)
    cache = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            cache = {}
    cache.setdefault(_eval_cache_key(env_name, num_eval_episodes), {})[ckpt.name] = reward
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp_path, path)


def find_peak_checkpoint(checkpoints: list, env_name: str, num_eval_episodes: int = 20, margin_percent: float = 2.0, max_workers: int = 1, tasks_per_child: int = 1) -> int:
    """
    Evaluate all checkpoints and find the one with highest performance.
    Uses a margin-based approach: selects the earliest checkpoint that is within
    margin_percent of the maximum reward, to avoid selecting overfitted later checkpoints.

    Results are cached in ``eval_cache.json`` next to each run's checkpoints
    (keyed by env + num_eval_episodes), so other difficulties of the same env
    and crash re-runs skip already-evaluated checkpoints.
    
    Args:
        checkpoints: List of checkpoint paths
        env_name: Environment name
        num_eval_episodes: Number of episodes for evaluation
        margin_percent: Margin percentage (e.g., 2.0 means within 2% of max reward)
        max_workers: Number of parallel workers for evaluation
        tasks_per_child: Checkpoints evaluated per worker process before respawn
    
    Returns:
        Tuple of (peak_index, list of rewards)
    """
    rewards = [None] * len(checkpoints)
    to_eval = []
    for i, ckpt in enumerate(checkpoints):
        cached = _load_cached_reward(ckpt, env_name, num_eval_episodes)
        if cached is not None:
            rewards[i] = cached
        else:
            to_eval.append(i)

    n_cached = len(checkpoints) - len(to_eval)
    if n_cached:
        print(f"Loaded {n_cached}/{len(checkpoints)} checkpoint evaluations from cache.")

    if to_eval:
        print(f"Evaluating {len(to_eval)} checkpoints to find peak performance (using {max_workers} workers)...")
        # Each checkpoint is evaluated in its own short-lived process so jax+warp fully
        # release VRAM between checkpoints (warp does not free its mempool in-process).
        ctx = multiprocessing.get_context("spawn")
        tasks = [(i, checkpoints[i], env_name, num_eval_episodes) for i in to_eval]
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, max_tasks_per_child=tasks_per_child) as executor:
            futures = {executor.submit(_eval_worker, t): t[0] for t in tasks}
            for future in tqdm(as_completed(futures), total=len(to_eval), desc="Evaluating checkpoints"):
                idx, ckpt_name, avg_reward = future.result()
                rewards[idx] = avg_reward
                _store_cached_reward(checkpoints[idx], env_name, num_eval_episodes, avg_reward)
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
    per_level_thresholds: Optional[dict] = None,
):
    """Collect ``num_samples`` stored transitions from a single policy.

    Centralizes the rollout loop shared by every difficulty. Handles:
      - batched collection: ``num_envs`` episodes are rolled out in lockstep per
        iteration (the collector ignores post-done steps of finished envs);
      - per-step RNG splitting (required for stochastic ``pi(a|s)`` sampling);
      - optional ``source_code`` tagging in info (medium-expert);
      - optional P90 ``return_threshold`` gating (expert): episodes below the
        threshold are dropped by the collector and do not count toward the budget,
        which is measured by transitions actually persisted to storage (and capped
        at the target so a finished batch cannot overshoot it).

    Returns:
        Tuple ``(rng, episode_returns)`` where ``episode_returns`` is the list of
        per-env episodic returns observed during this phase (item 7.1). It feeds
        the ``return_min`` estimate and the dataset return histogram in metadata.
    """
    # Configure the collector / wrapper for this phase.
    env.env.collection_source = source_code
    env.set_episode_return_threshold(return_threshold)
    env.set_per_level_return_thresholds(per_level_thresholds)

    env.reset_timesteps()
    _, info = env.reset()
    obs = info["env_state"].obs

    start_steps = env.get_stored_steps()
    target_steps = start_steps + num_samples
    env.set_max_stored_steps(target_steps)

    episode_returns = []
    pbar = tqdm(total=num_samples, desc="Collecting samples")
    last_stored = start_steps
    zero_progress_batches = 0
    while env.get_stored_steps() < target_steps:
        terminated = np.zeros(num_envs, dtype=bool)
        truncated = np.zeros(num_envs, dtype=bool)
        active = np.ones(num_envs, dtype=bool)
        episode_return = np.zeros(num_envs)
        while not (terminated.all() or truncated.all()):
            rng, act_rng = jax.random.split(rng)
            action, _ = inference_fn(obs, act_rng)
            _, rew, terminated, truncated, _, info = env.step(action)
            obs = info["env_state"].obs
            # Only accumulate rewards of envs still inside their episode.
            episode_return[active] += np.asarray(rew)[active]
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            active = ~(terminated | truncated)
        episode_returns.extend(float(r) for r in episode_return)
        # Batch finished: flush to storage (drops sub-threshold episodes).
        _, info = env.reset()
        obs = info["env_state"].obs

        stored = env.get_stored_steps()
        pbar.update(max(0, stored - last_stored))
        # Safeguard: with a return_threshold, a threshold that no episode can
        # reach loops forever at 0%. Surface it loudly instead of hanging silently.
        if stored == last_stored:
            zero_progress_batches += 1
            if zero_progress_batches in (20, 100, 500):
                gate = (
                    f"per_level_thresholds={per_level_thresholds}"
                    if per_level_thresholds is not None
                    else f"threshold={return_threshold}"
                )
                print(
                    f"[warning] {zero_progress_batches} consecutive episode batches "
                    f"stored 0 transitions ({gate}). "
                    f"Recent episode returns: "
                    f"{[round(r, 2) for r in episode_returns[-num_envs:]]}"
                )
        else:
            zero_progress_batches = 0
        last_stored = stored
    pbar.close()

    # Clear phase-specific configuration so it does not leak to the next call.
    env.env.collection_source = None
    env.set_episode_return_threshold(None)
    env.set_per_level_return_thresholds(None)
    env.set_max_stored_steps(None)

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

    # Skip work that is already done: lets the whole generation job be re-submitted
    # after a crash/fix without recreating (or failing on) existing datasets.
    task_id = resolve_task_id(config.env_name, config.command_type)
    dataset_id = f"playground/{task_id}/{config.difficulty}-v{config.dataset_version}"
    dataset_dir = os.path.join(config.minari_dataset_path, *dataset_id.split("/"))
    if os.path.isdir(dataset_dir):
        if config.overwrite:
            print(f"[overwrite] Removing existing dataset {dataset_id} at {dataset_dir}.")
            shutil.rmtree(dataset_dir)
        else:
            print(f"[skip] Dataset {dataset_id} already exists at {dataset_dir}.")
            return

    # Batched collection: num_envs episodes are rolled out in lockstep and the
    # collector keeps one buffer per env (post-done steps of finished envs are
    # ignored, the P90 filter is per-episode and the budget is capped at store
    # time), so throughput scales ~linearly with num_envs on GPU.
    print(f"Collecting with num_envs={config.num_envs} parallel environments.")

    os.environ["MINARI_DATASETS_PATH"] = config.minari_dataset_path

    env = wrapper_collector(
        config.env_name,
        config.num_envs,
        config.seed,
        config.action_repeat,
        config.device,
        config.command_type,
        curriculum_all_levels=config.curriculum_all_levels,
        config_overrides=env_config_overrides(config.env_name),
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
            tasks_per_child=config.eval_tasks_per_child,
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
        # command_type matters: the threshold must reflect the same command
        # regime as the collection (see rollout_episode_returns docstring).
        returns = rollout_episode_returns(
            expert_inference_fn, config.env_name, config.p90_eval_episodes,
            seed=config.seed, command_type=config.command_type,
        )
        threshold = float(np.percentile(returns, config.p90_percentile))
        print(
            f"Expert P{config.p90_percentile:.0f} threshold = {threshold:.2f} "
            f"(from {config.p90_eval_episodes} episodes; "
            f"mean={np.mean(returns):.2f}, max={np.max(returns):.2f})"
        )
        return threshold

    # Curriculum all-levels collection spreads episodes over every difficulty
    # level; the expert gate must then be a P90 threshold PER level (a single
    # global threshold would drop every harder, lower-return level).
    is_curriculum_all_levels = getattr(env.env, "curriculum", False) and getattr(
        env.env, "curriculum_all_levels", False
    )

    def estimate_per_level_p90_thresholds(expert_inference_fn):
        num_rows = int(env.env.curriculum_num_rows)
        num_cols = int(env.env.curriculum_num_cols)
        episodes_per_level = max(1, config.p90_eval_episodes // num_rows)
        returns, levels = rollout_returns_by_level(
            expert_inference_fn, config.env_name, num_rows, num_cols,
            episodes_per_level, seed=config.seed, max_parallel=config.num_envs,
        )
        thresholds = {}
        for level in range(num_rows):
            level_returns = returns[levels == level]
            if level_returns.size:
                thresholds[level] = float(
                    np.percentile(level_returns, config.p90_percentile)
                )
        print(
            f"Expert per-level P{config.p90_percentile:.0f} thresholds "
            f"({episodes_per_level} episodes/level):"
        )
        for level in range(num_rows):
            level_returns = returns[levels == level]
            if level_returns.size:
                print(
                    f"  level {level}: threshold={thresholds[level]:.2f} "
                    f"(mean={level_returns.mean():.2f}, max={level_returns.max():.2f})"
                )
        return thresholds

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
        if is_curriculum_all_levels:
            per_level = estimate_per_level_p90_thresholds(inference_fn)
            rng, episode_returns = collect_from_checkpoint(
                env, inference_fn, config.num_samples, rng, config.num_envs,
                per_level_thresholds=per_level,
            )
        else:
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
        if is_curriculum_all_levels:
            per_level = estimate_per_level_p90_thresholds(expert_inference)
            rng, returns = collect_from_checkpoint(
                env, expert_inference, half, rng, config.num_envs,
                source_code=SOURCE_EXPERT, per_level_thresholds=per_level,
            )
        else:
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
    # (task_id/dataset_id resolved at the top of main for the early-skip check.)

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
        # Collision backend used for collection. Go2RoughCurriculum is collected
        # with the JAX/MJX backend (see env_config_overrides) instead of the
        # env-default Warp backend, which OOMs on the heightfield.
        "collision_impl": (env_config_overrides(config.env_name) or {}).get("impl", "warp"),
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
        # all_levels: episodes were spread uniformly over every difficulty level
        # (per-level P90 gating for expert), rather than the score-based progression.
        metadata["curriculum_all_levels"] = getattr(env.env, "curriculum_all_levels", False)
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
