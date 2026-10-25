"""Load a trained car policy and watch it without doing any more training."""

# Register the MJCF importer's USD schemas before a scene is created.
from isaacsim.asset.importer.mjcf import MJCFImporter  # noqa: F401

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
from isaaclab.app import launch_simulation
from isaaclab.utils import to_dict, update_class_from_dict
from isaaclab.visualizers import WindowCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg import CartpoleDirectPPORunnerCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
from rsl_rl.runners import OnPolicyRunner

from diff_car_paths import find_latest_checkpoint


def load_agent_config(checkpoint: Path) -> dict:
    """Rebuild the saved network and its observation normalizers exactly."""
    config_path = checkpoint.parent / "agent_cfg.json"
    if config_path.is_file():
        print(f"Using saved PPO configuration: {config_path}", flush=True)
        return json.loads(config_path.read_text(encoding="utf-8"))
    # The earliest car checkpoints used these settings before configs were saved.
    print("Warning: no saved agent_cfg.json; using the original Cartpole PPO settings.", flush=True)
    cfg = CartpoleDirectPPORunnerCfg()
    cfg.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    return to_dict(cfg)


def load_environment(checkpoint: Path, use_current_env: bool = False):
    """Load the run's source and XML so older policies keep their action meaning."""
    if use_current_env:
        from diff_car_env import DiffCarEnv, DiffCarEnvCfg

        print("Warning: using the current environment by request; its actions/observations may differ from training.", flush=True)
        return DiffCarEnv, DiffCarEnvCfg()

    source_path = checkpoint.parent / "source" / "diff_car_env.py"
    xml_path = source_path.with_name("diff_car.xml")
    if not source_path.is_file() or not xml_path.is_file():
        raise FileNotFoundError(
            f"This run has no complete environment/XML snapshot in {source_path.parent}. "
            "To intentionally evaluate it with today's environment, add --use-current-env. "
            "The action and observation meanings may have changed."
        )
    module_name = "_diff_car_snapshot_" + hashlib.sha256(str(source_path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import environment snapshot: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    env_cfg = module.DiffCarEnvCfg()
    config_path = checkpoint.parent / "env_cfg.json"
    if config_path.is_file():
        update_class_from_dict(env_cfg, json.loads(config_path.read_text(encoding="utf-8")))
    # Saved config records the original working XML path. Use its archived copy.
    env_cfg.scene.car.spawn.asset_path = str(xml_path)
    print(f"Using environment snapshot: {source_path}", flush=True)
    return module.DiffCarEnv, env_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint to load; defaults to the newest saved file.")
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps; 0 means until the viewer closes.")
    parser.add_argument("--no-viewer", action="store_true", help="Check inference without opening a window.")
    parser.add_argument("--use-current-env", action="store_true",
                        help="Explicitly use today's environment instead of the run's saved source and XML.")
    args = parser.parse_args()
    if args.steps < 0 or (args.no_viewer and args.steps == 0):
        parser.error("Use a positive --steps value when running with --no-viewer.")

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint()
    checkpoint = checkpoint.resolve(strict=True)
    print(f"Loading checkpoint: {checkpoint}", flush=True)

    # Restore the training contract, with just one simulated car for viewing.
    agent_cfg = load_agent_config(checkpoint)
    environment_type, env_cfg = load_environment(checkpoint, args.use_current_env)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.get("seed", 42)
    env_cfg.sim.device = "cuda:0"
    env_cfg.sim.render_interval = env_cfg.decimation
    if not args.no_viewer:
        env_cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(
            eye=(4.0, -4.0, 3.0),
            lookat=(0.0, 0.0, 0.8),
            window=WindowCfg(size=(1100, 750)),
            streaming_view=False,
            # Newton hides the wheel colliders when their white visual marks exist.
            show_collision=True,
        )]

    with launch_simulation(env_cfg):
        raw_env = environment_type(cfg=env_cfg)
        try:
            # Earlier scripts always clipped at 1.0 despite storing null in their config.
            clip_actions = agent_cfg.get("clip_actions")
            env = RslRlVecEnvWrapper(raw_env, clip_actions=1.0 if clip_actions is None else clip_actions)
            runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=raw_env.device)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=raw_env.device)
            observations = env.get_observations()
            print(f"Loaded iteration {runner.current_learning_iteration}. Evaluation only; no learning updates.", flush=True)
            print("Close the viewer or press Ctrl+C to stop.", flush=True)

            steps = episode = episode_steps = 0
            episode_return = 0.0
            with torch.inference_mode():
                while raw_env.sim.is_running() and (args.steps == 0 or steps < args.steps):
                    start = time.perf_counter()
                    actions = policy(observations)
                    observations, rewards, dones, _ = env.step(actions)
                    policy.reset(dones)
                    episode_return += rewards[0].item()
                    episode_steps += 1
                    steps += 1
                    if dones[0].item():
                        episode += 1
                        print(f"Episode {episode}: return={episode_return:.3f}, duration={episode_steps * raw_env.step_dt:.2f}s", flush=True)
                        episode_return = 0.0
                        episode_steps = 0
                    # Pace viewing by simulated time; slow computation may still run below real time.
                    if not args.no_viewer:
                        time.sleep(max(0.0, raw_env.step_dt - (time.perf_counter() - start)))
            print(f"Evaluation finished after {steps} steps.", flush=True)
        except KeyboardInterrupt:
            print("Evaluation stopped.", flush=True)
        finally:
            raw_env.close()


if __name__ == "__main__":
    main()
