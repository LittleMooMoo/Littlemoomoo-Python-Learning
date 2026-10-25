"""Train the car, keeping each run's logs and checkpoints in its own folder."""
# Register the importer's schemas before creating any USD scene.
from isaacsim.asset.importer.mjcf import MJCFImporter  # noqa: F401

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
from isaaclab.app import launch_simulation
from isaaclab.utils import to_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from diff_car_env import DiffCarEnv, DiffCarEnvCfg
from diff_car_paths import RUN_TIMEZONE, create_run_directory
from diff_car_ppo_cfg import DiffCarPPORunnerCfg


def write_json(path: Path, values: dict) -> None:
    path.write_text(json.dumps(values, indent=2,
                    default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3_00,
                        help="Number of PPO learning iterations (default: 300).")
    parser.add_argument("--continue_training", action="store_true",
                        help="Continue training from the checkpoint given by --model_path")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model path to resume training")
    args = parser.parse_args()
    checkpoint_path = args.model_path
    if args.iterations <= 0:
        parser.error("--iterations must be positive.")

    # 1. Configure simulation and PPO for this car.
    agent_cfg = DiffCarPPORunnerCfg()
    env_cfg = DiffCarEnvCfg()
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = "cuda:0"
    if agent_cfg.algorithm.gamma != env_cfg.reward_discount:
        raise ValueError(
            "PPO gamma must match env_cfg.reward_discount for the progress reward.")
    learning_iterations = args.iterations
    agent_cfg.max_iterations = learning_iterations

    # 2. Every invocation receives its own folder, independent of terminal location.
    run_dir = create_run_directory()
    print(f"Training run folder: {run_dir}", flush=True)
    write_json(run_dir / "agent_cfg.json", to_dict(agent_cfg))
    write_json(run_dir / "env_cfg.json", to_dict(env_cfg))
    source_dir = run_dir / "source"
    source_dir.mkdir()
    for name in (
        "diff_car_env.py", "diff_car.xml", "diff_car_ppo_cfg.py",
        "diff_car_train.py", "diff_car_evaluate.py", "diff_car_paths.py",
    ):
        shutil.copy2(Path(__file__).resolve().parent / name, source_dir / name)
    run_info = {
        "started_at": datetime.now(RUN_TIMEZONE).isoformat(),
        "environment": "DiffCarEnv",
        "task_stage": env_cfg.task_stage,
        "learning_iterations": learning_iterations,
        "status": "started",
        "training_completed": False,
    }
    write_json(run_dir / "run_info.json", run_info)

    # 3. Start physics, then create the environment and learning runner.
    with launch_simulation(env_cfg):
        raw_env = DiffCarEnv(cfg=env_cfg)
        model = None
        try:
            env = RslRlVecEnvWrapper(
                raw_env, clip_actions=agent_cfg.clip_actions)
            model = OnPolicyRunner(
                env, to_dict(agent_cfg), log_dir=str(run_dir), device="cuda:0"
            )
            if args.continue_training:
                model.load(str(checkpoint_path), map_location="cuda:0")
            # RSL-RL also writes model_0.pt, model_50.pt, etc. into this run_dir.
            model.learn(
                num_learning_iterations=learning_iterations,
                init_at_random_ep_len=agent_cfg.init_at_random_ep_len,
            )
            checkpoint = run_dir / "model.pt"
            model.save(str(checkpoint))
            run_info.update({
                "status": "training_completed",
                "training_completed": True,
                "completed_at": datetime.now(RUN_TIMEZONE).isoformat(),
                "final_checkpoint": str(checkpoint),
                "iteration": model.current_learning_iteration,
            })
            write_json(run_dir / "run_info.json", run_info)
            print(f"Training finished. Model saved: {checkpoint}", flush=True)

            # 4. Evaluate without learning updates.
            policy = model.get_inference_policy(device="cuda:0")
            with torch.inference_mode():
                observations, _ = env.reset()
                for _ in range(300):
                    actions = policy(observations)
                    observations, rewards, dones, info = env.step(actions)
                    policy.reset(dones)
            print("Evaluation finished.", flush=True)

        except KeyboardInterrupt:
            if model is not None:
                checkpoint = run_dir / "model_interrupted.pt"
                model.save(str(checkpoint))
                run_info.update({
                    "status": "evaluation_interrupted" if run_info["training_completed"] else "training_interrupted",
                    "interrupted_checkpoint": str(checkpoint),
                })
                print(f"Run stopped. Model saved: {checkpoint}", flush=True)
            write_json(run_dir / "run_info.json", run_info)
            raise

        except Exception as error:
            # Keep periodic checkpoints; do not overwrite them with possibly invalid state.
            run_info.update({
                "status": "evaluation_failed" if run_info["training_completed"] else "training_failed",
                "error": str(error),
            })
            write_json(run_dir / "run_info.json", run_info)
            raise

        finally:
            raw_env.close()


if __name__ == "__main__":
    main()
