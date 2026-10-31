#!/usr/bin/env python3
"""Watch a trained policy, zero-input fall, or diagnostic PD response in MuJoCo."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.pivot_balance_env import PivotBalanceEnv, VISUAL_XML, TRAIN_XML


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['ppo', 'zero', 'pd'], default='ppo')
    parser.add_argument('--policy', default=str(ROOT / 'checkpoints/best_model.zip'))
    parser.add_argument('--fast', action='store_true', help='Omit detailed visual meshes')
    parser.add_argument('--tilt', type=float, default=5.0, help='Initial pitch in degrees')
    args = parser.parse_args()
    env = PivotBalanceEnv(xml_path=TRAIN_XML if args.fast else VISUAL_XML)
    policy = PPO.load(args.policy, device='cpu') if args.mode == 'ppo' else None
    obs, _ = env.reset(options={'pitch': np.deg2rad(args.tilt), 'pitch_rate': 0.0})
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            if args.mode == 'ppo':
                command, _ = policy.predict(obs, deterministic=True)
            elif args.mode == 'zero':
                command = np.array([0.0], dtype=np.float32)
            else:
                pitch, _, rate, forward, _, _, x, _ = env._state()
                # Initial diagnostic only. Gains are not tuned for the physical robot.
                cmd = np.clip(5.0 * pitch + 0.8 * rate
                              - 0.6 * forward - 0.12 * x, -1, 1)
                command = np.array([cmd], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(command)
            viewer.sync()
            time.sleep(env.model.opt.timestep * env.frame_skip)
            if terminated or truncated:
                obs, _ = env.reset(options={'pitch': np.deg2rad(args.tilt), 'pitch_rate': 0.0})
    env.close()


if __name__ == '__main__':
    main()
