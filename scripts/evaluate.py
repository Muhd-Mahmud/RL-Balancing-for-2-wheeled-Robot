#!/usr/bin/env python3
"""Evaluate trained PPO using multiple reproducible random reset seeds."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.pivot_balance_env import PivotBalanceEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--policy', default=str(ROOT / 'checkpoints/best_model.zip'))
    parser.add_argument('--seed', type=int, default=1000)
    args = parser.parse_args()
    policy = PPO.load(args.policy, device='cpu')
    env = PivotBalanceEnv()
    episodes = []
    for i in range(args.episodes):
        obs, _ = env.reset(seed=args.seed+i)
        done = False
        pitches, efforts, total_reward = [], [], 0.0
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            pitches.append(info['pitch_degrees'])
            efforts.append(abs(info['motor_command']))
            total_reward += reward
            done = terminated or truncated
        episodes.append({
            'seed': args.seed+i,
            'success': info['balance_success'],
            'duration_s': info['elapsed_seconds'],
            'pitch_rmse_deg': float(np.sqrt(np.mean(np.square(pitches)))),
            'mean_command_magnitude': float(np.mean(efforts)),
            'final_forward_displacement_m': info['forward_position_m'],
            'total_reward': total_reward,
        })
    summary = {
        'episodes': args.episodes,
        'balance_success_rate': float(np.mean([ep['success'] for ep in episodes])),
        'mean_duration_s': float(np.mean([ep['duration_s'] for ep in episodes])),
        'mean_pitch_rmse_deg': float(np.mean([ep['pitch_rmse_deg'] for ep in episodes])),
        'mean_abs_displacement_m': float(np.mean([abs(ep['final_forward_displacement_m']) for ep in episodes])),
        'mean_abs_motor_command': float(np.mean([ep['mean_command_magnitude'] for ep in episodes])),
        'per_episode': episodes,
    }
    (ROOT / 'results').mkdir(exist_ok=True)
    dest = ROOT / 'results/evaluation.json'
    dest.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_episode'}, indent=2))
    print('Saved:', dest)
    env.close()


if __name__ == '__main__':
    main()
