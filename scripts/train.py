#!/usr/bin/env python3
"""Train PPO on PIVOT-Reflex. Example: python scripts/train.py --timesteps 1000000"""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.pivot_balance_env import PivotBalanceEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    (ROOT / 'checkpoints').mkdir(exist_ok=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    env = PivotBalanceEnv()
    check_env(env, warn=True)
    env = Monitor(env)
    evaluation = Monitor(PivotBalanceEnv())
    callback = EvalCallback(
        evaluation, best_model_save_path=str(ROOT / 'checkpoints'),
        log_path=str(ROOT / 'logs'), eval_freq=10_000,
        n_eval_episodes=10, deterministic=True, verbose=1)
    policy = PPO('MlpPolicy', env, verbose=1, seed=args.seed,
                 learning_rate=3e-4, n_steps=2048, batch_size=256,
                 n_epochs=10, gamma=0.99, gae_lambda=0.95,
                 clip_range=0.2, ent_coef=0.002,
                 policy_kwargs={'net_arch': [64, 64]},
                 tensorboard_log=str(ROOT / 'logs'), device='cpu')
    policy.learn(total_timesteps=args.timesteps, callback=callback,
                 tb_log_name='PPO-PIVOT')
    policy.save(str(ROOT / 'checkpoints/pivot_ppo_final'))
    evaluation.close()
    env.close()
    print('Saved: checkpoints/pivot_ppo_final.zip')
    print('Best eval model (if evaluated): checkpoints/best_model.zip')


if __name__ == '__main__':
    main()
