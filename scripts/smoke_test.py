#!/usr/bin/env python3
"""Check MJCF compilation, expected structure, reset, zero-torque trajectory."""
import argparse
from pathlib import Path
import sys
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.pivot_balance_env import PivotBalanceEnv, TRAIN_XML, VISUAL_XML


def check(path):
    env = PivotBalanceEnv(xml_path=path)
    print(f'MODEL {path.name}: nq={env.model.nq}, nv={env.model.nv}, '
          f'nu={env.model.nu}, nsensor={env.model.nsensor}')
    assert env.model.nq == 9, 'Expected 7 freejoint + 2 wheel qpos'
    assert env.model.nv == 8, 'Expected 6 freejoint + 2 wheel velocities'
    assert env.model.nu == 2, 'Expected one motor per wheel'
    obs, info = env.reset(seed=4, options={'pitch': np.deg2rad(7), 'pitch_rate': 0})
    assert obs.shape == (7,)
    assert np.isfinite(obs).all()
    pitch0 = float(obs[0])
    contacts = env.data.ncon
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.zeros(1))
        assert np.isfinite(obs).all()
        if terminated or truncated:
            break
    print(f' Initial tilt = {np.rad2deg(pitch0):.2f} deg; '
          f'after {info["elapsed_seconds"]:.3f}s tilt = '
          f'{info["pitch_degrees"]:.2f} deg; '
          f'contacts at start = {contacts}; terminated={terminated}')
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visual', action='store_true', help='Also compile high-detail mesh model')
    args = parser.parse_args()
    check(TRAIN_XML)
    if args.visual:
        check(VISUAL_XML)
    print('Smoke test passed (not a guarantee of physical fidelity).')


if __name__ == '__main__':
    main()
