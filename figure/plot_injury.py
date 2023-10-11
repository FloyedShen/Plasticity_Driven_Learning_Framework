#%%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import argparse
from collections import OrderedDict

import yaml
import csv
import time
from datetime import datetime
import shutil
import logging
from typing import Tuple, Sequence, Any, Union
from functools import partial

from jax import random
import jax.numpy as jnp
import jax
from brax.io import html
from brax import jumpy as jp
from brax import envs
from IPython.display import HTML

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import evojax
from evojax.util import get_params_format_fn
from evojax.task.base import TaskState
from evojax.policy.base import PolicyState
from evojax.policy.base import PolicyNetwork
from flax.struct import dataclass
from flax import linen as nn

from evojax.task.brax_task import BraxTask
from evojax.task.gym_task import Gym
from evojax.algo import PGPE, OpenES
from evojax import ObsNormalizer
from evojax import SimManager

sns.set_style('darkgrid')
sns.set_palette('muted', desat=.6)
sns.set_context("notebook", font_scale=1.0,
                rc={"lines.linewidth": 2.5})

output_dir = '/home/floyed/evojax/examples/metaplasticity_plot/injury'
#%%
def load_model(model_dir: str) -> Tuple[np.ndarray, np.ndarray, int, float, int]:
    model_file = os.path.join(model_dir, 'best.npz')
    if not os.path.exists(model_file):
        raise ValueError('Model file {} does not exist.')
    with np.load(model_file, allow_pickle=True) as data:
        params = data['params']
        obs_params = data['obs_params']
        epoch = data.get('epoch', 0)
        score = data.get('score', -float('Inf'))
        steps = data.get('steps', 0)
    return params, obs_params, epoch, score, steps
#%% md
### Temporal Injury
#%%
def eval_with_injury(policy, env_name, obs_normalizer, params, obs_params, target=None, injury=False, plot=True):
    # def step_once(carry, input_data):
    #     (task_state, policy_state, params, obs_params,
    #      accumulated_reward) = carry
    #     task_state = task_state.replace(
    #         obs=obs_norm_fn(task_state.obs[None, :], obs_params))
    #     act, policy_state = act_fn(task_state, params[None, :], policy_state)
    #     task_state = task_state.replace(obs=task_state.obs[0])
    #     task_state = env.step(task_state, act[0])
    #     policy_state = update_fn(params[None, :], policy_state, None)
    #     accumulated_reward = accumulated_reward + task_state.reward
    #
    #     return ((task_state, policy_state, params, obs_params, accumulated_reward), (task_state))
    if target is None:
        target = {}
    env_fn = envs.create_fn(
        env_name=env_name,
        legacy_spring=True,
        # terminate_when_unhealthy=args.terminate_when_unhealthy,
        **target
    )
    env = env_fn()
    # state = env.reset(rng=jp.random_prngkey(seed=0))
    task_reset_fn = jax.jit(env.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(env.step)
    act_fn = jax.jit(policy.get_actions)
    # update_fn = jax.jit(policy.update_fn) if hasattr(policy, 'update_fn') else lambda params, p_state, t_state: p_state
    obs_norm_fn = jax.jit(obs_normalizer.normalize_obs)

    total_reward = 0
    rollout = []
    rewards = []
    ws = []
    rng = jax.random.PRNGKey(42 + 1)
    task_state = task_reset_fn(rng=rng)
    policy_state = policy_reset_fn(task_state)

    # accumulated_rewards = jnp.zeros((1, ))
    # task_states = jax.lax.scan(
    #     step_once,
    #     (task_state, policy_state, params, obs_params, accumulated_rewards),
    #     (), 1000
    # )
    # print(task_states)

    step = 0
    while not task_state.done:
        rollout.append(task_state)
        task_state = task_state.replace(
            obs=obs_norm_fn(task_state.obs[None, :], obs_params))
        # jax.debug.breakpoint()
        act, policy_state = act_fn(task_state, params[None, :], policy_state)
        # For 50 steps, completely zero out the weights to see if the policy can
        # relearn weights that allow the ant to walk.
        if step > 500 and step < 550 and injury:
            policy_state.fast_Ws[1] *= 0

        # jax.debug.breakpoint()
        task_state = task_state.replace(obs=task_state.obs[0])
        task_state = step_fn(task_state, act[0])
        # task_state = env.step(task_state, act[0])
        # policy_state = update_fn(params[None, :], policy_state, None)
        total_reward = total_reward + task_state.reward
        rewards.append(task_state.reward)
        # Record the weights every 10 steps for visualisations.
        if hasattr(policy_state, 'fast_Ws') and step % 10 == 0:
            ws.append(policy_state.fast_Ws)
        step += 1

        if step % 100 == 0 and plot:
            print('[Eval with injury] {}, reward:{:.3e}'.format(step, total_reward))
    print('[Eval with injury] rollout reward = {}'.format(total_reward))

    if plot:
        ws = list(zip(*ws))
        for i, w in enumerate(ws):
            w = np.array(jnp.concatenate(w)[jnp.newaxis, :, jnp.newaxis, :, :])
            # TODO: w

        html.save_html(os.path.join(output_dir, 'eval_with_injury_{}.html'.format(target)), env.sys, [s.qp for s in rollout])

        fig, ax = plt.subplots()
        ax.plot(rewards)
        plt.savefig(os.path.join(output_dir, 'BraxMetaLearning_{}.pdf').format(target))

    return rewards, rollout


def eval_with_injury_base(policy, env_name, obs_normalizer, params, obs_params, target=None, injury=False, plot=True):
    # def step_once(carry, input_data):
    #     (task_state, policy_state, params, obs_params,
    #      accumulated_reward) = carry
    #     task_state = task_state.replace(
    #         obs=obs_norm_fn(task_state.obs[None, :], obs_params))
    #     act, policy_state = act_fn(task_state, params[None, :], policy_state)
    #     task_state = task_state.replace(obs=task_state.obs[0])
    #     task_state = env.step(task_state, act[0])
    #     policy_state = update_fn(params[None, :], policy_state, None)
    #     accumulated_reward = accumulated_reward + task_state.reward
    #
    #     return ((task_state, policy_state, params, obs_params, accumulated_reward), (task_state))
    if target is None:
        target = {}
    env_fn = envs.create_fn(
        env_name=env_name,
        legacy_spring=True,
        # terminate_when_unhealthy=args.terminate_when_unhealthy,
        **target
    )
    env = env_fn()
    # state = env.reset(rng=jp.random_prngkey(seed=0))
    task_reset_fn = jax.jit(env.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(env.step)
    act_fn = jax.jit(policy.get_actions)
    # update_fn = jax.jit(policy.update_fn) if hasattr(policy, 'update_fn') else lambda params, p_state, t_state: p_state
    obs_norm_fn = jax.jit(obs_normalizer.normalize_obs)

    total_reward = 0
    rollout = []
    rewards = []
    ws = []
    rng = jax.random.PRNGKey(42 + 1)
    task_state = task_reset_fn(rng=rng)
    policy_state = policy_reset_fn(task_state)

    # accumulated_rewards = jnp.zeros((1, ))
    # task_states = jax.lax.scan(
    #     step_once,
    #     (task_state, policy_state, params, obs_params, accumulated_rewards),
    #     (), 1000
    # )
    # print(task_states)

    step = 0
    while not task_state.done:
        rollout.append(task_state)
        task_state = task_state.replace(
            obs=obs_norm_fn(task_state.obs[None, :], obs_params))
        # jax.debug.breakpoint()
        act, policy_state = act_fn(task_state, params[None, :], policy_state)
        # For 50 steps, completely zero out the weights to see if the policy can
        # relearn weights that allow the ant to walk.
        if step > 500 and step < 550 and injury:
            params *= 0
            # policy_state.fast_Ws[1] *= 0

        # jax.debug.breakpoint()
        task_state = task_state.replace(obs=task_state.obs[0])
        task_state = step_fn(task_state, act)
        # task_state = env.step(task_state, act[0])
        # policy_state = update_fn(params[None, :], policy_state, None)
        total_reward = total_reward + task_state.reward
        rewards.append(task_state.reward)
        # Record the weights every 10 steps for visualisations.
        if hasattr(policy_state, 'fast_Ws') and step % 10 == 0:
            ws.append(policy_state.fast_Ws)
        step += 1

        if step % 100 == 0 and plot:
            print('[Eval with injury] {}, reward:{:.3e}'.format(step, total_reward))
    print('[Eval with injury] rollout reward = {}'.format(total_reward))

    if plot:
        ws = list(zip(*ws))
        for i, w in enumerate(ws):
            w = np.array(jnp.concatenate(w)[jnp.newaxis, :, jnp.newaxis, :, :])
            # TODO: w

        html.save_html(os.path.join(output_dir, 'eval_with_injury_{}.html'.format(target)), env.sys, [s.qp for s in rollout])

        fig, ax = plt.subplots()
        ax.plot(rewards)
        plt.savefig(os.path.join(output_dir, 'BraxMetaLearning_{}.pdf').format(target))

    return rewards, rollout
#%%
env_names = ['ant_dir', 'halfcheetah_vel', 'fetch']
num_tasks = 8
hidden_dims = [128]

resume_meta = [
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-ant_dir-ABCD-20230622-192211',
    # '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-swimmer_dir-ABCD-20230622-192221',
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-halfcheetah_vel-ABCD-20230622-192203',
    # '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-hopper_vel-ABCD-20230622-192225',
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-fetch-ABCD-20230622-212210',
    # '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-ur5e-ABCD-20230622-192216',
]

resume_base = [
    '/data/floyed/meta/train/PGPE-MLPSnnPolicy-ant_dir--20230622-113608',
    # '/data/floyed/meta/train/PGPE-MLPSnnPolicy-swimmer_dir--20230622-113637',
    '/data/floyed/meta/train/PGPE-MLPSnnPolicy-halfcheetah_vel--20230622-113532',
    # '/data/floyed/meta/train/PGPE-MLPSnnPolicy-hopper_vel--20230622-113649',
    '/data/floyed/meta/train/PGPE-MLPSnnPolicy-fetch--20230622-132230',
    # '/data/floyed/meta/train/PGPE-MLPSnnPolicy-ur5e--20230622-113621',
]
#%%
targets = [0, 5, None]
y_bases, y_metas = [], []

for idx, env in enumerate(env_names):
    print(env)
    train_task = BraxTask(
        env_name=env,
        test=False,
        num_tasks=num_tasks
        # terminate_when_unhealthy=args.terminate_when_unhealthy
    )
    policy_meta = getattr(evojax.policy, 'MetaStdpMLPPolicy')(
        input_dim=train_task.obs_shape[0],
        hidden_dims=[
            *hidden_dims,
        ],
        output_dim=train_task.act_shape[0],
    )

    policy_base = getattr(evojax.policy, 'MLPSnnPolicy')(
        input_dim=train_task.obs_shape[0],
        hidden_dims=[
            *hidden_dims,
        ],
        output_dim=train_task.act_shape[0],
    )

    meta_best_params, meta_obs_params, meta_epoch, meta_best_score, meta_steps = load_model(model_dir=resume_meta[idx])
    base_best_params, base_obs_params, base_epoch, base_best_score, base_steps = load_model(model_dir=resume_base[idx])
    meta_obs_normalizer = ObsNormalizer(obs_shape=train_task.obs_shape, task_info_holder=train_task.task_info_holder)
    base_obs_normalizer = ObsNormalizer(obs_shape=train_task.obs_shape, task_info_holder=train_task.task_info_holder)

    x, y_meta, y_base = [], [], []
    if targets[idx] is None:
        meta_rewards, meta_rollout = \
            eval_with_injury(policy_meta, env, meta_obs_normalizer, meta_best_params, meta_obs_params, target=None, injury=True, plot=False)
        base_rewards, base_rollout = \
            eval_with_injury_base(policy_base, env, base_obs_normalizer, base_best_params, base_obs_params, target=None, injury=True, plot=False)
        y_metas.append(meta_rewards)
        y_bases.append(base_rewards)

    else:
        _target = targets[idx]
        meta_rewards, meta_rollout = \
            eval_with_injury(policy_meta, env, meta_obs_normalizer, meta_best_params, meta_obs_params, target={train_task.task_kwargs: _target}, injury=True, plot=False)
        base_rewards, base_rollout = \
            eval_with_injury_base(policy_base, env, base_obs_normalizer, base_best_params, base_obs_params, target={train_task.task_kwargs: _target}, injury=True, plot=False)
        y_metas.append(sum(meta_rewards))
        y_bases.append(sum(base_rewards))
#%%
fig = plt.figure(figsize=(9, 3), dpi=100, constrained_layout=True)
spec = fig.add_gridspec(nrows=1, ncols=3, figure=fig)  #, width_ratios=[.2, 1, 1, 1, 1, .08, 1.6], height_ratios=[0.02, 1, 1, 0.01])
