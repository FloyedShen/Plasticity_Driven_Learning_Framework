# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2023/1/13 16:52
# User      : Floyed
# Product   : PyCharm
# Project   : evojax
# File      : meta_learning.py
# explain   :
# /data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-ant_dir-ABCD-20230625-001333
import os
import argparse
from collections import OrderedDict

import yaml
import csv
from datetime import datetime
import logging
from typing import Tuple, Sequence, Any, Union

import evojax
from evojax.policy.base import PolicyState

from evojax.task.brax_task import BraxTask
from evojax.algo import PGPE
from evojax import ObsNormalizer
from evojax import SimManager

import jax.numpy as jnp
import jax
from brax.io import html
from brax import envs

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from timm.utils import get_outdir
from timm.utils.log import setup_default_logging

_logger = logging.getLogger('meta learning')
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)

parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Meta Hebbian Training and Evaluating')

# Key parameters
parser.add_argument('--env', type=str, default='ant')
parser.add_argument('--policy', type=str, default='MetaMLPPolicy')
parser.add_argument('--algo', type=str, default='PGPE')

# Policy parameters
parser.add_argument('--hidden-dims', nargs='+', type=int, default=[128])

# Evo parameters
parser.add_argument('--pop-size', type=int, default=256)
parser.add_argument('--center-lr', type=float, default=0.01)
parser.add_argument('--init-std', type=float, default=0.04)
parser.add_argument('--decay-std', type=float, default=0.999)
parser.add_argument('--limit-std', type=float, default=0.001)
parser.add_argument('--std-lr', type=float, default=0.07)
parser.add_argument('--terminate-when-unhealthy', action='store_true')

# Training parameters
parser.add_argument('--max-iters', type=int, default=300000)
parser.add_argument('--num-tasks', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num-tests', type=int, default=128)
parser.add_argument('--eval-epoch', type=int, default=10)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--eval-with-injury', action='store_true')
parser.add_argument('--resume', type=str, default='')

# Checkpoint parameters
parser.add_argument('--root-dir', type=str, default='/data/floyed/meta')
parser.add_argument('--tensorboard-dir', type=str, default='./runs')
parser.add_argument('--suffix', type=str, default='')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def update_summary(epoch, metrics, filename, write_header=False, log_wandb=False):
    rowd = OrderedDict(epoch=epoch)
    # rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([(k, v) for k, v in metrics.items()])
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)


# def save_weight_as_gif(ws, output_dir):
#     for i, w in enumerate(ws):
#         w = jnp.concatenate(w)
#         w += 1.0
#         w /= 2.0
#         media.show_video(w, height=w.shape[1] * 4, codec='gif', fps=5)
#         media.write_video(os.path.join(output_dir, '{}.gif'.format(i)), w, codec='gif', fps=5)


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


def save_model(model_dir: str,
               model_name: str,
               params: Union[np.ndarray, jnp.ndarray],
               obs_params: Union[np.ndarray, jnp.ndarray] = None,
               epoch: int = -1,
               score: float = -float('Inf'),
               steps: int = -1,
               best: bool = False) -> None:
    model_file = os.path.join(model_dir, '{}.npz'.format(model_name))
    np.savez(
        model_file,
        params=np.array(params),
        obs_params=np.array(obs_params),
        epoch=np.array(epoch),
        score=np.array(score),
        steps=np.array(steps)
    )
    if best:
        model_file = os.path.join(model_dir, 'best.npz')
        np.savez(
            model_file,
            params=np.array(params),
            obs_params=np.array(obs_params),
            epoch=epoch,
            score=score,
            steps=steps
        )


def train_epoch(sim_mgr, solver, _logger, args):
    # Training.
    params = solver.ask()
    scores = 0
    for i in range(args.num_tasks):
        score, _ = sim_mgr.eval_params(params=params, test=False)
        # jax.debug.breakpoint()
        scores += score
    solver.tell(fitness=scores)
    return


def eval_epoch(sim_mgr, solver, train_iters, best_score, _logger, args):
    best_params = solver.best_params
    scores = np.array(sim_mgr.eval_params(params=best_params, test=True)[0])
    score_best = scores.max()
    score_avg = np.mean(scores)
    score_std = np.std(scores)
    return OrderedDict([
        ('score_best', score_best),
        ('score_avg', score_avg),
        ('score_std', score_std)
    ])


def eval_with_injury(policy, env_name, obs_normalizer, params, obs_params, _logger, args, target=None):
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
    rng = jax.random.PRNGKey(seed=args.seed + 1)
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
        # if step > 500 and step < 550:
        #     policy_state.fast_Ws[1] *= 0

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

        if step % 100 == 0:
            _logger.info('[Eval with injury] {}, reward:{:.3e}'.format(step, total_reward))
    _logger.info('[Eval with injury] rollout reward = {}'.format(total_reward))

    ws = list(zip(*ws))
    for i, w in enumerate(ws):
        w = np.array(jnp.concatenate(w)[jnp.newaxis, :, jnp.newaxis, :, :])
        args.summary_writer.add_video(os.path.join(args.tb_prefix, 'eval_with_injury/layer_{}'.format(i)), w)

    html.save_html(os.path.join(args.output_dir, 'eval_with_injury_{}.html'.format(target)), env.sys, [s.qp for s in rollout])
    args.summary_writer.add_text(os.path.join(args.tb_prefix, 'eval_with_injury_{}'.format(target)),
                                 'HTML dir: {}'.format(os.path.join(args.output_dir, 'eval_with_injury_{}.html').format(target)))

    fig, ax = plt.subplots()
    ax.plot(rewards)
    args.summary_writer.add_figure(os.path.join(args.tb_prefix, 'eval_with_injury_{}'.format(target)), fig, close=False)
    plt.savefig(os.path.join(args.output_dir, 'BraxMetaLearning_{}.pdf').format(target))

    return


def main():
    args, args_text = _parse_args()
    n_devices = jax.local_device_count()
    assert args.pop_size % n_devices == 0
    assert args.num_tests % n_devices == 0

    # Set Exp DIRs
    exp_name = '-'.join([
        args.algo,
        args.policy,
        args.env,
        args.suffix,
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    ])
    output_dir = get_outdir(args.root_dir, 'train', exp_name)
    args.output_dir = output_dir
    setup_default_logging(log_path=os.path.join(output_dir, 'log.txt'))
    summary_writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_dir, exp_name))
    args.summary_writer = summary_writer
    args.tb_prefix = os.path.join(args.algo, args.env, args.policy)
    _logger.info('[INFO] checkpoint saved to: {}'.format(args.output_dir))
    _logger.info('[INFO] tensorboard dir set to: {}'.format(os.path.join(args.tensorboard_dir, exp_name)))
    _logger.info('[ARGS]: {}'.format(args))
    # Create Envs
    train_task = BraxTask(
        env_name=args.env,
        test=False,
        num_tasks=args.num_tasks
        # terminate_when_unhealthy=args.terminate_when_unhealthy
    )
    test_task = BraxTask(
        env_name=args.env,
        test=True,
        num_tasks=args.num_tasks 
        # terminate_when_unhealthy=args.terminate_when_unhealthy
    )
    # train_task = Gym('HumanoidStandup-v4', pop_size=args.pop_size, test=False)
    # test_task = Gym('HumanoidStandup-v4', pop_size=args.pop_size, test=True)

    # Create  Policy
    # policy = getattr(evojax.policy, args.policy)(
    #     hidden_dims=[
    #         train_task.obs_shape[0],
    #         *args.hidden_dims,
    #         train_task.act_shape[0]
    #     ]
    # )
    policy = getattr(evojax.policy, args.policy)(
        input_dim=train_task.obs_shape[0],
        hidden_dims=[
            *args.hidden_dims,
        ],
        output_dim=train_task.act_shape[0],
    )

    _logger.info('[Total Params]: params={}'.format(policy.num_params))

    best_score = -float('Inf')
    epoch = 0
    best_params = None
    obs_params = None
    steps = 0
    if args.resume != '':
        best_params, obs_params, epoch, best_score, steps = load_model(model_dir=args.resume)

    # Create Solver
    solver = getattr(evojax.algo, args.algo)(
        pop_size=args.pop_size,
        param_size=policy.num_params,
        optimizer='adam',
        center_learning_rate=args.center_lr,
        stdev_learning_rate=args.std_lr,
        init_stdev=args.init_std,
        seed=args.seed,
        init_params=best_params,
    )

    # solver = getattr(evojax.algo, args.algo)(
    #     pop_size=args.pop_size,
    #     elite_ratio=1.,
    #     param_size=policy.num_params,
    #     optimizer='adam',
    #     init_stdev=args.init_std,
    #     decay_stdev=args.decay_std,
    #     limit_stdev=0.01,
    #     seed=args.seed,
    # )

    if best_params is not None:  # FOR OpenES
        solver.best_params = best_params

    # Sim Manager
    obs_normalizer = ObsNormalizer(obs_shape=train_task.obs_shape, task_info_holder=train_task.task_info_holder)
    sim_mgr = SimManager(
        n_repeats=1,
        test_n_repeats=1,
        pop_size=args.pop_size,
        n_evaluations=args.num_tests,
        policy_net=policy,
        train_vec_task=train_task,
        valid_vec_task=test_task,
        seed=args.seed,
        obs_normalizer=obs_normalizer,
        num_tasks=args.num_tasks
    )
    sim_mgr.tot_steps = steps
    if obs_params is not None:
        sim_mgr.obs_params = obs_params

    if args.eval:
        # assert best_params is not None
        metrics = eval_epoch(sim_mgr, solver, -1, best_score, _logger, args)
        _logger.info(
            '[EVAL]: {0:}, '
            'best={1:.2f}, '
            'avg={2:.2f}, '
            'std={3:.2f}, '
            'steps={4:.3e}'.format(
                -1,
                metrics['score_best'],
                metrics['score_avg'],
                metrics['score_std'],
                sim_mgr.tot_steps
            )
        )
    if args.eval_with_injury:
        if train_task.task_info_holder != 0:
            for i in range(12):
                target = (1 - i / 12) * train_task.task_min + i / 12 * train_task.task_max
                eval_with_injury(policy, args.env, obs_normalizer, best_params, obs_params,
                                 _logger, args, target={train_task.task_kwargs: target})
        else:
            eval_with_injury(policy, args.env, obs_normalizer, best_params, obs_params, _logger, args)
        exit(0)

    # Train
    try:
        for train_iters in range(epoch, args.max_iters):
            train_epoch(sim_mgr, solver, _logger, args)

            if (train_iters > 0 and train_iters % args.eval_epoch == 0) or train_iters == args.max_iters - 1:
                metrics = eval_epoch(sim_mgr, solver, train_iters, best_score, _logger, args)
                best = metrics['score_avg'] > best_score
                if best or train_iters % 100:
                    save_model(
                        model_dir=args.output_dir,
                        model_name='iter_{}'.format(train_iters),
                        params=solver.best_params,
                        obs_params=sim_mgr.obs_params,
                        epoch=epoch,
                        score=metrics['score_avg'],
                        steps=sim_mgr.tot_steps,
                        best=best
                    )

                _logger.info(
                    '[Train]: {0:}, '
                    'best={1:.2f}, '
                    'avg={2:.2f}, '
                    'std={3:.2f}, '
                    'steps={4:.3e}'.format(
                        train_iters,
                        metrics['score_best'],
                        metrics['score_avg'],
                        metrics['score_std'],
                        sim_mgr.tot_steps
                    )
                )
                summary_writer.add_scalar(os.path.join(args.tb_prefix, 'episode/score_best'),
                                          metrics['score_best'], train_iters)
                summary_writer.add_scalar(os.path.join(args.tb_prefix, 'episode/score_avg'),
                                          metrics['score_avg'], train_iters)
                summary_writer.add_scalar(os.path.join(args.tb_prefix, 'episode/score_std'),
                                          metrics['score_std'], train_iters)
                summary_writer.add_scalar(os.path.join(args.tb_prefix, 'frame/score_best'),
                                          metrics['score_best'], sim_mgr.tot_steps)
                summary_writer.add_scalar(os.path.join(args.tb_prefix, 'frame/score_avg'),
                                          metrics['score_avg'], sim_mgr.tot_steps)
                summary_writer.add_scalar(os.path.join(args.tb_prefix, 'frame/score_std'),
                                          metrics['score_std'], sim_mgr.tot_steps)

                update_summary(train_iters, metrics, os.path.join(args.output_dir, 'summary.csv'),
                               write_header=(best_score == -float('Inf')))
                best_score = max(best_score, metrics['score_avg'])
    except KeyboardInterrupt:
        _logger.info('KeyboardInterrupt, Begin eval_with_injury.')
        _logger.info('[OUTPUT DIR]: {}'.format(args.output_dir))
    for i in range(12):
        target = (1 - i / 12) * train_task.task_min + i / 12 * train_task.task_max
        eval_with_injury(policy, args.env, obs_normalizer, best_params, sim_mgr.obs_params,
                         _logger, args, target={train_task.task_kwargs: target})
    # eval_with_injury(policy, args.env, obs_normalizer, solver.best_params, sim_mgr.obs_params, _logger, args)
    _logger.info('[OUTPUT DIR]: {}'.format(args.output_dir))


# /data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-humanoid--20230116-005021
if __name__ == '__main__':
    main()
