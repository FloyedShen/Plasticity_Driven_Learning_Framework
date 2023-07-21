# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2023/4/3 17:09
# User      : yu
# Product   : PyCharm
# Project   : evojax
# File      : seq_learning.py
# explain   :

import os
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

import evojax
from evojax.util import get_params_format_fn
from evojax.task.base import TaskState
from evojax.policy.base import PolicyState
from evojax.policy.base import PolicyNetwork
from flax.struct import dataclass
from flax import linen as nn

from evojax.task.seq_task import SeqTask
from evojax.task.cue_reward_task import CueRewardTask
from evojax.algo import PGPE, OpenES
from evojax import ObsNormalizer
from evojax import SimManager

from jax import random
import jax.numpy as jnp
import jax
from brax.io import html
from brax import jumpy as jp
from brax import envs
from IPython.display import HTML

import matplotlib.pyplot as plt
import mediapy as media
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
parser.add_argument('--policy', type=str, default='BatchedGruMetaStdpMLPPolicy')
parser.add_argument('--algo', type=str, default='PGPE')
parser.add_argument('--task', type=str, default='SeqTask')

parser.add_argument('--seq-length', type=int, default=20)
parser.add_argument('--latency', type=int, default=24)
parser.add_argument('--num-cls', type=int, default=4)
parser.add_argument('--feature-dims', type=int, default=14)
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--batch-size', type=int, default=512)

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
parser.add_argument('--max-iters', type=int, default=12000)
parser.add_argument('--num-tasks', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num-tests', type=int, default=128)
parser.add_argument('--eval-epoch', type=int, default=100)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--eval-with-injury', action='store_true')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--save', action='store_true')
parser.add_argument('--repeat', type=int, default=20)

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
        ('score_std', score_std),
        ('seq_length', args.seq_length),
        ('latency', args.latency),
        ('sigma', args.sigma),
        ('repeat_id', args.repeat_id),
    ])


def main():
    args, args_text = _parse_args()
    n_devices = jax.local_device_count()
    assert args.pop_size % n_devices == 0
    assert args.num_tests % n_devices == 0

    # Set Exp DIRs
    exp_name = '-'.join([
        args.algo,
        args.policy,
        args.task,
        args.suffix,
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    ])
    output_dir = get_outdir(args.root_dir, 'train', exp_name)
    args.output_dir = output_dir
    setup_default_logging(log_path=os.path.join(output_dir, 'log.txt'))
    summary_writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_dir, exp_name))
    args.summary_writer = summary_writer
    args.tb_prefix = os.path.join(args.algo, args.task, args.policy)
    _logger.info('[INFO] checkpoint saved to: {}'.format(args.output_dir))
    _logger.info('[INFO] tensorboard dir set to: {}'.format(os.path.join(args.tensorboard_dir, exp_name)))
    _logger.info('[ARGS]: {}'.format(args))
    # Create Envs

    if args.task == 'SeqTask':
        attr = ['seq_length', 'latency']
        val = [
            [8, 16, 32, 64],
            [0, 8, 16, 32]
        ]
    else:
        attr = ['sigma', 'seq_length']
        val = [
            [0, 0.3, 0.38, 0.48, 0.59],
            [16, 32, 48, 64],
        ]
    add_header = True
    for repeat in range(args.repeat):
        args.repeat_id = repeat
        for val0 in val[0]:
            for val1 in val[1]:
                setattr(args, attr[0], val0)
                setattr(args, attr[1], val1)
                args.seed += 1
                if args.task == 'SeqTask':
                    train_task = SeqTask(
                        seq_length=args.seq_length,
                        latency=args.latency,
                        batch_size=args.batch_size,
                        test=False
                    )
                    test_task = SeqTask(
                        seq_length=args.seq_length,
                        latency=args.latency,
                        batch_size=args.batch_size,
                        test=True
                    )
                elif args.task == 'CueRewardTask':
                    train_task = CueRewardTask(
                        batch_size=args.batch_size,
                        num_cls=args.num_cls,
                        feature_dims=args.feature_dims,
                        seq_length=args.seq_length,
                        sigma=args.sigma,
                        test=False
                    )
                    test_task = CueRewardTask(
                        batch_size=args.batch_size,
                        num_cls=args.num_cls,
                        feature_dims=args.feature_dims,
                        seq_length=args.seq_length,
                        sigma=args.sigma,
                        test=True
                    )
                else:
                    raise NotImplementedError('[Task] {}'.format(args.task))
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
                # obs_normalizer = ObsNormalizer(obs_shape=train_task.obs_shape)
                sim_mgr = SimManager(
                    n_repeats=1,
                    test_n_repeats=1,
                    pop_size=args.pop_size,
                    n_evaluations=args.num_tests,
                    policy_net=policy,
                    train_vec_task=train_task,
                    valid_vec_task=test_task,
                    seed=args.seed,
                    # obs_normalizer=obs_normalizer,
                    num_tasks=args.num_tasks
                )
                sim_mgr.tot_steps = steps
                if obs_params is not None:
                    sim_mgr.obs_params = obs_params

                if args.eval:
                    assert best_params is not None
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
                # Train
                try:
                    for train_iters in range(epoch, args.max_iters):
                        train_epoch(sim_mgr, solver, _logger, args)
                        if (train_iters > 0 and train_iters % args.eval_epoch == 0) or train_iters == args.max_iters - 1:
                            metrics = eval_epoch(sim_mgr, solver, train_iters, best_score, _logger, args)
                            best = metrics['score_avg'] > best_score
                            if (best or train_iters % 100) and args.save:
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
                                '[Len Lat Rep Sig]: {0:}, {1:}, {2:}, {3:}'
                                '[Train]: {4:}, '
                                'best={5:.2f}, '
                                'avg={6:.2f}, '
                                'std={7:.2f}, '
                                'steps={8:.3e}'.format(
                                    metrics['seq_length'],
                                    metrics['latency'],
                                    metrics['repeat_id'],
                                    metrics['sigma'],
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
                                           write_header=add_header)
                            add_header = False
                            best_score = max(best_score, metrics['score_avg'])
                except KeyboardInterrupt:
                    _logger.info('KeyboardInterrupt, Begin eval_with_injury.')
                    _logger.info('[OUTPUT DIR]: {}'.format(args.output_dir))
                _logger.info('[OUTPUT DIR]: {}'.format(args.output_dir))


# /data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-humanoid--20230116-005021
if __name__ == '__main__':
    main()
