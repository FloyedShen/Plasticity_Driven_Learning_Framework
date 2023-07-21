# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2023/4/3 15:54
# User      : yu
# Product   : PyCharm
# Project   : evojax
# File      : seq_task.py
# explain   :

from functools import partial
import sys
import numpy as np
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState
from flax import linen as nn
from jax.experimental.host_callback import call, id_print


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    inputs: jnp.ndarray
    outputs: jnp.ndarray
    step: int


# def loss(action, target, mask):
#     action = nn.sigmoid(action)
#     return -jnp.mean(((action - target) ** 2) * mask)

def loss(y_pred, y_true, mask):
    # predicted_class = jnp.argmax(y_pred, axis=1)
    # return jnp.mean((predicted_class == y_true) * mask)
    y_true = nn.one_hot(y_true, num_classes=y_pred.shape[-1])
    ce_loss = jnp.sum(y_true * nn.log_softmax(y_pred), axis=-1)
    return (ce_loss * mask).mean()


def accuracy(y_pred: jnp.ndarray, y_true: jnp.ndarray, mask) -> jnp.float32:
    predicted_class = jnp.argmax(y_pred, axis=1)
    return jnp.mean((predicted_class == y_true) * mask)


class CueRewardTask(VectorizedTask):

    def __init__(
        self,
        batch_size: int = 1024,
        feature_dims: int = 14,
        seq_length: int = 20,
        num_cls: int = 5,
        sigma: float = 0.1,
        test: bool = False
    ):
        max_steps = seq_length
        self.max_steps = max_steps
        self.obs_shape = tuple((feature_dims + 2, ))
        self.act_shape = tuple((num_cls, ))
        # Delayed importing of torchvision

        def reset_fn(key):
            key, *rng = random.split(key, batch_size + 1)
            rng = jnp.stack(rng, axis=0)

            def generate(key):
                key, rng1, rng2, rng3, rng4 = random.split(key, 5)
                feature = random.randint(rng1, shape=(num_cls, feature_dims), minval=0, maxval=2)  # feature \in [0, 1]^d
                reward = random.choice(rng2, shape=(num_cls, 1), a=num_cls, replace=False)  # reward \in (0, 1)
                indices = random.choice(rng3, shape=(seq_length, ), a=num_cls)
                noise = random.normal(rng4, shape=(seq_length, feature_dims)) * sigma

                inputs_seq = jnp.where(jnp.take(feature, indices, axis=0) + noise > .5, 1., 0.)
                suffix_seq = jnp.concatenate([jnp.zeros((seq_length // 2, 1)), jnp.ones((seq_length // 2, 1))], axis=0)
                result_seq = jnp.take(reward, indices, axis=0)
                reward_seq = result_seq * (1 - suffix_seq)

                result = jnp.concatenate([
                    inputs_seq,
                    reward_seq,
                    suffix_seq,  # True -> calc loss
                    result_seq
                ], axis=1)

                return result

            seq = jax.vmap(generate)(rng)  # batch, seq, dim
            seq_input = seq[:, :, :-1]
            seq_output = seq[:, :, -1]
            return State(obs=seq_input[:, 0], step=0, inputs=seq_input, outputs=seq_output)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            if test:
                reward = accuracy(action, state.outputs[:, state.step], state.obs[:, -1]) / seq_length * 2
            else:
                reward = loss(action, state.outputs[:, state.step], state.obs[:, -1]) / seq_length * 2
            step = state.step + 1
            # id_print(step, tap_with_device=True)
            return state.replace(step=step, obs=state.inputs[:, step]), reward, jnp.array(step >= max_steps)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: TaskState,
             action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
