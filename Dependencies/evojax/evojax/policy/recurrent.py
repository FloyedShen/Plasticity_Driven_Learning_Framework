# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2023/1/13 14:43
# User      : Floyed
# Product   : PyCharm
# Project   : evojax
# File      : recurrent.py
# explain   :
# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Sequence, Tuple, Any, Callable
from functools import partial
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.linen.recurrent import RNNCellBase
from flax.struct import dataclass
from flax.core.frozen_dict import freeze
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn


@dataclass
class GruPolicyState(PolicyState):
    carrys: Sequence[jnp.ndarray]

@dataclass
class GruSnnPolicyState(PolicyState):
    carrys: Sequence[jnp.ndarray]
    variables_tree: Sequence[jnp.ndarray]

class GRU(nn.Module):
    feat_dims: Sequence[int]
    out_dim: int
    out_fn: str

    @nn.compact
    def __call__(self, carrys, x):
        new_carrys = []
        for hidden_dim, carry in zip(self.feat_dims, carrys):
            new_carry, x = nn.GRUCell()(carry, x)
            new_carrys.append(new_carry)
        x = nn.Dense(self.out_dim)(x)
        if self.out_fn == 'tanh':
            x = nn.tanh(x)
        elif self.out_fn == 'softmax':
            x = nn.softmax(x, axis=-1)
        elif self.out_fn == 'linear':
            pass
        else:
            raise ValueError(
                'Unsupported output activation: {}'.format(self.out_fn))
        return new_carrys, x


class GruWithMlp(nn.Module):
    feat_dims: Sequence[int]
    out_dim: int
    out_fn: str

    @nn.compact
    def __call__(self, carrys, x):
        new_carrys = []
        inputs = x
        for hidden_dim, carry in zip(self.feat_dims, carrys):
            new_carry, x = nn.GRUCell()(carry, x)
            new_carrys.append(new_carry)
        x = nn.Dense(self.out_dim)(x) + nn.Dense(self.out_dim)(inputs)
        if self.out_fn == 'tanh':
            x = nn.tanh(x)
        elif self.out_fn == 'softmax':
            x = nn.softmax(x, axis=-1)
        elif self.out_fn == 'linear':
            pass
        else:
            raise ValueError(
                'Unsupported output activation: {}'.format(self.out_fn))
        return new_carrys, x
