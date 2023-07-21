# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2023/1/10 13:36
# User      : Floyed
# Product   : PyCharm
# Project   : evojax
# File      : mlp_snn.py
# explain   :

import logging
from typing import Sequence
from typing import Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.struct import dataclass

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn


@dataclass
class MplsPolicyState(PolicyState):
    variables_tree: Sequence[jnp.ndarray]


class LifNeuron(nn.Module):
    neu_kwargs: dict
    # v_th: float = .5
    v_th: float = 1.  # default
    v_reset: float = 0.
    tau: float = .5

    @nn.compact
    def __call__(self, inputs, key=random.PRNGKey(0)):
        mem = self.variable(
            'mem_status',
            'mem',
            lambda s: jnp.zeros(s),
            inputs.shape
        )
        # spike = self.variable(
        #     'mem_status',
        #     'spike',
        #     lambda s: jnp.zeros(s),
        #     inputs.shape
        # )
        # mem.value = inputs
        # mem.value = mem.value + (inputs - mem.value) / self.tau
        mem.value = mem.value * self.tau + inputs * (1 - self.tau)
        sk = (mem.value > self.v_th).astype(float) - (mem.value < -self.v_th).astype(float)
        mem.value = jnp.where(jnp.abs(sk) > 1e-5, jnp.zeros_like(mem.value), mem.value)

        # spike.value = sk
        # sk = (mem.value > self.v_th).astype(float)
        # mem.value = jnp.where(sk > 1e-5, jnp.zeros_like(mem.value), mem.value)
        return sk

class MLPs(nn.Module):
    feat_dims: Sequence[int]
    out_dim: int
    out_fn: str
    Neuron: nn.Module
    neu_kwargs: dict = None

    @nn.compact
    def __call__(self, x, key=random.PRNGKey(0)):
        for hidden_dim in self.feat_dims:
            rng, key = random.split(key, 2)
            x = self.Neuron(self.neu_kwargs)(nn.Dense(hidden_dim)(x), rng)
        x = nn.Dense(self.out_dim)(x)
        # x = nn.Dense(self.out_dim)(x)
        if self.out_fn == 'tanh':
            x = nn.tanh(x)
        elif 'softmax' in self.out_fn:
            x = nn.softmax(x, axis=-1)
        elif self.out_fn == 'linear':
            pass
        else:
            raise ValueError(
                'Unsupported output activation: {}'.format(self.out_fn))
        if 'select' in self.out_fn:
            x = jnp.argmax(x, axis=-1)
        return x


class MLPSnnPolicy(PolicyNetwork):
    """A general purpose multi-layer perceptron model."""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int],
                 output_dim: int,
                 output_act_fn: str = 'tanh',
                 logger: logging.Logger = None,
                 **kwargs):
        if logger is None:
            self._logger = create_logger(name='MLPPolicy')
        else:
            self._logger = logger

        model = MLPs(
            feat_dims=hidden_dims,
            out_dim=output_dim,
            out_fn=output_act_fn,
            Neuron=LifNeuron,
            neu_kwargs={'tau': 'param'},
        )
        variables = model.init(random.PRNGKey(0), jnp.ones([1, input_dim]))
        mem_status, params = variables.pop('params')
        self.num_params, format_params_fn = get_params_format_fn(params)
        self.num_var, format_var_fn = get_params_format_fn(mem_status)

        # self.model = model
        # self.variables = variables
        # self.mem_status = mem_status
        # self.params = params

        self._logger.info('MLPPolicy.num_params = {}'.format(self.num_params))
        self._format_params_fn = jax.vmap(format_params_fn)
        self._format_var_fn = jax.vmap(format_var_fn)
        self._forward_fn = jax.vmap(partial(model.apply, mutable=mem_status.keys()))

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        # print('GETACTION GETACTION GETACTION GETACTION')
        variables_tree = p_states.variables_tree
        params = freeze({'params': params, **variables_tree})
        actions, new_variables_tree = self._forward_fn(params, t_states.obs)

        return actions.squeeze(), p_states.replace(variables_tree=new_variables_tree)

    def reset(self, states: TaskState) -> PolicyState:
        keys = jax.random.split(jax.random.PRNGKey(0), states.obs.shape[0])

        pop = states.obs.shape[0] if len(states.obs.shape) > 1 else 1
        variables = jnp.array([jnp.zeros(self.num_var)] * pop)
        variables_tree = self._format_var_fn(variables)
        return MplsPolicyState(
            keys=keys,
            variables_tree=variables_tree
        )


class BatchedMLPSnnPolicy(PolicyNetwork):
    """A general purpose multi-layer perceptron model."""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int],
                 output_dim: int,
                 output_act_fn: str = 'tanh',
                 logger: logging.Logger = None,
                 **kwargs):
        if logger is None:
            self._logger = create_logger(name='MLPPolicy')
        else:
            self._logger = logger

        model = MLPs(
            feat_dims=hidden_dims,
            out_dim=output_dim,
            out_fn=output_act_fn,
            Neuron=LifNeuron,
        )
        self.model = model
        variables = model.init(random.PRNGKey(0), jnp.ones([1, input_dim]))
        mem_status, params = variables.pop('params')
        self.num_params, format_params_fn = get_params_format_fn(params)
        self.num_var, format_var_fn = get_params_format_fn(mem_status)
        self.variables_tree = None

        # self.model = model
        # self.variables = variables
        # self.mem_status = mem_status
        # self.params = params

        self._logger.info('MLPPolicy.num_params = {}'.format(self.num_params))
        self._format_params_fn = jax.vmap(format_params_fn)
        self._format_var_fn = jax.vmap(format_var_fn)
        self._forward_fn = jax.vmap(partial(model.apply, mutable=mem_status.keys()))

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: MplsPolicyState) -> Tuple[jnp.ndarray, MplsPolicyState]:
        params = self._format_params_fn(params)
        # print('GETACTION GETACTION GETACTION GETACTION')
        variables_tree = p_states.variables_tree
        params = freeze({'params': params, **variables_tree})
        actions, new_variables_tree = self._forward_fn(params, t_states.obs)

        # call(lambda x: print(f"mem_1: {x}"), self.variables_tree['mem_status']['LifNeuron_0']['mem'][0])
        # call(lambda x: print(f"mem_2: {x}"), self.variables_tree['mem_status']['LifNeuron_0']['mem'][1])
        # call(lambda x: print(f"mem_3: {x}"), self.variables_tree['mem_status']['LifNeuron_0']['mem'][2])
        # call(lambda x: print(f"actions: {x}"), actions)
        return actions, p_states.replace(variables_tree=new_variables_tree)

    def reset(self, states: TaskState) -> PolicyState:
        keys = jax.random.split(jax.random.PRNGKey(0), states.obs.shape[0])

        pop = states.obs.shape[0]

        variables = self.model.init(random.PRNGKey(0), jnp.zeros(states.obs.shape[1:]))
        mem_status, params = variables.pop('params')
        num_var, format_var_fn = get_params_format_fn(mem_status)
        _format_var_fn = jax.vmap(format_var_fn)

        variables = jnp.array([jnp.zeros(num_var)] * pop)
        variables_tree = _format_var_fn(variables)
        return MplsPolicyState(
            keys=keys,
            variables_tree=variables_tree
        )


if __name__ == '__main__':
    key = random.PRNGKey(42)
    model = LifNeuron()

    key, subkey = random.split(key)
    variables = model.init(subkey, jnp.zeros(5, ))
    # mem_status, params = variables.pop('params')
    for t in range(16):
        key, subkey = random.split(key)
        x = random.uniform(subkey, shape=(5,))
        spike, variables = model.apply(variables, x, mutable=variables.keys())
        # variables = freeze({'params': params, **mem_status})
        print('x: {}'.format(x))
        print('mem: {}'.format(variables['mem_status']['mem']))
        print('spk: {}'.format(spike))
