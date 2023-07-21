# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2023/1/13 16:29
# User      : Floyed
# Product   : PyCharm
# Project   : evojax
# File      : stdp.py
# explain   :

import logging
from functools import partial
from typing import Sequence
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.struct import dataclass

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.policy.recurrent import GRU
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn

# _resi = 5e7
# _delta_t = 1e-3
# _tau = 5e-3


@partial(jax.jit, static_argnums=(2, 3))
def lif_neuron(mem_status, inputs, tau=jnp.e, v_th=.1):
    mem = inputs + (mem_status - inputs) / tau
    # mem = mem_status + (_delta_t / _tau) * (- mem_status + inputs * _resi)
    spike = (mem > v_th).astype(float) - (mem < -v_th).astype(float)
    # spike = nn.tanh(mem)
    # mem = jnp.where(jnp.logical_or(spike >= v_th, spike <= -v_th), jnp.zeros_like(mem), mem)
    mem = jnp.where(jnp.abs(spike) > 1e-5, jnp.zeros_like(mem), mem)
    # mem = mem - v_th * spike
    return mem, spike


@dataclass
class MetaStdpMLPPolicyState(PolicyState):
    # Flag to indicate if this is the first step in the policy
    first: jnp.ndarray
    # Since we update the wights using the hebbian rules each time the policy
    # takes a step, these 'fast weights' need to be stored in the policy state.
    fast_Ws: Sequence[jnp.ndarray]

    mem_status: Sequence[jnp.ndarray]
    traces: Sequence[jnp.ndarray]

@dataclass
class BaseMLPSnnPolicyState(PolicyState):
    mem_status: Sequence[jnp.ndarray]

@dataclass
class MetaGruStdpMLPPolicyState(PolicyState):
    # Flag to indicate if this is the first step in the policy
    first: jnp.ndarray
    # Since we update the wights using the hebbian rules each time the policy
    # takes a step, these 'fast weights' need to be stored in the policy state.
    fast_Ws: Sequence[jnp.ndarray]

    mem_status: Sequence[jnp.ndarray]
    traces: Sequence[jnp.ndarray]
    carrys: Sequence[jnp.ndarray]

@dataclass
class GruSnnPolicyState(PolicyState):
    # Flag to indicate if this is the first step in the policy
    first: jnp.ndarray
    mem_status: Sequence[jnp.ndarray]
    carrys: Sequence[jnp.ndarray]
    traces: Sequence[jnp.ndarray]

@dataclass
class MetaStdpRecurrentPolicyState(PolicyState):
    # Flag to indicate if this is the first step in the policy
    first: jnp.ndarray
    # Since we update the wights using the hebbian rules each time the policy
    # takes a step, these 'fast weights' need to be stored in the policy state.
    fast_Ws: Sequence[jnp.ndarray]

    mem_status: Sequence[jnp.ndarray]
    traces: Sequence[jnp.ndarray]
    spikes: jnp.ndarray

@dataclass
class MetaGruStdpRecurrentPolicyState(PolicyState):
    # Flag to indicate if this is the first step in the policy
    first: jnp.ndarray
    # Since we update the wights using the hebbian rules each time the policy
    # takes a step, these 'fast weights' need to be stored in the policy state.
    fast_Ws: Sequence[jnp.ndarray]

    mem_status: Sequence[jnp.ndarray]
    traces: Sequence[jnp.ndarray]
    spikes: jnp.ndarray
    carrys: Sequence[jnp.ndarray]


class MetaStdpMLPPolicy(PolicyNetwork):
    """A meta-learning policy. Uses the same algorithm as
    Meta-Learning through Hebbian Plasticity in Random Networks
    [Najarro, Elias, and Sebastian Risi.]

    NOTE: No bias params are added to match the paper.
    """

    def __init__(
            self,
            input_dim: Sequence[int],
            hidden_dims: Sequence[int],
            output_dim: Sequence[int],
            output_act_fn: str = 'linear',
    ):
        self.out_fn = output_act_fn
        hidden_dims = [input_dim, *hidden_dims, output_dim]
        self._hidden_dims = hidden_dims
        # Initial values of the weights, used when state.first == True.
        params = []
        for i in range(len(hidden_dims) - 1):
            params.append(jnp.zeros((hidden_dims[i], hidden_dims[i + 1])))

        # Add hebbian learning parameters for each weight matrix.
        hebbian = []
        for param in params:
            # Create 5 copies of the params for the A, B, C, D and learning rate
            # meta learning parameters.
            hebbian.append([param] * 5)

        # params.append(hebbian)
        # self.num_params, format_params_fn = get_params_format_fn(params)
        self.num_params, format_params_fn = get_params_format_fn(hebbian)
        self._format_params_fn = jax.vmap(format_params_fn)

        def _forward(fast_params, mem_status, traces, x):
            # Forward computation of a normal neural network, record the activations
            # for the meta-learning step next.
            acts = [x]
            for i, w in enumerate(fast_params[:-1]):
                mem, x = lif_neuron(mem_status[i], jnp.dot(x, w))
                mem_status[i] = mem
                acts.append(x)

            final_w = fast_params[-1]
            out = jnp.dot(x, final_w)
            acts.append(out)

            for i in range(len(traces)):
                traces[i] = traces[i] / jnp.e + acts[i + 1] * (1. - 1. / jnp.e)
            acts[1: -1] = traces

            if self.out_fn == 'tanh':
                out = nn.tanh(x)
            elif 'softmax' in self.out_fn:
                out = nn.softmax(x, axis=-1)
            elif self.out_fn == 'linear':
                pass
            if 'select' in self.out_fn:
                out = jnp.argmax(x, axis=-1)

            return acts, out, mem_status, traces

        def _update(fast_params, hebbian, acts):
            # Meta-learning
            new_weights = []
            for i, (fast_param, heb) in enumerate(zip(fast_params, hebbian)):
                lr, A, B, C, D = heb
                # lr, A, C, D = heb
                # lr, B, D = heb
                # This code makes use of broadcasting to compute the activation
                # interactions as shown.
                _xy = acts[i][:, jnp.newaxis] * acts[i + 1][jnp.newaxis, :]
                _x = acts[i][:, jnp.newaxis]
                _y = acts[i + 1][jnp.newaxis, :]
                update = (
                        A * _xy - A * C * _x + B * _y + D
                        # A * _xy - A * C * _x + B * _y
                        # A * _xy - A * C * _x + D
                        # (B * _y + D)
                        # D
                )
                # Update the weights modulated by the learning rate
                new_w = fast_param + lr * update
                new_w /= jnp.abs(new_w).max()
                new_weights.append(new_w)

            return new_weights

        def apply(params, x, p_states):
            # *params, hebbian = params
            hebbian = params
            mem_status = p_states.mem_status
            traces = p_states.traces

            # Fetch the weights, if it's the first step get the learnt initial
            # weights, otherwise fetch the fast weights from the state.
            fast_params = p_states.fast_Ws

            acts, out, mem_status, traces = _forward(fast_params, mem_status, traces, x)

            # Meta-learning
            new_weights = _update(fast_params, hebbian, acts)

            p_states = MetaStdpMLPPolicyState(
                keys=p_states.keys,
                first=False,  # Next step will not be the first step
                fast_Ws=new_weights,
                traces=traces,
                mem_status=mem_status
            )
            # self.traces = traces
            # self.mem_status = mem_status
            return out, p_states

        self._forward_fn = jax.vmap(apply)

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        return self._forward_fn(params, t_states.obs, p_states)

    def reset(self, states: TaskState) -> PolicyState:
        if len(states.obs.shape) == 2:
            dims = states.obs.shape[0]
        else:
            dims = 1

        # Here we seem to initialize the fast weights to zero but these are never
        # used. They are set here to ensure that the shapes are consistent.
        # The first time the policy is called this is set to the learned inital
        # weights from the policy's paramteres.
        fast_W = []
        for i in range(len(self._hidden_dims) - 1):
            fast_W.append(
                jnp.zeros((dims, self._hidden_dims[i], self._hidden_dims[i + 1])))

        mem_status = []
        traces = []
        for hid_dim in self._hidden_dims[1:-1]:
            mem_status.append(jnp.zeros((dims, hid_dim)))
            traces.append(jnp.zeros((dims, hid_dim)))

        return MetaStdpMLPPolicyState(
            keys=jax.random.split(jax.random.PRNGKey(42), dims),
            first=jnp.array([True] * dims),
            fast_Ws=fast_W,
            mem_status=mem_status,
            traces=traces,
        )


class BatchedGruMetaStdpMLPPolicy(PolicyNetwork):
    """A meta-learning policy. Uses the same algorithm as
    Meta-Learning through Hebbian Plasticity in Random Networks
    [Najarro, Elias, and Sebastian Risi.]

    NOTE: No bias params are added to match the paper.
    """

    def __init__(self, input_dim: Sequence[int], hidden_dims: Sequence[int], output_dim: Sequence[int]):
        hidden_dims = [input_dim, *hidden_dims, output_dim]
        self._hidden_dims = hidden_dims
        # Initial values of the weights, used when state.first == True.
        params = []
        for i in range(len(hidden_dims) - 1):
            params.append(jnp.zeros((hidden_dims[i], hidden_dims[i + 1])))

        # Add hebbian learning parameters for each weight matrix.
        hebbian = []
        for param in params:
            # Create 5 copies of the params for the A, B, C, D and learning rate
            # meta learning parameters.
            hebbian.append([param] * 5)

        model = GRU(
            feat_dims=[hidden_dims[1]], out_dim=hidden_dims[-1], out_fn='linear'
        )
        params_gru = model.init(
            random.PRNGKey(0),
            carrys=[jnp.zeros([1, hidden_dims[1]])],
            x=jnp.ones([1, input_dim])
        )
        hebbian.append(params_gru)

        # params.append(hebbian)
        # self.num_params, format_params_fn = get_params_format_fn(params)
        self.num_params, format_params_fn = get_params_format_fn(hebbian)
        self._format_params_fn = jax.vmap(format_params_fn)

        def _forward(fast_params, mem_status, traces, params_gru, carrys, x):
            # Forward computation of a normal neural network, record the activations
            # for the meta-learning step next.
            acts = [x]
            for i, w in enumerate(fast_params[:-1]):
                mem, x = lif_neuron(mem_status[i], jnp.dot(x, w))
                mem_status[i] = mem
                acts.append(x)

            final_w = fast_params[-1]
            out = jnp.dot(x, final_w)

            new_carrys, out_gru = model.apply(params_gru, carrys, acts[0])
            out = (out + out_gru)
            acts.append(nn.tanh(out))

            for i in range(len(traces)):
                traces[i] = traces[i] / jnp.e + acts[i + 1] * (1. - 1. / jnp.e)
            acts[1: -1] = traces

            return acts, out, mem_status, traces, new_carrys

        def _update(fast_params, hebbian, acts):
            # Meta-learning
            new_weights = []
            for i, (fast_param, heb) in enumerate(zip(fast_params, hebbian)):
                lr, A, B, C, D = heb
                # This code makes use of broadcasting to compute the activation
                # interactions as shown.
                _xy = acts[i][:, :, jnp.newaxis] * acts[i + 1][:, jnp.newaxis, :]
                _x = acts[i][:, :, jnp.newaxis]
                _y = acts[i + 1][:, jnp.newaxis, :]
                _xy = jnp.sum(_xy, axis=0)
                _x = jnp.sum(_x, axis=0)
                _y = jnp.sum(_y, axis=0)
                update = (
                        (A * _xy + B * _x + C * _y + D)
                )
                # Update the weights modulated by the learning rate
                new_w = fast_param + lr * update

                # Normalize the weights after updating, very important!
                # https://github.com/enajx/HebbianMetaLearning/blob/master/fitness_functions.py#L226
                new_w /= jnp.abs(new_w).max()
                new_weights.append(new_w)

            return new_weights

        def apply(params, x, p_states):
            # *params, hebbian = params
            *hebbian, params_gru = params
            mem_status = p_states.mem_status
            traces = p_states.traces

            # Fetch the weights, if it's the first step get the learnt initial
            # weights, otherwise fetch the fast weights from the state.
            fast_params = p_states.fast_Ws
            carrys = p_states.carrys

            acts, out, mem_status, traces, carrys = _forward(fast_params, mem_status, traces, params_gru, carrys, x)

            # Meta-learning
            new_weights = _update(fast_params, hebbian, acts)

            p_states = MetaGruStdpMLPPolicyState(
                keys=p_states.keys,
                first=False,  # Next step will not be the first step
                fast_Ws=new_weights,
                traces=traces,
                mem_status=mem_status,
                carrys=carrys
            )
            # self.traces = traces
            # self.mem_status = mem_status
            return out, p_states

        self._forward_fn = jax.vmap(apply)

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        return self._forward_fn(params, t_states.obs, p_states)

    def reset(self, states: TaskState) -> PolicyState:
        dims = states.obs.shape[0]
        bs = states.obs.shape[1]

        # Here we seem to initialize the fast weights to zero but these are never
        # used. They are set here to ensure that the shapes are consistent.
        # The first time the policy is called this is set to the learned inital
        # weights from the policy's paramteres.
        fast_W = []
        for i in range(len(self._hidden_dims) - 1):
            fast_W.append(
                jnp.zeros((dims, self._hidden_dims[i], self._hidden_dims[i + 1])))

        mem_status = []
        traces = []
        carrys = []
        for hid_dim in self._hidden_dims[1:-1]:
            mem_status.append(jnp.zeros((dims, bs, hid_dim)))
            traces.append(jnp.zeros((dims, bs, hid_dim)))
            carrys.append(jnp.zeros((dims, bs, hid_dim)))

        return MetaGruStdpMLPPolicyState(
            keys=jax.random.split(jax.random.PRNGKey(42), dims),
            first=jnp.array([True] * dims),
            fast_Ws=fast_W,
            mem_status=mem_status,
            traces=traces,
            carrys=carrys
        )