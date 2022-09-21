# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Neural network building blocks."""

import functools
import itertools
from typing import Any, Mapping, Optional, Sequence, Tuple, Callable

import chex
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


def array_partitions(sizes: Sequence[int]) -> Sequence[int]:
  """Returns the indices for splitting an array into separate partitions.

  Args:
    sizes: size of each of N partitions. The dimension of the array along
    the relevant axis is assumed to be sum(sizes).

  Returns:
    sequence of indices (length len(sizes)-1) at which an array should be split
    to give the desired partitions.
  """
  return list(itertools.accumulate(sizes))[:-1]


def init_linear_layer(key: chex.PRNGKey,
                      in_dim: int,
                      out_dim: int,
                      include_bias: bool = True) -> Mapping[str, jnp.ndarray]:
  """Initialises parameters for a linear layer, x w + b.

  Args:
    key: JAX PRNG state.
    in_dim: input dimension to linear layer.
    out_dim: output dimension (number of hidden units) of linear layer.
    include_bias: if true, include a bias in the linear layer.

  Returns:
    A mapping containing the weight matrix (key 'w') and, if required, bias
    unit (key 'b').
  """
  key1, key2 = jax.random.split(key)
  weight = (jax.random.normal(key1, shape=(in_dim, out_dim)) /
            jnp.sqrt(float(in_dim)))
  if include_bias:
    bias = jax.random.normal(key2, shape=(out_dim,))
    return {'w': weight, 'b': bias}
  else:
    return {'w': weight}


def init_nci(
    key: chex.PRNGKey,
    input_dim: int,
    nci_dims: Sequence[int],
    nci_tau: Sequence[float],
    tau_target: Optional[float] = None,
    clip_target: Sequence[Optional[float]] = None,
) -> Sequence[Mapping[str, jnp.ndarray]]:
  nci = []
  for out_dim, init_tau, init_clip in zip(nci_dims, nci_tau, clip_target):
    nci.append({})
    key, subkey = jax.random.split(key, num=2)
    nci[-1]['w'] = (jax.random.normal(subkey, shape=(input_dim, out_dim)) /
                    jnp.sqrt(float(input_dim)))
    if tau_target:
      nci[-1]['tau'] = jnp.ones(1) * init_tau
    if init_clip:
      nci[-1]['clip'] = jnp.ones(1) * init_clip
    # NOTE(@shizk): no bias to keep antisymmetry
    input_dim = out_dim
  return nci


def linear_layer(x: jnp.ndarray,
                 w: jnp.ndarray,
                 b: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Evaluates a linear layer, x w + b.

  Args:
    x: inputs.
    w: weights.
    b: optional bias.

  Returns:
    x w + b if b is given, x w otherwise.
  """
  y = jnp.dot(x, w)
  return y + b if b is not None else y


vmap_linear_layer = jax.vmap(linear_layer, in_axes=(0, None, None), out_axes=0)


def slogdet(x):
  """Computes sign and log of determinants of matrices.

  This is a jnp.linalg.slogdet with a special (fast) path for small matrices.

  Args:
    x: square matrix.

  Returns:
    sign, (natural) logarithm of the determinant of x.
  """
  if x.shape[-1] == 1:
    sign = jnp.sign(x[..., 0, 0])
    logdet = jnp.log(jnp.abs(x[..., 0, 0]))
  else:
    sign, logdet = jnp.linalg.slogdet(x)

  return sign, logdet


def logdet_matmul(
    xs: Sequence[jnp.ndarray],
    params: Any = None,
    options: Any = None,
) -> jnp.ndarray:
  """Combines determinants and takes dot product with weights in log-domain.

  We use the log-sum-exp trick to reduce numerical instabilities.

  Args:
    xs: FermiNet orbitals in each determinant. Either of length 1 with shape
      (ndet, nelectron, nelectron) (full_det=True) or length 2 with shapes
      (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta) (full_det=False,
      determinants are factorised into block-diagonals for each spin channel).
    w: weight of each determinant. If none, a uniform weight is assumed.

  Returns:
    sum_i w_i D_i in the log domain, where w_i is the weight of D_i, the i-th
    determinant (or product of the i-th determinant in each spin channel, if
    full_det is not used).
  """
  # 1x1 determinants appear to be numerically sensitive and can become 0
  # (especially when multiple determinants are used with the spin-factored
  # wavefunction). Avoid this by not going into the log domain for 1x1 matrices.
  # Pass initial value to functools so det1d = 1 if all matrices are larger than
  # 1x1.
  det1d = functools.reduce(lambda a, b: a * b,
                           [x.reshape(-1) for x in xs if x.shape[-1] == 1], 1)

  # Pass initial value to functools so sign_in = 1, logdet = 0 if all matrices
  # are 1x1.
  sign_in, logdet = functools.reduce(
      lambda a, b: (a[0] * b[0], a[1] + b[1]),
      [slogdet(x) for x in xs if x.shape[-1] > 1], (1, 0))

  debug_stats = {'logdet_abs': jnp.mean(logdet)}

  if options.use_nci:  # neural CI in log domain
    det1d = 1  # HACK(@shizk): do we need to care about this?

    if options.nci_remain_in_log:
      for i in range(len(params['nci'])):
        logdet, sign_in, debug_stats_ = log_linear_layer(
            logdet,
            params=params['nci'][i:i + 1],
            prev_sign=sign_in,
            activation=options.nci_act[i:i + 1],
            clip=options.nci_clip[i:i + 1],
            tau=options.nci_tau[i:i + 1],
            residual=options.nci_res,
            softmax_w=options.nci_softmax_w,
            tau_target=options.nci_tau_target,
            leak=options.nci_leak,
        )
        debug_stats = {**debug_stats, **debug_stats_}
        debug_stats[f'logdet_abs_{i}'] = logdet
    else:
      logdet, sign_in, debug_stats_ = log_linear_layer(
          logdet,
          params=params['nci'],
          prev_sign=sign_in,
          activation=options.nci_act,
          clip=options.nci_clip,
          tau=options.nci_tau,
          residual=options.nci_res,
          softmax_w=options.nci_softmax_w,
          tau_target=options.nci_tau_target,
          in_log=options.nci_first_layer_in_log,
          leak=options.nci_leak,
      )
      debug_stats = {**debug_stats, **debug_stats_}

  # log-sum-exp trick
  maxlogdet = jnp.max(logdet)
  det = sign_in * det1d * jnp.exp(logdet - maxlogdet)
  result = jnp.sum(det)

  sign_out = jnp.sign(result)
  log_out = jnp.log(jnp.abs(result)) + maxlogdet
  return sign_out, log_out, debug_stats


def reduce_weighted_logsumexp(logx,
                              w=None,
                              axis=None,
                              keep_dims=False,
                              return_sign=False,
                              name=None):
  log_absw_x = logx + jnp.log(jnp.abs(w))
  max_log_absw_x = jnp.max(log_absw_x, axis=axis, keepdims=True)
  # If the largest element is `-inf` or `inf` then we don't bother subtracting
  # off the max. We do this because otherwise we'd get `inf - inf = NaN`. That
  # this is ok follows from the fact that we're actually free to subtract any
  # value we like, so long as we add it back after taking the `log(sum(...))`.
  max_log_absw_x = jnp.where(jnp.isinf(max_log_absw_x),
                             jnp.zeros([], max_log_absw_x.dtype),
                             max_log_absw_x)
  wx_over_max_absw_x = (jnp.sign(w) * jnp.exp(log_absw_x - max_log_absw_x))
  sum_wx_over_max_absw_x = jnp.sum(wx_over_max_absw_x,
                                   axis=axis,
                                   keepdims=keep_dims)
  if not keep_dims:
    max_log_absw_x = jnp.squeeze(max_log_absw_x, axis)
  sgn = jnp.sign(sum_wx_over_max_absw_x)
  lswe = max_log_absw_x + jnp.log(sgn * sum_wx_over_max_absw_x)
  return lswe, sgn


def make_act_fn(activation: str) -> Callable:
  # activation in original domain
  if activation == 'none':
    act_fn = lambda x: x
  elif 'lecun_tanh' in activation:
    alpha = float(activation.split('_')[-1])
    act_fn = lambda x: 1.7159 * jax.nn.tanh(x * 2. / 3.) + alpha * x
  elif 'xe' in activation:
    leak = float(activation.split('_')[-1])
    act_fn = lambda x: x * (jnp.exp(-x**2) + leak)
  else:
    act_fn = getattr(jax.nn, activation)
  return act_fn


def log_linear_layer(
    logx: jnp.ndarray,
    params: Any,
    prev_sign: Optional[jnp.ndarray] = None,
    activation: Sequence[str] = ['tanh'],
    clip: Sequence[Optional[float]] = [None],
    tau: Sequence[float] = [1.],
    residual: str = 'none',
    softmax_w: bool = False,
    tau_target: Optional[float] = None,
    in_log: bool = True,
    leak: float = 0.5,
) -> Tuple[jnp.ndarray, jnp.ndarray, Mapping[str, jnp.ndarray]]:
  """Evaluate act(x @ w) in log domain, i.e. compute with logx.

  No bias to keep antisymmetry.

  Args:
    logx: inputs in log domain [B, in_dim]
    w: weights [in_dim, out_dim]
    prev_sign: sign from the previous log linear layer [B, in_dim]
    activation: activation function.
    clip: if not None, activation becomes linear within the range [-clip, clip]
    tau: width of activation
    residual: whether to use residual
    softmax_w: whether to softmax the weights
    tau_target: auto tune tau
    in_log: whether to do the first layer computation in log domain

  Returns:
    log(abs(act(x @ w))): [B, last_hdim]
    sign(act(x @ w)): [B, last_hdim]
  """
  debug_stats = {}
  has_extra_linear = len(params) > 1
  residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y

  x: jnp.ndarray
  for i in range(len(params)):
    # LINEAR LAYER
    if i == 0:  # first layer is from log domain
      if prev_sign is None:
        prev_sign = jnp.ones_like(logx)

      if in_log:  # do the linear layer in log domain
        vmap_over_hidden = jax.vmap(
            lambda logx, w, prev_sign: reduce_weighted_logsumexp(
                logx=logx,
                w=(jax.nn.softmax(w) if softmax_w else w) * prev_sign,
                return_sign=True),
            in_axes=(None, 1, None),
            out_axes=0)
        log_wx, sign = vmap_over_hidden(logx, params[i]['w'], prev_sign)

        # go back to original domain
        wx = sign * jnp.exp(log_wx)

      else:  # convert input back to the original domain
        x = prev_sign * jnp.exp(logx)
        w = params[i]['w']
        wx = linear_layer(x, w=(jax.nn.softmax(w) if softmax_w else w))
        sign = jnp.sign(wx)

    else:
      w = params[i]['w']
      wx = linear_layer(x, w=(jax.nn.softmax(w) if softmax_w else w))
      sign = jnp.sign(wx)

    debug_stats[f'pre_act_{i}'] = jnp.mean(wx)

    # SCALE: change the "width" of the activation function
    if tau_target:
      wx = wx / jnp.abs(params[i]['tau'])
      debug_stats['tau_loss'] += jax.nn.relu(jnp.abs(wx) - tau_target)
    else:
      wx = wx / tau[i]
    wx = jnp.nan_to_num(wx)

    # ACT+CILP
    act_fn = make_act_fn(activation[i])
    trainable_clip = 'clip' in params[i].keys()
    if clip[i] is not None or trainable_clip:  # linear when wx is LARGE
      if trainable_clip:  # adjust clip_val s.t. clip doesn't happen very often
        clip_val = jnp.abs(params[i]['clip'])
        debug_stats['clip_loss'] += jax.nn.relu(jnp.abs(wx) - clip_val)
      else:
        clip_val = clip[i]
      cond = jnp.abs(wx) > clip_val  # 1e-8
      offset = clip_val - act_fn(clip_val)  # to make sure act is continuous
      y = jnp.where(cond, leak * wx - sign * offset, act_fn(wx))
    else:
      y = act_fn(wx)

    debug_stats[f'act_{i}'] = jnp.mean(y)
    y = jnp.nan_to_num(y)

    # RESIDUAL
    if i == len(params) - 1:  # last layer
      if has_extra_linear:
        y = residual(x, y)

      # convert back to log domain
      sign = jnp.sign(y)
      logy = jnp.log(sign * y)
      logy = jnp.nan_to_num(logy)

      # residual in log domain since input/x is in log domain
      if not has_extra_linear and residual == 'post_act' and logx.shape == logy.shape:
        logy, sign = jax.vmap(
            lambda logx, logy, prev_sign, sign: reduce_weighted_logsumexp(
                logx=[logx, logy], w=[prev_sign, sign], return_sign=True),
            in_axes=(0, 0, 0, 0),
            out_axes=0)(logx, logy, prev_sign, sign)

      return logy, sign, debug_stats

    else:  # not the last layer
      if i == 0:
        # for the first layer, convert the input from log domain
        x = prev_sign * jnp.exp(logx)
      # for all subsequence layers, x is already in the original domain
      x = residual(x, y)
