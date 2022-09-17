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
from typing import Any, Mapping, Optional, Sequence, Tuple

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
) -> Sequence[Mapping[str, jnp.ndarray]]:
  nci = []
  for out_dim, init_tau in zip(nci_dims, nci_tau):
    nci.append({})
    key, subkey = jax.random.split(key, num=2)
    nci[-1]['w'] = (jax.random.normal(subkey, shape=(input_dim, out_dim)) /
                    jnp.sqrt(float(input_dim)))
    if tau_target:
      nci[-1]['tau'] = jnp.ones(1) * init_tau
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
            activation=options.nci_act,
            clip=options.nci_clip,
            tau=options.nci_tau[i:i + 1],
            residual=options.nci_res,
            softmax_w=options.nci_softmax_w,
            tau_target=options.nci_tau_target,
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


def log_linear_layer(
    logx: jnp.ndarray,
    params: Any,
    prev_sign: Optional[jnp.ndarray] = None,
    activation: str = 'tanh',
    clip: Optional[float] = None,
    tau: float = 1.,
    residual: str = 'none',
    softmax_w: bool = False,
    tau_target: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Evaluate act(x @ w) in log domain, i.e. compute with logx.

  No bias to keep antisymmetry.

  TODO(@shizk): conjecture: any activation function that is only zero
  at origin works, as it does not alter the nodal structure

  TODO(@shizk): why tanh doesn't work?

  TODO(@shizk): kfac registration

  Args:
    logx: inputs in log domain [B, in_dim]
    w: weights [in_dim, out_dim]
    prev_sign: sign from the previous log linear layer [B, in_dim]
    activation: activation function.
    clip: if not None, activation becomes linear within the range [-clip, clip]
    tau: temperature

  Returns:
    log(abs(act(x @ w))): [B, last_hdim]
    sign(act(x @ w)): [B, last_hdim]
  """
  debug_stats = {}

  if prev_sign is None:
    prev_sign = jnp.ones_like(logx)
  vmap_over_hidden = jax.vmap(
      lambda logx, w, prev_sign: reduce_weighted_logsumexp(
          logx=logx,
          w=(jax.nn.softmax(w) if softmax_w else w) * prev_sign,
          return_sign=True),
      in_axes=(None, 1, None),
      out_axes=0)
  logy, sign = vmap_over_hidden(logx, params[0]['w'], prev_sign)

  # residule in original domain
  if residual == 'pre_act' and logx.shape == logy.shape:
    logy, sign = jax.vmap(
        lambda logx, logy, prev_sign, sign: tfp.math.reduce_weighted_logsumexp(
            logx=[logx, logy], w=[prev_sign, sign], return_sign=True),
        in_axes=(0, 0, 0, 0),
        out_axes=0)(logx, logy, prev_sign, sign)

  # activation in original domain
  y = sign * jnp.exp(logy)
  debug_stats['pre_act_0'] = jnp.mean(y)

  if tau_target:
    y = y / params[0]['tau']
    debug_stats['tau_loss'] = jax.nn.relu(y - tau_target)
  else:
    y = y / tau[0]
  y = jnp.nan_to_num(y)

  # activation in original domain
  if activation == 'none':
    act_fn = lambda x: x
  elif 'lecun_tanh' in activation:
    alpha = float(activation.split('_')[-1])
    act_fn = lambda x: 1.7159 * jax.nn.tanh(x * 2. / 3.) + alpha * x
  else:
    act_fn = getattr(jax.nn, activation)
  if clip is not None:  # linear when y is small
    cond = jnp.abs(y) > clip  # 1e-8
    offset = clip - act_fn(clip)  # to make sure act is continuous
    y_act = jnp.where(cond, act_fn(y) + sign * offset, y)
  else:
    y_act = act_fn(y)

  # extra linears
  residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y
  for i in range(1, len(params)):
    w = params[i]['w']
    y_out = linear_layer(y, w=(jax.nn.softmax(w) if softmax_w else w))
    debug_stats[f'pre_act_{i}'] = jnp.mean(y_out)
    y_act = act_fn(y_out)
    debug_stats[f'act_{i}'] = jnp.mean(y_act)
    if tau_target:
      y = y / params[i]['tau']
      debug_stats['tau_loss'] += jax.nn.relu(y - tau_target)
    else:
      y = y / tau[i]
    y = jnp.nan_to_num(y)
    y = residual(y, y_act)

  if len(params) > 1:
    y_act = jnp.sum(y)
    sign = jnp.sign(y_act)
    logy_act = jnp.log(sign * y_act)
    return logy_act, sign, debug_stats

  sign = jnp.sign(y_act)
  logy_act = jnp.log(sign * y_act)
  logy_act = jnp.nan_to_num(logy_act)

  # residule in original domain
  if residual == 'post_act' and logx.shape == logy_act.shape:
    logy_act, sign = jax.vmap(
        lambda logx, logy, prev_sign, sign: tfp.math.reduce_weighted_logsumexp(
            logx=[logx, logy], w=[prev_sign, sign], return_sign=True),
        in_axes=(0, 0, 0, 0),
        out_axes=0)(logx, logy_act, prev_sign, sign)

  return logy_act, sign, debug_stats
