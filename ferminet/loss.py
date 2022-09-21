# Copyright 2020 DeepMind Technologies Limited.
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
"""Helper functions to create the loss and custom gradient of the loss."""

from typing import Tuple, Dict, Optional

from absl import logging
import chex
from ferminet import constants
from ferminet import hamiltonian
from ferminet import networks
import jax
import jax.numpy as jnp
import kfac_jax
from typing_extensions import Protocol
from jax.tree_util import tree_flatten
import jax.experimental.host_callback as hcb
import wandb


@chex.dataclass
class AuxiliaryLossData:
  """Auxiliary data returned by total_energy.

  Attributes:
    variance: mean variance over batch, and over all devices if inside a pmap.
    local_energy: local energy for each MCMC configuration.
  """
  variance: jnp.DeviceArray
  local_energy: jnp.DeviceArray
  k: jnp.DeviceArray
  v_ee: jnp.DeviceArray
  v_ae: jnp.DeviceArray
  v_aa: jnp.DeviceArray
  stats: Dict[str, jnp.DeviceArray]


class LossFn(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched electron positions to pass to the network.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """


# define alpha of control variate
alpha_c = 1.
ratio_step = 0
record_ratio_c = []
record_ratio_fs = []


def callback_func(args, transforms=None):
  global alpha_c
  ratio, mom = args
  alpha_c = mom * alpha_c + (1. - mom) * ratio


def update_alpha_c(ratio, mom):
  return hcb.id_tap(callback_func, (ratio, mom))


def callback_func_log(args, transforms=None):
  global ratio_step, alpha_c
  info = {'alpha_c': alpha_c}
  info['ratio'], info['cov_all'], info['ratio_fs'] = args
  info = dict(jax.tree_map(lambda x: float(jnp.nan_to_num(jnp.mean(x))), info))
  logging.info(f"Ratio Iter: {ratio_step}, " +
               ", ".join([f"{k}: {v:.4f}" for k, v in info.items()]))
  # wandb.log(data=info, step=ratio_step)
  ratio_step += 1


def log_ratio_c(ratio, cov_all, ratio_fs):
  return hcb.id_tap(callback_func_log, (ratio, cov_all, ratio_fs))


def make_loss(
    # network: networks.LogFermiNetLike,
    signed_network: networks.FermiNetLike,
    local_energy: hamiltonian.LocalEnergy,
    clip_local_energy: float = 0.0,
    grad_norm_reg: float = 0.0,
    logdet_reg_lambda: float = 0.0,
    nci_w_reg_lambda: float = 0.0,
    tau_loss_lambda: float = 0.0,
    clip_lambda: float = 0.0,
    kinetic: str = 'baseline',
    control_variate: bool = False,
    control_variate_mom: float = 0.9,
) -> LossFn:
  """Creates the loss function, including custom gradients.

  Args:
    network: callable which evaluates the log of the magnitude of the
      wavefunction (square root of the log probability distribution) at a
      single MCMC configuration given the network parameters.
    local_energy: callable which evaluates the local energy.
    clip_local_energy: If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.

  Returns:
    Callable with signature (params, data) and returns (loss, aux_data), where
    loss is the mean energy, and aux_data is an AuxiliaryLossDataobject. The
    loss is averaged over the batch and over all devices inside a pmap.
  """
  network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
  logdet_abs = lambda *args, **kwargs: signed_network(*args, **kwargs)[-1][
      'logdet_abs']
  tau_loss = lambda *args, **kwargs: signed_network(*args, **kwargs)[-1][
      'tau_loss']
  clip_loss = lambda *args, **kwargs: signed_network(*args, **kwargs)[-1][
      'clip_loss']
  batch_local_energy = jax.vmap(local_energy,
                                in_axes=(None, 0, 0, 0),
                                out_axes=0)
  batch_network = jax.vmap(network, in_axes=(None, 0), out_axes=0)
  batch_logdet_abs = jax.vmap(logdet_abs, in_axes=(None, 0), out_axes=0)
  batch_tau_loss = jax.vmap(tau_loss, in_axes=(None, 0), out_axes=0)
  batch_clip_loss = jax.vmap(clip_loss, in_axes=(None, 0), out_axes=0)

  def grad_norm(params, single_example_batch):
    grads = jax.grad(network)(params, single_example_batch)
    nonempty_grads, _ = tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm(
        [jnp.linalg.norm(grad.ravel()) for grad in nonempty_grads])
    return jnp.square(total_grad_norm)

  batch_grad_norm = jax.vmap(grad_norm, in_axes=(None, 0), out_axes=0)

  @jax.custom_jvp
  def total_energy(
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched MCMC configurations to pass to the local energy function.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """
    if kinetic != 'baseline':
      seed = 1e8 * jnp.mean(jnp.abs(data))
      # random projection vector
      v = jax.random.normal(key=jax.random.PRNGKey(seed.astype(int)),
                            shape=data.shape)
    else:
      v = None

    keys = jax.random.split(key, num=data.shape[0])
    k_first, k_second, v_ee, v_ae, v_aa, stats = batch_local_energy(
        params, keys, data, v)

    if control_variate:
      global alpha_c  # initialized to 1
      potential = v_ee + v_ae + v_aa
      batch_size = data.shape[0]
      f_s = jnp.concatenate([
          jnp.reshape(k_second, (1, batch_size)),
          jnp.reshape(k_first, (1, batch_size)),
          jnp.reshape(potential, (1, batch_size))
      ])
      cov_all = jnp.cov(f_s)
      qudra_a = cov_all[0][0] + 4 * cov_all[1][1] + 4 * cov_all[0][1]
      qudra_minus_b = (2 * cov_all[1][1] + cov_all[0][1] + 2 * cov_all[0][2] +
                       4 * cov_all[1][2])
      ratio = qudra_minus_b / qudra_a
      ratio_fs = jnp.mean(k_first) / jnp.mean(k_second)
      update_alpha_c(ratio, control_variate_mom)
      log_ratio_c(ratio, cov_all, ratio_fs)

      # alpha initialized to 1
      k = -0.5 * ((2 * alpha_c - 1) * k_first + alpha_c * k_second)
    else:
      k = -0.5 * (k_first + k_second)

    e_l = k + v_ee + v_ae + v_aa
    loss = constants.pmean(jnp.mean(e_l))
    variance = constants.pmean(jnp.mean((e_l - loss)**2))
    # if grad_norm_reg > 0.0:
    stats['grad_norm'] = batch_grad_norm(params, data)
    logdet_abs_val = batch_logdet_abs(params, data)
    stats['logdet_abs'] = constants.pmean(jnp.mean(logdet_abs_val))
    if 'nci' in params.keys():
      stats['nci_w_norm'] = jnp.linalg.norm(
          [jnp.linalg.norm(layer['w']) for layer in params['nci']])
      for i, layer in enumerate(params['nci']):
        for key in ['tau', 'clip']:
          if key in params['nci'][i].keys():
            stats[f'{key}_{i}'] = layer[key]
    return loss, AuxiliaryLossData(
        variance=variance,
        local_energy=e_l,
        k=k,
        v_ee=v_ee,
        v_ae=v_ae,
        v_aa=v_aa,
        stats=stats,
    )

  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
    """Custom Jacobian-vector product for unbiased local energy gradients."""
    params, key, data = primals
    loss, aux_data = total_energy(params, key, data)

    # diff = e_l - E[e_l]
    if clip_local_energy > 0.0:
      # Try centering the window around the median instead of the mean?
      tv = jnp.mean(jnp.abs(aux_data.local_energy - loss))
      tv = constants.pmean(tv)
      diff = jnp.clip(aux_data.local_energy, loss - clip_local_energy * tv,
                      loss + clip_local_energy * tv) - loss
    else:
      diff = aux_data.local_energy - loss

    # Due to the simultaneous requirements of KFAC (calling convention must be
    # (params, rng, data)) and Laplacian calculation (only want to take
    # Laplacian wrt electron positions) we need to change up the calling
    # convention between total_energy and batch_network
    primals = primals[0], primals[2]
    tangents = tangents[0], tangents[2]
    psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
    kfac_jax.register_normal_predictive_distribution(psi_primal[:, None])
    primals_out = loss, aux_data
    device_batch_size = jnp.shape(aux_data.local_energy)[0]
    grad_est = jnp.dot(psi_tangent, diff)

    reg = 0.0

    if logdet_reg_lambda > 0.0:
      _, logdet_abs_tangent = jax.jvp(batch_logdet_abs, primals, tangents)
      reg += logdet_reg_lambda * jnp.mean(logdet_abs_tangent)

    if grad_norm_reg > 0.0:
      _, grad_norm_tangent = jax.jvp(batch_grad_norm, primals, tangents)
      reg += grad_norm_reg * jnp.mean(grad_norm_tangent)

    if 'nci' in params.keys():
      if nci_w_reg_lambda > 0.0:
        reg += nci_w_reg_lambda * aux_data.stats['nci_w_norm']

      if 'tau' in params['nci'][0].keys() and tau_loss_lambda > 0.0:
        _, tau_loss_tangent = jax.jvp(batch_tau_loss, primals, tangents)
        reg += tau_loss_lambda * jnp.mean(tau_loss_tangent)

    if clip_lambda > 0.0:
      _, clip_loss_tangent = jax.jvp(batch_clip_loss, primals, tangents)
      reg += clip_lambda * jnp.mean(clip_loss_tangent)

    tangents_out = ((grad_est + reg) / device_batch_size, aux_data)
    return primals_out, tangents_out

  return total_energy
