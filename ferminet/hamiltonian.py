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
"""Evaluating the Hamiltonian on a wavefunction."""

from typing import Any, Sequence, Optional

from absl import flags
import chex
from ferminet import networks
import jax
from jax import lax
import jax.numpy as jnp
from typing_extensions import Protocol

FLAGS = flags.FLAGS


class LocalEnergy(Protocol):

  def __call__(self, params: networks.ParamTree, key: chex.PRNGKey,
               data: jnp.ndarray) -> jnp.ndarray:
    """Returns the local energy of a Hamiltonian at a configuration.

    Args:
      params: network parameters.
      key: JAX PRNG state.
      data: MCMC configuration to evaluate.
    """


class MakeLocalEnergy(Protocol):

  def __call__(self,
               f: networks.FermiNetLike,
               atoms: jnp.ndarray,
               charges: jnp.ndarray,
               nspins: Sequence[int],
               use_scan: bool = False,
               **kwargs: Any) -> LocalEnergy:
    """Builds the LocalEnergy function.

    Args:
      f: Callable which evaluates the sign and log of the magnitude of the
        wavefunction.
      atoms: atomic positions.
      charges: nuclear charges.
      nspins: Number of particles of each spin.
      use_scan: Whether to use a `lax.scan` for computing the laplacian.
      **kwargs: additional kwargs to use for creating the specific Hamiltonian.
    """


def local_kinetic_energy(f: networks.LogFermiNetLike, use_scan: bool = False):
  r"""Creates a function to for the local kinetic energy, -1/2 \nabla^2 ln|f|.

  Args:
    f: Callable which evaluates the log of the magnitude of the wavefunction.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.

  Returns:
    Callable which evaluates the local kinetic energy,
    -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| + (\nabla log|f|)^2).
  """

  def _lapl_over_f(params, data, v=None):
    n = data.shape[0]
    grad_f = jax.grad(f, argnums=1)
    grad_f_closure = lambda x: grad_f(params, x)

    primal, dgrad_f = jax.linearize(grad_f_closure, data)
    first_d = jnp.sum(primal**2)
    eye = jnp.eye(n)

    if use_scan:
      _, diagonal = lax.scan(lambda i, _: (i + 1, dgrad_f(eye[i])[i]),
                             0,
                             None,
                             length=n)
      second_d = jnp.sum(diagonal)
    else:
      second_d = lax.fori_loop(0, n, lambda i, val: val + dgrad_f(eye[i])[i],
                               0.0)
    return first_d, second_d

  return _lapl_over_f


def local_kinetic_energy_fd(f: networks.LogFermiNetLike,
                            use_scan: bool = False):
  """calculate ke with finite difference"""

  def _lapl_over_f(params, data, v, eps=0.1):
    """
    Args:
      eps: the difference used in FD
    """
    n = data.shape[0]
    grad_f = jax.grad(f, argnums=1)
    grad_f_closure = lambda y: grad_f(params, y)

    eps = eps * jnp.mean(jnp.abs(data))  # scale by data scale
    v = v / jnp.linalg.norm(v, axis=0, keepdims=True) * eps  # normalize
    grad_P, grad_N = grad_f_closure(data + v), grad_f_closure(data - v)

    first = jnp.sum(jnp.square(grad_P + grad_N), axis=0) / 4
    second = jnp.dot(v, grad_P - grad_N) * n / (2 * eps**2)
    return first, second

  return _lapl_over_f


def local_kinetic_energy_hutchinson(f, use_scan: bool = False):

  def _lapl_over_f(params, data, v):
    n = data.shape[0]
    grad_f = jax.grad(f, argnums=1)
    grad_f_closure = lambda x: grad_f(params, x)

    primal, dgrad_f = jax.linearize(grad_f_closure, data)
    first_d = jnp.sum(primal**2)
    eye = jnp.eye(n)

    if use_scan:
      _, diagonal = lax.scan(lambda i, _: (i + 1, dgrad_f(eye[i])[i] * v[i]**2),
                             0,
                             None,
                             length=n)
      second_d = jnp.sum(diagonal)
    else:
      second_d = lax.fori_loop(
          0, n, lambda i, val: val + dgrad_f(eye[i])[i] * v[i]**2, 0.0)
    return first_d, second_d

  return _lapl_over_f


def potential_electron_electron(r_ee: jnp.ndarray) -> jnp.ndarray:
  """Returns the electron-electron potential.

  Args:
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
  """
  return jnp.sum(jnp.triu(1 / r_ee[..., 0], k=1))


def potential_electron_nuclear(charges: jnp.ndarray,
                               r_ae: jnp.ndarray) -> jnp.ndarray:
  """Returns the electron-nuclearpotential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
  """
  return -jnp.sum(charges / r_ae[..., 0])


def potential_nuclear_nuclear(charges: jnp.ndarray,
                              atoms: jnp.ndarray) -> jnp.ndarray:
  """Returns the electron-nuclearpotential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    atoms: Shape (natoms, ndim). Positions of the atoms.
  """
  r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
  return jnp.sum(jnp.triu((charges[None, ...] * charges[..., None]) / r_aa,
                          k=1))


def potential_energy(r_ae: jnp.ndarray, r_ee: jnp.ndarray, atoms: jnp.ndarray,
                     charges: jnp.ndarray) -> jnp.ndarray:
  """Returns the potential energy for this electron configuration.

  Args:
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
  """
  return (potential_electron_electron(r_ee) +
          potential_electron_nuclear(charges, r_ae) +
          potential_nuclear_nuclear(charges, atoms))


def local_energy(f: networks.FermiNetLike,
                 atoms: jnp.ndarray,
                 charges: jnp.ndarray,
                 nspins: Sequence[int],
                 use_scan: bool = False,
                 kinetic: str = 'baseline') -> LocalEnergy:
  """Creates the function to evaluate the local energy.

  Args:
    f: Callable which returns the sign and log of the magnitude of the
      wavefunction given the network parameters and configurations data.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
    nspins: Number of particles of each spin.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction given the parameters params, RNG state key,
    and a single MCMC configuration in data.
  """
  del nspins
  log_abs_f = lambda *args, **kwargs: f(*args, **kwargs)[1]
  f_stats = lambda *args, **kwargs: f(*args, **kwargs)[-1]

  if kinetic == 'baseline':
    ke = local_kinetic_energy(log_abs_f, use_scan=use_scan)
  elif kinetic == 'hutchinson':
    ke = local_kinetic_energy_hutchinson(log_abs_f, use_scan=use_scan)
  elif kinetic == 'fd':
    ke = local_kinetic_energy_fd(log_abs_f, use_scan=use_scan)
  else:
    raise RuntimeError

  def _e_l(params: networks.ParamTree,
           key: chex.PRNGKey,
           data: jnp.ndarray,
           v: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Returns the total energy.

    Args:
      params: network parameters.
      key: RNG state.
      data: MCMC configuration.
    """
    del key  # unused
    _, _, r_ae, r_ee = networks.construct_input_features(data, atoms)
    # potential = potential_energy(r_ae, r_ee, atoms, charges)
    v_ee = potential_electron_electron(r_ee)
    v_ae = potential_electron_nuclear(charges, r_ae)
    v_aa = potential_nuclear_nuclear(charges, atoms)
    k_first, k_second = ke(params, data, v)
    if FLAGS.log_debug_stats:
      stats = f_stats(params, data)
    else:
      stats = {}
    return k_first, k_second, v_ee, v_ae, v_aa, stats

  return _e_l
