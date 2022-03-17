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

"""Core training loop for neural QMC in JAX."""

import functools
import importlib
import time
from typing import Optional, Sequence, Tuple, Union

from absl import logging
import chex
from ferminet import checkpoint
from ferminet import constants
from ferminet import hamiltonian
from ferminet import loss as qmc_loss_functions
from ferminet import mcmc
from ferminet import networks
from ferminet import pretrain
from ferminet.utils import statistics
from ferminet.utils import system
from ferminet.utils import writers
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from typing_extensions import Protocol

from kfac_ferminet_alpha import optimizer as kfac_optim
from kfac_ferminet_alpha import utils as kfac_utils


def init_electrons(
    key,
    molecule: Sequence[system.Atom],
    electrons: Sequence[int],
    batch_size: int,
    init_width: float,
) -> jnp.ndarray:
  """Initializes electron positions around each atom.

  Args:
    key: JAX RNG state.
    molecule: system.Atom objects making up the molecule.
    electrons: tuple of number of alpha and beta electrons.
    batch_size: total number of MCMC configurations to generate across all
      devices.
    init_width: width of (atom-centred) Gaussian used to generate initial
      electron configurations.

  Returns:
    array of (batch_size, (nalpha+nbeta)*ndim) of initial (random) electron
    positions in the initial MCMC configurations and ndim is the dimensionality
    of the space (i.e. typically 3).
  """
  if sum(atom.charge for atom in molecule) != sum(electrons):
    if len(molecule) == 1:
      atomic_spin_configs = [electrons]
    else:
      raise NotImplementedError('No initialization policy yet '
                                'exists for charged molecules.')
  else:
    atomic_spin_configs = [
        (atom.element.nalpha, atom.element.nbeta) for atom in molecule
    ]
    assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
    while tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons:
      i = np.random.randint(len(atomic_spin_configs))
      nalpha, nbeta = atomic_spin_configs[i]
      atomic_spin_configs[i] = nbeta, nalpha

  # Assign each electron to an atom initially.
  electron_positions = []
  for i in range(2):
    for j in range(len(molecule)):
      atom_position = jnp.asarray(molecule[j].coords)
      electron_positions.append(
          jnp.tile(atom_position, atomic_spin_configs[j][i]))
  electron_positions = jnp.concatenate(electron_positions)
  # Create a batch of configurations with a Gaussian distribution about each
  # atom.
  key, subkey = jax.random.split(key)
  return (
      electron_positions +
      init_width *
      jax.random.normal(subkey, shape=(batch_size, electron_positions.size)))


# All optimizer states (KFAC and optax-based).
OptimizerState = Union[optax.OptState, kfac_optim.State]
OptUpdateResults = Tuple[networks.ParamTree, Optional[OptimizerState],
                         jnp.ndarray,
                         Optional[qmc_loss_functions.AuxiliaryLossData]]


class OptUpdate(Protocol):

  def __call__(self, params: networks.ParamTree,
               data: jnp.ndarray,
               opt_state: optax.OptState,
               key: chex.PRNGKey) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters accordingly.

    Args:
      params: network parameters.
      data: electron positions.
      opt_state: optimizer internal state.
      key: RNG state.

    Returns:
      Tuple of (params, opt_state, loss, aux_data), where params and opt_state
      are the updated parameters and optimizer state, loss is the evaluated loss
      and aux_data auxiliary data (see AuxiliaryLossData docstring).
    """


StepResults = Tuple[jnp.ndarray, networks.ParamTree, Optional[optax.OptState],
                    jnp.ndarray, qmc_loss_functions.AuxiliaryLossData,
                    jnp.ndarray]


class Step(Protocol):

  def __call__(self,
               data: jnp.ndarray,
               params: networks.ParamTree,
               state: OptimizerState,
               key: chex.PRNGKey,
               mcmc_width: jnp.ndarray) -> StepResults:
    """Performs one set of MCMC moves and an optimization step.

    Args:
      data: batch of MCMC configurations.
      params: network parameters.
      state: optimizer internal state.
      key: JAX RNG state.
      mcmc_width: width of MCMC move proposal. See mcmc.make_mcmc_step.

    Returns:
      Tuple of (data, params, state, loss, aux_data, pmove).
        data: Updated MCMC configurations drawn from the network given the
          *input* network parameters.
        params: updated network parameters after the gradient update.
        state: updated optimization state.
        loss: energy of system based on input network parameters averaged over
          the entire set of MCMC configurations.
        aux_data: AuxiliaryLossData object also returned from evaluating the
          loss of the system.
        pmove: probability that a proposed MCMC move was accepted.
    """


def make_training_step(
    mcmc_step,
    optimizer_step: OptUpdate,
) -> Step:
  """Factory to create traning step for non-KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    optimizer_step: OptUpdate callable which evaluates the forward and backward
      passes and updates the parameters and optimizer state, as required.

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  @functools.partial(constants.pmap, donate_argnums=(1, 2, 3, 4))
  def step(data: jnp.ndarray,
           params: networks.ParamTree, state: Optional[optax.OptState],
           key: chex.PRNGKey, mcmc_width: jnp.ndarray) -> StepResults:
    """A full update iteration (except for KFAC): MCMC steps + optimization."""
    # MCMC loop
    mcmc_key, loss_key = jax.random.split(key, num=2)
    data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)

    # Optimization step
    new_params, state, loss, aux_data = optimizer_step(params, data, state,
                                                       loss_key)
    return data, new_params, state, loss, aux_data, pmove

  return step


def make_kfac_training_step(mcmc_step, damping: float,
                            optimizer: kfac_optim.Optimizer) -> Step:
  """Factory to create traning step for KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    damping: value of damping to use for each KFAC update step.
    optimizer: KFAC optimizer instance.

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
  shared_mom = kfac_utils.replicate_all_local_devices(jnp.zeros([]))
  shared_damping = kfac_utils.replicate_all_local_devices(jnp.asarray(damping))

  def step(data: jnp.ndarray,
           params: networks.ParamTree, state: kfac_optim.State,
           key: chex.PRNGKey, mcmc_width: jnp.ndarray) -> StepResults:
    """A full update iteration for KFAC: MCMC steps + optimization."""
    # KFAC requires control of the loss and gradient eval, so everything called
    # here must be already pmapped.

    # MCMC loop
    mcmc_keys, loss_keys = kfac_utils.p_split(key)
    new_data, pmove = mcmc_step(params, data, mcmc_keys, mcmc_width)

    # Optimization step
    new_params, state, stats = optimizer.step(
        params=params,
        state=state,
        rng=loss_keys,
        data_iterator=iter([new_data]),
        momentum=shared_mom,
        damping=shared_damping)
    return new_data, new_params, state, stats['loss'], stats['aux'], pmove

  return step


def train(cfg: ml_collections.ConfigDict):
  """Runs training loop for QMC.

  Args:
    cfg: ConfigDict containing the system and training parameters to run on. See
      base_config.default for more details.

  Raises:
    ValueError: if an illegal or unsupported value in cfg is detected.
  """
  # Device logging
  num_devices = jax.local_device_count()
  logging.info('Starting QMC with %i XLA devices', num_devices)
  if cfg.batch_size % num_devices != 0:
    raise ValueError('Batch size must be divisible by number of devices, '
                     'got batch size {} for {} devices.'.format(
                         cfg.batch_size, num_devices))
  if cfg.system.ndim != 3:
    # The network (at least the input feature construction) and initial MCMC
    # molecule configuration (via system.Atom) assume 3D systems. This can be
    # lifted with a little work.
    raise ValueError('Only 3D systems are currently supported.')
  device_batch_size = cfg.batch_size // num_devices  # batch size per device
  data_shape = (num_devices, device_batch_size)

  # Check if mol is a pyscf molecule and convert to internal representation
  if cfg.system.pyscf_mol:
    cfg.update(
        system.pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))

  # Convert mol config into array of atomic positions and charges
  atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
  charges = jnp.array([atom.charge for atom in cfg.system.molecule])
  nspins = cfg.system.electrons

  if cfg.debug.deterministic:
    seed = 23
  else:
    seed = int(1e6 * time.time())
  key = jax.random.PRNGKey(seed)

  # Create parameters, network, and vmaped/pmaped derivations

  if cfg.pretrain.method == 'direct_init' or (
      cfg.pretrain.method == 'hf' and cfg.pretrain.iterations > 0):
    hartree_fock = pretrain.get_hf(
        pyscf_mol=cfg.system.get('pyscf_mol'),
        molecule=cfg.system.molecule,
        nspins=nspins,
        restricted=False,
        basis=cfg.pretrain.basis)

  hf_solution = hartree_fock if cfg.pretrain.method == 'direct_init' else None
  network_init, signed_network, network_options = networks.make_fermi_net(
      atoms,
      nspins,
      charges,
      envelope_type=cfg.network.envelope_type,
      bias_orbitals=cfg.network.bias_orbitals,
      use_last_layer=cfg.network.use_last_layer,
      hf_solution=hf_solution,
      full_det=cfg.network.full_det,
      **cfg.network.detnet)
  key, subkey = jax.random.split(key)
  params = network_init(subkey)
  params = kfac_utils.replicate_all_local_devices(params)
  # Often just need log|psi(x)|.
  network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]  # type: networks.LogFermiNetLike
  batch_network = jax.vmap(
      network, in_axes=(None, 0), out_axes=0)  # batched network

  # Set up checkpointing and restore params/data if necessary
  # Mirror behaviour of checkpoints in TF FermiNet.
  # Checkpoints are saved to save_path.
  # When restoring, we first check for a checkpoint in save_path. If none are
  # found, then we check in restore_path.  This enables calculations to be
  # started from a previous calculation but then resume from their own
  # checkpoints in the event of pre-emption.

  ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)
  ckpt_restore_path = checkpoint.get_restore_path(cfg.log.restore_path)

  ckpt_restore_filename = (
      checkpoint.find_last_checkpoint(ckpt_save_path) or
      checkpoint.find_last_checkpoint(ckpt_restore_path))

  if ckpt_restore_filename:
    t_init, data, params, opt_state_ckpt, mcmc_width_ckpt = checkpoint.restore(
        ckpt_restore_filename, cfg.batch_size)
  else:
    logging.info('No checkpoint found. Training new model.')
    key, subkey = jax.random.split(key)
    data = init_electrons(subkey, cfg.system.molecule, cfg.system.electrons,
                          cfg.batch_size, cfg.mcmc.init_width)
    data = jnp.reshape(data, data_shape + data.shape[1:])
    data = kfac_utils.broadcast_all_local_devices(data)
    t_init = 0
    opt_state_ckpt = None
    mcmc_width_ckpt = None

  # Set up logging
  train_schema = ['step', 'energy', 'ewmean', 'ewvar', 'pmove']

  # Initialisation done. We now want to have different PRNG streams on each
  # device. Shard the key over devices
  sharded_key = kfac_utils.make_different_rng_key_on_all_devices(key)

  # Pretraining to match Hartree-Fock

  if (t_init == 0 and cfg.pretrain.method == 'hf' and
      cfg.pretrain.iterations > 0):
    orbitals = functools.partial(
        networks.fermi_net_orbitals,
        atoms=atoms,
        nspins=cfg.system.electrons,
        options=network_options,
    )
    batch_orbitals = jax.vmap(
        lambda params, data: orbitals(params, data)[0],
        in_axes=(None, 0),
        out_axes=0)
    sharded_key, subkeys = kfac_utils.p_split(sharded_key)
    params, data = pretrain.pretrain_hartree_fock(
        params=params,
        data=data,
        batch_network=batch_network,
        batch_orbitals=batch_orbitals,
        network_options=network_options,
        sharded_key=subkeys,
        atoms=atoms,
        electrons=cfg.system.electrons,
        scf_approx=hartree_fock,
        iterations=cfg.pretrain.iterations)

  # Main training

  # Construct MCMC step
  atoms_to_mcmc = atoms if cfg.mcmc.scale_by_nuclear_distance else None
  mcmc_step = mcmc.make_mcmc_step(
      batch_network,
      device_batch_size,
      steps=cfg.mcmc.steps,
      atoms=atoms_to_mcmc,
      one_electron_moves=cfg.mcmc.one_electron,
  )
  # Construct loss and optimizer
  if cfg.system.make_local_energy_fn:
    local_energy_module, local_energy_fn = (
        cfg.system.make_local_energy_fn.rsplit('.', maxsplit=1))
    local_energy_module = importlib.import_module(local_energy_module)
    make_local_energy = getattr(local_energy_module, local_energy_fn)  # type: hamiltonian.MakeLocalEnergy
    local_energy = make_local_energy(
        f=signed_network,
        atoms=atoms,
        charges=charges,
        nspins=nspins,
        use_scan=False,
        **cfg.system.make_local_energy_kwargs)
  else:
    local_energy = hamiltonian.local_energy(
        f=signed_network,
        atoms=atoms,
        charges=charges,
        nspins=nspins,
        use_scan=False)
  total_energy = qmc_loss_functions.make_loss(
      network,
      local_energy,
      clip_local_energy=cfg.optim.clip_el)
  # Compute the learning rate
  def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
    return cfg.optim.lr.rate * jnp.power(
        (1.0 / (1.0 + (t_/cfg.optim.lr.delay))), cfg.optim.lr.decay)
  # Differentiate wrt parameters (argument 0)
  val_and_grad = jax.value_and_grad(total_energy, argnums=0, has_aux=True)

  # Construct and setup optimizer
  if cfg.optim.optimizer == 'none':
    optimizer = None
  elif cfg.optim.optimizer == 'adam':
    optimizer = optax.chain(
        optax.scale_by_adam(**cfg.optim.adam),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.))
  elif cfg.optim.optimizer == 'lamb':
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(eps=1e-7),
        optax.scale_by_trust_ratio(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1))
  elif cfg.optim.optimizer == 'kfac':
    optimizer = kfac_optim.Optimizer(
        val_and_grad,
        l2_reg=cfg.optim.kfac.l2_reg,
        norm_constraint=cfg.optim.kfac.norm_constraint,
        value_func_has_aux=True,
        value_func_has_rng=True,
        learning_rate_schedule=learning_rate_schedule,
        curvature_ema=cfg.optim.kfac.cov_ema_decay,
        inverse_update_period=cfg.optim.kfac.invert_every,
        min_damping=cfg.optim.kfac.min_damping,
        num_burnin_steps=0,
        register_only_generic=cfg.optim.kfac.register_only_generic,
        estimation_mode='fisher_exact',
        multi_device=True,
        pmap_axis_name=constants.PMAP_AXIS_NAME
        # debug=True
    )
    sharded_key, subkeys = kfac_utils.p_split(sharded_key)
    opt_state = optimizer.init(params, subkeys, data)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
  else:
    raise ValueError(f'Not a recognized optimizer: {cfg.optim.optimizer}')

  if not optimizer:
    opt_state = None

    def energy_eval(params: networks.ParamTree, data: jnp.ndarray,
                    opt_state: Optional[optax.OptState],
                    key: chex.PRNGKey) -> OptUpdateResults:
      loss, aux_data = total_energy(params, key, data)
      return params, opt_state, loss, aux_data

    step = make_training_step(
        mcmc_step=mcmc_step,
        optimizer_step=energy_eval)
  elif isinstance(optimizer, optax.GradientTransformation):
    # optax/optax-compatible optimizer (ADAM, LAMB, ...)
    opt_state = jax.pmap(optimizer.init)(params)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state

    def opt_update(params: networks.ParamTree, data: jnp.ndarray,
                   opt_state: Optional[optax.OptState],
                   key: chex.PRNGKey) -> OptUpdateResults:
      (loss, aux_data), grad = val_and_grad(params, key, data)
      grad = constants.pmean(grad)
      updates, opt_state = optimizer.update(grad, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss, aux_data

    step = make_training_step(mcmc_step=mcmc_step, optimizer_step=opt_update)
  elif isinstance(optimizer, kfac_optim.Optimizer):
    step = make_kfac_training_step(
        mcmc_step=mcmc_step,
        damping=cfg.optim.kfac.damping,
        optimizer=optimizer)
  else:
    raise ValueError(f'Unknown optimizer: {optimizer}')

  if mcmc_width_ckpt is not None:
    mcmc_width = kfac_utils.replicate_all_local_devices(mcmc_width_ckpt[0])
  else:
    mcmc_width = kfac_utils.replicate_all_local_devices(
        jnp.asarray(cfg.mcmc.move_width))
  pmoves = np.zeros(cfg.mcmc.adapt_frequency)

  if t_init == 0:
    logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)

    def null_update(params: networks.ParamTree, data: jnp.ndarray,
                    opt_state: Optional[optax.OptState],
                    key: chex.PRNGKey) -> OptUpdateResults:
      del data, key
      return params, opt_state, jnp.zeros(1), None

    burn_in_step = make_training_step(
        mcmc_step=mcmc_step, optimizer_step=null_update)

    for t in range(cfg.mcmc.burn_in):
      sharded_key, subkeys = kfac_utils.p_split(sharded_key)
      data, params, *_ = burn_in_step(
          data=data,
          params=params,
          state=None,
          key=subkeys,
          mcmc_width=mcmc_width)
    logging.info('Completed burn-in MCMC steps')
    sharded_key, subkeys = kfac_utils.p_split(sharded_key)
    ptotal_energy = constants.pmap(total_energy)
    initial_energy, _ = ptotal_energy(params, subkeys, data)
    logging.info('Initial energy: %03.4f E_h', initial_energy[0])

  time_of_last_ckpt = time.time()
  weighted_stats = None

  if cfg.optim.optimizer == 'none' and opt_state_ckpt is not None:
    # If opt_state_ckpt is None, then we're restarting from a previous inference
    # run (most likely due to preemption) and so should continue from the last
    # iteration in the checkpoint. Otherwise, starting an inference run from a
    # training run.
    logging.info('No optimizer provided. Assuming inference run.')
    logging.info('Setting initial iteration to 0.')
    t_init = 0

  with writers.Writer(
      name='train_stats',
      schema=train_schema,
      directory=ckpt_save_path,
      iteration_key=None,
      log=False) as writer:
    # Main training loop
    for t in range(t_init, cfg.optim.iterations):
      sharded_key, subkeys = kfac_utils.p_split(sharded_key)
      data, params, opt_state, loss, unused_aux_data, pmove = step(
          data=data,
          params=params,
          state=opt_state,
          key=subkeys,
          mcmc_width=mcmc_width)

      # due to pmean, loss, and pmove should be the same across
      # devices.
      loss = loss[0]
      # per batch variance isn't informative. Use weighted mean and variance
      # instead.
      weighted_stats = statistics.exponentialy_weighted_stats(
          alpha=0.1, observation=loss, previous_stats=weighted_stats)
      pmove = pmove[0]

      # Update MCMC move width
      if t > 0 and t % cfg.mcmc.adapt_frequency == 0:
        if np.mean(pmoves) > 0.55:
          mcmc_width *= 1.1
        if np.mean(pmoves) < 0.5:
          mcmc_width /= 1.1
        pmoves[:] = 0
      pmoves[t%cfg.mcmc.adapt_frequency] = pmove

      if cfg.debug.check_nan:
        tree = {'params': params, 'loss': loss}
        if cfg.optim.optimizer != 'none':
          tree['optim'] = opt_state
        chex.assert_tree_all_finite(tree)

      # Logging
      if t % cfg.log.stats_frequency == 0:
        logging.info(
            'Step %05d: %03.4f E_h, exp. variance=%03.4f E_h^2, pmove=%0.2f', t,
            loss, weighted_stats.variance, pmove)
        writer.write(
            t,
            step=t,
            energy=np.asarray(loss),
            ewmean=np.asarray(weighted_stats.mean),
            ewvar=np.asarray(weighted_stats.variance),
            pmove=np.asarray(pmove))

      # Checkpointing
      if time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60:
        checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width)
        time_of_last_ckpt = time.time()
