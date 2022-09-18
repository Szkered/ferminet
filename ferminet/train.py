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

import collections
import datetime
import functools
import importlib
import time
from typing import Optional, Sequence, Tuple, Union
import requests

import git
from absl import logging, flags
import chex
from ferminet import checkpoint
from ferminet import constants
from ferminet import curvature_tags_and_blocks
from ferminet import envelopes
from ferminet import hamiltonian
from ferminet import loss as qmc_loss_functions
from ferminet import mcmc
from ferminet import networks
from ferminet import pretrain
from ferminet.utils import multi_host
from ferminet.utils import statistics
from ferminet.utils import system
from ferminet.utils import writers
import jax
import jax.numpy as jnp
import kfac_jax
import ml_collections
import numpy as np
import optax
from typing_extensions import Protocol
import wandb

FLAGS = flags.FLAGS

# Hatree-Fock energy (no corr)
hf_energy = {
    "atom": {
        "Li": -7.432747,
        "Be": -14.57301,
        "B": -24.53316,
        "C": -37.6938,
        "N": -54.4047,
        "O": -74.8192,
        "F": -99.4168,
        "Ne": -128.5479,
    },
    "diatomic": {
        "LiH": -7.98737,
        "Li2": -14.87155,
        "CO": -112.7871,
        "N2": -108.9940,
    },
    "organic": {
        "bicbut": -154.9372,
    }
}

# TODO(@shizk): geometries of bicbut is different from the paper?
# CCSD(T)/CBS or other accuracy values
exact_energy = {
    "atom": {
        "Li": -7.47806032,
        "Be": -14.66736,
        "B": -24.65391,
        "C": -37.8450,
        "N": -54.5892,
        "O": -75.0673,
        "F": -99.7339,
        "Ne": -128.9376,
    },
    "diatomic": {
        "LiH": -8.070548,
        "Li2": -14.9954,
        "CO": -113.3255,
        "N2": -109.5423,
    },
    "organic": {
        "bicbut": -155.9575,
    }
}


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
      electron_positions + init_width *
      jax.random.normal(subkey, shape=(batch_size, electron_positions.size)))


# All optimizer states (KFAC and optax-based).
OptimizerState = Union[optax.OptState, kfac_jax.optimizer.OptimizerState]
OptUpdateResults = Tuple[networks.ParamTree, Optional[OptimizerState],
                         jnp.ndarray,
                         Optional[qmc_loss_functions.AuxiliaryLossData]]


class OptUpdate(Protocol):

  def __call__(self, params: networks.ParamTree, data: jnp.ndarray,
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

  def __call__(self, data: jnp.ndarray, params: networks.ParamTree,
               state: OptimizerState, key: chex.PRNGKey,
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


def null_update(params: networks.ParamTree, data: jnp.ndarray,
                opt_state: Optional[optax.OptState],
                key: chex.PRNGKey) -> OptUpdateResults:
  """Performs an identity operation with an OptUpdate interface."""
  del data, key
  return params, opt_state, jnp.zeros(1), None


def make_opt_update_step(evaluate_loss: qmc_loss_functions.LossFn,
                         optimizer: optax.GradientTransformation) -> OptUpdate:
  """Returns an OptUpdate function for performing a parameter update."""

  # Differentiate wrt parameters (argument 0)
  loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

  def opt_update(params: networks.ParamTree, data: jnp.ndarray,
                 opt_state: Optional[optax.OptState],
                 key: chex.PRNGKey) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters using optax."""
    (loss, aux_data), grad = loss_and_grad(params, key, data)
    grad = constants.pmean(grad)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux_data

  return opt_update


def make_loss_step(evaluate_loss: qmc_loss_functions.LossFn) -> OptUpdate:
  """Returns an OptUpdate function for evaluating the loss."""

  def loss_eval(params: networks.ParamTree, data: Tuple[jnp.ndarray, ...],
                opt_state: Optional[optax.OptState],
                key: chex.PRNGKey) -> OptUpdateResults:
    """Evaluates just the loss and gradients with an OptUpdate interface."""
    loss, aux_data = evaluate_loss(params, key, data)
    return params, opt_state, loss, aux_data

  return loss_eval


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

  @functools.partial(constants.pmap, donate_argnums=(0, 1, 2))
  def step(data: jnp.ndarray, params: networks.ParamTree,
           state: Optional[optax.OptState], key: chex.PRNGKey,
           mcmc_width: jnp.ndarray) -> StepResults:
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
                            optimizer: kfac_jax.Optimizer) -> Step:
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
  shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
  shared_damping = kfac_jax.utils.replicate_all_local_devices(
      jnp.asarray(damping))

  def step(data: jnp.ndarray, params: networks.ParamTree,
           state: kfac_jax.optimizer.OptimizerState, key: chex.PRNGKey,
           mcmc_width: jnp.ndarray) -> StepResults:
    """A full update iteration for KFAC: MCMC steps + optimization."""
    # KFAC requires control of the loss and gradient eval, so everything called
    # here must be already pmapped.

    # MCMC loop
    mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
    new_data, pmove = mcmc_step(params, data, mcmc_keys, mcmc_width)

    # Optimization step
    new_params, state, stats = optimizer.step(params=params,
                                              state=state,
                                              rng=loss_keys,
                                              data_iterator=iter([new_data]),
                                              momentum=shared_mom,
                                              damping=shared_damping)
    return new_data, new_params, state, stats['loss'], stats['aux'], pmove

  return step


def flatten_cfg(d, parent_key='', sep='.'):
  items = []
  for k, v in d.items():
    new_key = parent_key + sep + k if parent_key else k
    if isinstance(v, collections.MutableMapping):
      items.extend(flatten_cfg(v, new_key, sep=sep).items())
    else:
      items.append((new_key, v))
  return dict(items)


def train(cfg: ml_collections.ConfigDict, writer_manager=None):
  """Runs training loop for QMC.

  Args:
    cfg: ConfigDict containing the system and training parameters to run on. See
      base_config.default for more details.
    writer_manager: context manager with a write method for logging output. If
      None, a default writer (ferminet.utils.writers.Writer) is used.

  Raises:
    ValueError: if an illegal or unsupported value in cfg is detected.
  """
  system_type = cfg.config_module[1:]
  flat_cfg = flatten_cfg(cfg.to_dict())
  flat_cfg = {
      k: str(v) if isinstance(v, tuple) else v
      for k, v in flat_cfg.items()
      if v is not None
  }

  if "system.molecule_name" in flat_cfg.keys():  # is molecule
    system_name = flat_cfg["system.molecule_name"]
  elif "system.atom" in flat_cfg.keys():  # is atom
    system_name = flat_cfg["system.atom"]
  else:
    raise RuntimeError

  exp_name = FLAGS.exp_name + "-" + system_name
  if FLAGS.use_wandb:
    repo = git.Repo(search_parent_directories=True)
    branch_name = repo.active_branch.name
    commit_sha = repo.head.object.hexsha
    flat_cfg['branch'] = branch_name
    flat_cfg['commit'] = commit_sha
    try:
      flat_cfg['tpu_vm'] = requests.get(
          "http://metadata/computeMetadata/v1/instance/description",
          headers={
              'Metadata-Flavor': 'Google'
          }).text
    except Exception as e:
      logging.warn(e)
    wandb.init(name=exp_name, project="QMC", config=flat_cfg)

  # Device logging
  num_devices = jax.local_device_count()
  num_hosts = jax.device_count() // num_devices
  logging.info('Starting QMC with %i XLA devices per host '
               'across %i hosts.', num_devices, num_hosts)
  if cfg.batch_size % (num_devices * num_hosts) != 0:
    raise ValueError('Batch size must be divisible by number of devices, '
                     f'got batch size {cfg.batch_size} for '
                     f'{num_devices * num_hosts} devices.')
  if cfg.system.ndim != 3:
    # The network (at least the input feature construction) and initial MCMC
    # molecule configuration (via system.Atom) assume 3D systems. This can be
    # lifted with a little work.
    raise ValueError('Only 3D systems are currently supported.')
  host_batch_size = cfg.batch_size // num_hosts  # batch size per host
  device_batch_size = host_batch_size // num_devices  # batch size per device
  data_shape = (num_devices, device_batch_size)

  # Check if mol is a pyscf molecule and convert to internal representation
  if cfg.system.pyscf_mol:
    cfg.update(system.pyscf_mol_to_internal_representation(
        cfg.system.pyscf_mol))

  # Convert mol config into array of atomic positions and charges
  atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
  charges = jnp.array([atom.charge for atom in cfg.system.molecule])
  nspins = cfg.system.electrons

  if cfg.debug.deterministic:
    seed = 23
  else:
    seed = 1e6 * time.time()
    seed = int(multi_host.broadcast_to_hosts(seed))
  key = jax.random.PRNGKey(seed)

  # Create parameters, network, and vmaped/pmaped derivations

  if cfg.pretrain.method == 'direct_init' or (cfg.pretrain.method == 'hf' and
                                              cfg.pretrain.iterations > 0):
    hartree_fock = pretrain.get_hf(pyscf_mol=cfg.system.get('pyscf_mol'),
                                   molecule=cfg.system.molecule,
                                   nspins=nspins,
                                   restricted=False,
                                   basis=cfg.pretrain.basis)
    # broadcast the result of PySCF from host 0 to all other hosts
    hartree_fock.mean_field.mo_coeff = tuple([
        multi_host.broadcast_to_hosts(x)
        for x in hartree_fock.mean_field.mo_coeff
    ])

  hf_solution = hartree_fock if cfg.pretrain.method == 'direct_init' else None

  if cfg.network.make_feature_layer_fn:
    feature_layer_module, feature_layer_fn = (
        cfg.network.make_feature_layer_fn.rsplit('.', maxsplit=1))
    feature_layer_module = importlib.import_module(feature_layer_module)
    make_feature_layer = getattr(feature_layer_module, feature_layer_fn)
    feature_layer = make_feature_layer(
        charges, cfg.system.electrons, cfg.system.ndim,
        **cfg.network.make_feature_layer_kwargs)  # type: networks.FeatureLayer
  else:
    feature_layer = networks.make_ferminet_features(
        charges,
        cfg.system.electrons,
        cfg.system.ndim,
    )

  if cfg.network.make_envelope_fn:
    envelope_module, envelope_fn = (cfg.network.make_envelope_fn.rsplit(
        '.', maxsplit=1))
    envelope_module = importlib.import_module(envelope_module)
    make_envelope = getattr(envelope_module, envelope_fn)
    envelope = make_envelope(
        **cfg.network.make_envelope_kwargs)  # type: envelopes.Envelope
  else:
    envelope = envelopes.make_isotropic_envelope()

  network_init, signed_network, network_options = networks.make_fermi_net(
      atoms,
      nspins,
      charges,
      envelope=envelope,
      feature_layer=feature_layer,
      hf_solution=hf_solution,
      cfg=cfg,
  )
  key, subkey = jax.random.split(key)
  params = network_init(subkey)
  params = kfac_jax.utils.replicate_all_local_devices(params)
  # Often just need log|psi(x)|.
  network = lambda *args, **kwargs: signed_network(*args, **kwargs)[
      1]  # type: networks.LogFermiNetLike
  batch_network = jax.vmap(network, in_axes=(None, 0),
                           out_axes=0)  # batched network

  # Set up checkpointing and restore params/data if necessary
  # Mirror behaviour of checkpoints in TF FermiNet.
  # Checkpoints are saved to save_path.
  # When restoring, we first check for a checkpoint in save_path. If none are
  # found, then we check in restore_path.  This enables calculations to be
  # started from a previous calculation but then resume from their own
  # checkpoints in the event of pre-emption.
  timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
  save_path = cfg.log.save_path or f"{exp_name}_{timestamp}"
  ckpt_save_path = checkpoint.create_save_path(save_path)
  ckpt_restore_path = checkpoint.get_restore_path(cfg.log.restore_path)

  ckpt_restore_filename = (checkpoint.find_last_checkpoint(ckpt_save_path) or
                           checkpoint.find_last_checkpoint(ckpt_restore_path))

  if ckpt_restore_filename:
    t_init, data, params, opt_state_ckpt, mcmc_width_ckpt = checkpoint.restore(
        ckpt_restore_filename, host_batch_size)
  else:
    logging.info('No checkpoint found. Training new model.')
    key, subkey = jax.random.split(key)
    # make sure data on each host is initialized differently
    subkey = jax.random.fold_in(subkey, jax.process_index())
    data = init_electrons(subkey,
                          cfg.system.molecule,
                          cfg.system.electrons,
                          batch_size=host_batch_size,
                          init_width=cfg.mcmc.init_width)
    data = jnp.reshape(data, data_shape + data.shape[1:])
    data = kfac_jax.utils.broadcast_all_local_devices(data)
    t_init = 0
    opt_state_ckpt = None
    mcmc_width_ckpt = None

  # Set up logging
  train_schema = ['step', 'energy', 'ewmean', 'ewvar', 'pmove']

  # Initialisation done. We now want to have different PRNG streams on each
  # device. Shard the key over devices
  sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

  # Pretraining to match Hartree-Fock

  if (t_init == 0 and cfg.pretrain.method == 'hf' and
      cfg.pretrain.iterations > 0):
    orbitals = functools.partial(
        networks.fermi_net_orbitals,
        atoms=atoms,
        nspins=cfg.system.electrons,
        options=network_options,
    )
    batch_orbitals = jax.vmap(lambda params, data: orbitals(params, data)[0],
                              in_axes=(None, 0),
                              out_axes=0)
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
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
      use_sgld=cfg.mcmc.use_sgld,
  )
  # Construct loss and optimizer
  if cfg.system.make_local_energy_fn:
    local_energy_module, local_energy_fn = (
        cfg.system.make_local_energy_fn.rsplit('.', maxsplit=1))
    local_energy_module = importlib.import_module(local_energy_module)
    make_local_energy = getattr(
        local_energy_module,
        local_energy_fn)  # type: hamiltonian.MakeLocalEnergy
    local_energy = make_local_energy(f=signed_network,
                                     atoms=atoms,
                                     charges=charges,
                                     nspins=nspins,
                                     use_scan=False,
                                     **cfg.system.make_local_energy_kwargs)
  else:
    local_energy = hamiltonian.local_energy(f=signed_network,
                                            atoms=atoms,
                                            charges=charges,
                                            nspins=nspins,
                                            use_scan=False)
  evaluate_loss = qmc_loss_functions.make_loss(
      signed_network,
      local_energy,
      clip_local_energy=cfg.optim.clip_el,
      logdet_reg_lambda=cfg.optim.logdet_reg_lambda,
      nci_w_reg_lambda=cfg.optim.nci_w_reg_lambda,
      tau_loss_lambda=cfg.optim.tau_loss_lambda,
  )

  # Compute the learning rate
  def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
    return cfg.optim.lr.rate * jnp.power(
        (1.0 / (1.0 + (t_ / cfg.optim.lr.delay))), cfg.optim.lr.decay)

  # Construct and setup optimizer
  if cfg.optim.optimizer == 'none':
    optimizer = None
  elif cfg.optim.optimizer == 'adam':
    optimizer = optax.chain(optax.scale_by_adam(**cfg.optim.adam),
                            optax.scale_by_schedule(learning_rate_schedule),
                            optax.scale(-1.))
  elif cfg.optim.optimizer == 'lamb':
    optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                            optax.scale_by_adam(eps=1e-7),
                            optax.scale_by_trust_ratio(),
                            optax.scale_by_schedule(learning_rate_schedule),
                            optax.scale(-1))
  elif cfg.optim.optimizer == 'kfac':
    # Differentiate wrt parameters (argument 0)
    val_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)
    optimizer = kfac_jax.Optimizer(
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
        pmap_axis_name=constants.PMAP_AXIS_NAME,
        auto_register_kwargs=dict(
            graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,),
        # debug=True
    )
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    opt_state = optimizer.init(params, subkeys, data)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
  else:
    raise ValueError(f'Not a recognized optimizer: {cfg.optim.optimizer}')

  if not optimizer:
    opt_state = None
    step = make_training_step(mcmc_step=mcmc_step,
                              optimizer_step=make_loss_step(evaluate_loss))
  elif isinstance(optimizer, optax.GradientTransformation):
    # optax/optax-compatible optimizer (ADAM, LAMB, ...)
    opt_state = jax.pmap(optimizer.init)(params)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
    step = make_training_step(mcmc_step=mcmc_step,
                              optimizer_step=make_opt_update_step(
                                  evaluate_loss, optimizer))
  elif isinstance(optimizer, kfac_jax.Optimizer):
    step = make_kfac_training_step(mcmc_step=mcmc_step,
                                   damping=cfg.optim.kfac.damping,
                                   optimizer=optimizer)
  else:
    raise ValueError(f'Unknown optimizer: {optimizer}')

  if mcmc_width_ckpt is not None:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(mcmc_width_ckpt[0])
  else:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(
        jnp.asarray(cfg.mcmc.move_width))
  pmoves = np.zeros(cfg.mcmc.adapt_frequency)

  if t_init == 0:
    logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)

    burn_in_step = make_training_step(mcmc_step=mcmc_step,
                                      optimizer_step=null_update)

    for t in range(cfg.mcmc.burn_in):
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      data, params, *_ = burn_in_step(data,
                                      params,
                                      state=None,
                                      key=subkeys,
                                      mcmc_width=mcmc_width)
    logging.info('Completed burn-in MCMC steps')
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    ptotal_energy = constants.pmap(evaluate_loss)
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

  if writer_manager is None:
    writer_manager = writers.Writer(name='train_stats',
                                    schema=train_schema,
                                    directory=ckpt_save_path,
                                    iteration_key=None,
                                    log=False)
  with writer_manager as writer:
    # Main training loop
    for t in range(t_init, cfg.optim.iterations):
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      data, params, opt_state, loss, aux_data, pmove = step(
          data, params, opt_state, subkeys, mcmc_width)

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
      pmoves[t % cfg.mcmc.adapt_frequency] = pmove

      if cfg.debug.check_nan:
        tree = {'params': params, 'loss': loss}
        if cfg.optim.optimizer != 'none':
          tree['optim'] = opt_state
        chex.assert_tree_all_finite(tree)

      # Logging
      if t % cfg.log.stats_frequency == 0:
        info = dict(
            jax.tree_map(lambda x: float(jnp.nan_to_num(jnp.mean(x))),
                         aux_data))
        info = {**info, **info['stats']}
        del info['stats']
        info["weighted_var"] = weighted_stats.variance
        info["pmove"] = pmove
        if system_name in exact_energy[system_type].keys():
          le = info["local_energy"]
          exact = exact_energy[system_type][system_name]
          hf = hf_energy[system_type][system_name]
          info["error"] = le - exact
          corr_cap = 0.0
          if le < hf:  # corr is only valid when reach hf level
            corr_cap = (le - hf) / (exact - hf) * 100
          info["corr_captured"] = corr_cap

        logging.info(f"Iter: {t}, " +
                     ", ".join([f"{k}: {v:.4f}" for k, v in info.items()]))

        writer.write(t,
                     step=t,
                     energy=np.asarray(loss),
                     ewmean=np.asarray(weighted_stats.mean),
                     ewvar=np.asarray(weighted_stats.variance),
                     pmove=np.asarray(pmove))

        if FLAGS.use_wandb:
          wandb.log(data=info, step=t)

      # Checkpointing
      if time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60:
        checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width)
        time_of_last_ckpt = time.time()

  if FLAGS.use_wandb:
    wandb.finish()
