## Copyright 2020 DeepMind Technologies Limited.
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
"""Main wrapper for FermiNet in JAX."""

from absl import app
from absl import flags
from absl import logging
from ferminet import base_config
from ferminet import train
from ml_collections.config_flags import config_flags
from copy import deepcopy

# internal imports

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "ferminet", "sota experiment name")
flags.DEFINE_bool("use_wandb", False, "if true, use wandb")
flags.DEFINE_bool("run_all", False, "if true, run all system")
config_flags.DEFINE_config_file('config', None, 'Path to config file.')


def main(_):
  raw_cfg = FLAGS.config
  if FLAGS.run_all:
    system_type = raw_cfg.config_module[1:]
    for system_name in train.exact_energy[system_type].keys():
      if system_type == "atom":
        raw_cfg.system.atom = system_name
      else:
        raw_cfg.system.molecule_name = system_name
      cfg = base_config.resolve(deepcopy(raw_cfg))
      logging.info("System config:\n\n%s", cfg)
      try:
        train.train(cfg)
      except Exception:
        logging.exception("train failed")
  else:
    cfg = base_config.resolve(raw_cfg)
    logging.info("System config:\n\n%s", cfg)
    train.train(cfg)


if __name__ == '__main__':
  app.run(main)
