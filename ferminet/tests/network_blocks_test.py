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
"""Tests for ferminet.network_blocks."""

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import network_blocks
import numpy as np
import jax
from jax import numpy as jnp


class NetworkBlocksTest(parameterized.TestCase):

  @parameterized.parameters([
      {
          'sizes': [],
          'expected_indices': []
      },
      {
          'sizes': [3],
          'expected_indices': []
      },
      {
          'sizes': [3, 0],
          'expected_indices': [3]
      },
      {
          'sizes': [3, 6],
          'expected_indices': [3]
      },
      {
          'sizes': [3, 6, 0],
          'expected_indices': [3, 9]
      },
      {
          'sizes': [2, 0, 6],
          'expected_indices': [2, 2]
      },
  ])
  def test_array_partitions(self, sizes, expected_indices):
    self.assertEqual(network_blocks.array_partitions(sizes), expected_indices)

  @parameterized.parameters(
      {'shape': shape} for shape in [(1, 1, 1), (10, 2, 2), (10, 3, 3)])
  def test_slogdet(self, shape, dtype=np.float32):
    a = np.random.normal(size=shape).astype(dtype)
    s1, ld1 = network_blocks.slogdet(a)
    s2, ld2 = np.linalg.slogdet(a)
    np.testing.assert_allclose(s1, s2, atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(ld1, ld2, atol=1E-5, rtol=1E-5)

  def test_log_linear(self):
    w1 = np.random.randn(16, 32)
    w2 = np.random.randn(32, 16)
    logx = np.random.randn(2, 16)
    x = jnp.exp(logx)

    def network(logx):
      logh1, sign1 = network_blocks.log_linear_layer(logx,
                                                     w1,
                                                     activation='tanh')
      logh2, sign2 = network_blocks.log_linear_layer(logh1,
                                                     w2,
                                                     prev_sign=sign1,
                                                     activation='tanh')
      h2_fromlog = sign2 * jnp.exp(logh2)
      return h2_fromlog

    h2_fromlog = jax.vmap(network, in_axes=0, out_axes=0)(logx)

    h1 = jax.nn.tanh(jnp.dot(x, w1))
    h2 = jax.nn.tanh(jnp.dot(h1, w2))

    self.assertTrue(jnp.allclose(h2_fromlog, h2, atol=1e-4))


if __name__ == '__main__':
  absltest.main()
