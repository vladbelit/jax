# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime

from absl.testing import absltest

import jax
from jax._src import test_util as jtu
from jax._src import tpu_custom_call

jax.config.parse_flags_with_absl()


class _FakeModuleContext:

  def __init__(self, backend=None):
    self._backend = backend

  def get_backend(self, optional=False):
    del optional
    return self._backend


class _FakeLoweringRuleContext:

  def __init__(self, *, backend=None, forward_compatible=False):
    self.module_context = _FakeModuleContext(backend)
    self._forward_compatible = forward_compatible

  def is_forward_compat(self):
    return self._forward_compatible


class _FakeBackend:

  def __init__(self, platform_version):
    self.platform_version = platform_version


def _make_tpu_backend(year, month, day):
  build_time = datetime.datetime(
      year, month, day, 12, tzinfo=datetime.timezone.utc
  )
  timestamp = int(build_time.timestamp())
  return _FakeBackend(
      'PJRT C API\n'
      'TFRT TPU v6e\n'
      f'Built on test date ({timestamp}) cl/123'
  )


class TpuCustomCallTest(jtu.JaxTestCase):

  def test_get_ir_version_uses_v11_before_may_1_libtpu(self):
    ctx = _FakeLoweringRuleContext(backend=_make_tpu_backend(2026, 4, 30))

    self.assertEqual(tpu_custom_call.get_ir_version(ctx), 11)

  def test_get_ir_version_uses_v12_before_may_4_libtpu(self):
    ctx = _FakeLoweringRuleContext(backend=_make_tpu_backend(2026, 5, 1))

    self.assertEqual(tpu_custom_call.get_ir_version(ctx), 12)

  def test_get_ir_version_uses_latest_for_may_4_libtpu(self):
    ctx = _FakeLoweringRuleContext(backend=_make_tpu_backend(2026, 5, 4))

    self.assertIsNone(tpu_custom_call.get_ir_version(ctx))

  def test_get_ir_version_uses_v12_for_forward_compat(self):
    ctx = _FakeLoweringRuleContext(
        backend=_make_tpu_backend(2026, 5, 4), forward_compatible=True
    )

    self.assertEqual(tpu_custom_call.get_ir_version(ctx), 12)

  def test_get_ir_version_uses_v12_without_backend(self):
    ctx = _FakeLoweringRuleContext()

    self.assertEqual(tpu_custom_call.get_ir_version(ctx), 12)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
