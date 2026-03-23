# Copyright 2022 The JAX Authors.
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

import importlib
from importlib import metadata
import json
import os
import warnings

from absl.testing import absltest
import jax
from jax._src import api
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb
import jaxlib

jax.config.parse_flags_with_absl()

_DEBUG_ENV_VAR = 'JAXCI_NIGHTLY_WHEEL_RESOLUTION_DEBUG'


def _distribution_versions() -> dict[str, str]:
  versions = {}
  for dist_name in (
      'jax',
      'jaxlib',
      'jax-cuda12-plugin',
      'jax-cuda12-pjrt',
      'jax-cuda13-plugin',
      'jax-cuda13-pjrt',
  ):
    try:
      versions[dist_name] = metadata.version(dist_name)
    except metadata.PackageNotFoundError:
      versions[dist_name] = '<not installed>'
  return versions


def _module_locations() -> dict[str, str]:
  locations = {
      'jax': getattr(jax, '__file__', '<missing>'),
      'jaxlib': getattr(jaxlib, '__file__', '<missing>'),
  }
  for module_name in (
      'jax_cuda12_plugin',
      'jax_cuda12_pjrt',
      'jax_cuda13_plugin',
      'jax_cuda13_pjrt',
  ):
    try:
      module = importlib.import_module(module_name)
      locations[module_name] = getattr(module, '__file__', '<missing>')
    except Exception as exc:  # pylint: disable=broad-except
      locations[module_name] = f'<import failed: {type(exc).__name__}: {exc}>'
  return locations


class ClearBackendsTest(jtu.JaxTestCase):

  def test_clear_backends(self):
    if os.environ.get(_DEBUG_ENV_VAR) == '1':
      self.skipTest('Debug target only needs the runtime dump test.')

    g = jax.jit(lambda x, y: x * y)
    self.assertEqual(g(1, 2), 2)
    self.assertNotEmpty(xb.get_backend().live_executables())
    api.clear_backends()
    self.assertEmpty(xb.get_backend().live_executables())
    self.assertEqual(g(1, 2), 2)

  def test_dump_runtime_state(self):
    if os.environ.get(_DEBUG_ENV_VAR) != '1':
      self.skipTest('Runtime dump is only for the dedicated debug target.')

    print('=== Nightly wheel resolution debug ===')
    print('cwd:', os.getcwd())
    print('TEST_SRCDIR:', os.environ.get('TEST_SRCDIR', '<unset>'))
    print('PYTHON_RUNFILES:', os.environ.get('PYTHON_RUNFILES', '<unset>'))
    print('Distribution versions:')
    print(json.dumps(_distribution_versions(), indent=2, sort_keys=True))
    print('Module locations:')
    print(json.dumps(_module_locations(), indent=2, sort_keys=True))

    warning_payload = []
    runtime_payload = {}
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      runtime_payload['default_backend'] = jax.default_backend()
      runtime_payload['device_count'] = jax.device_count()
      runtime_payload['devices'] = [str(device) for device in jax.devices()]
      try:
        runtime_payload['gpu_devices'] = [
            str(device) for device in jax.devices('gpu')
        ]
      except Exception as exc:  # pylint: disable=broad-except
        runtime_payload['gpu_devices_error'] = (
            f'{type(exc).__name__}: {exc}'
        )
      runtime_payload['backend_keys'] = sorted(xb.backends().keys())
      runtime_payload['backend_errors'] = {
          key: str(value) for key, value in xb._backend_errors.items()
      }
      for warning in caught:
        warning_payload.append(
            {
                'category': warning.category.__name__,
                'message': str(warning.message),
            }
        )

    print('Runtime payload:')
    print(json.dumps(runtime_payload, indent=2, sort_keys=True))
    print('Captured warnings:')
    print(json.dumps(warning_payload, indent=2, sort_keys=True))

    self.assertNotEmpty(jax.devices())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
