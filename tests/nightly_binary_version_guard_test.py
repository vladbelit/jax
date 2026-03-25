# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

from importlib import metadata

from absl.testing import absltest
import jax
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


def _distribution_version(distribution_name: str) -> str:
  try:
    return metadata.version(distribution_name)
  except metadata.PackageNotFoundError:
    return '<not installed>'


class NightlyBinaryVersionGuardTest(jtu.JaxTestCase):

  def test_nightly_binary_versions_match_expected_override(self):
    expected_binary_version = os.environ.get('JAXCI_EXPECTED_BINARY_VERSION')
    self.assertIsNotNone(
        expected_binary_version,
        'JAXCI_EXPECTED_BINARY_VERSION must be set for the nightly guard.',
    )

    package_names = [
        name
        for name in os.environ.get('JAXCI_EXPECTED_BINARY_PACKAGES', '').split(',')
        if name
    ]
    self.assertLen(
        package_names,
        3,
        'JAXCI_EXPECTED_BINARY_PACKAGES must describe jaxlib, plugin, and '
        'PJRT distributions.',
    )

    distribution_versions = {
        package_name: _distribution_version(package_name)
        for package_name in package_names
    }
    print('Nightly binary version guard distribution versions:')
    print(json.dumps(distribution_versions, indent=2, sort_keys=True))

    missing_distributions = [
        package_name
        for package_name, version in distribution_versions.items()
        if version == '<not installed>'
    ]
    self.assertEmpty(
        missing_distributions,
        'Expected nightly binary distributions are not installed: '
        f'{missing_distributions}',
    )

    mismatched_versions = {
        package_name: version
        for package_name, version in distribution_versions.items()
        if version != expected_binary_version
    }
    self.assertEmpty(
        mismatched_versions,
        'Installed binary distributions do not match the nightly override '
        f'version {expected_binary_version}: {mismatched_versions}',
    )

    try:
      jax.default_backend()
    except Exception as exc:
      self.fail(
          'jax runtime initialization failed even though package metadata matched the '
          f'expected nightly binary version {expected_binary_version}: {exc}'
      )

    runtime_payload = {
        'default_backend': jax.default_backend(),
        'device_count': jax.device_count(),
        'devices': [str(device) for device in jax.devices()],
    }
    print('Nightly binary version guard runtime payload:')
    print(json.dumps(runtime_payload, indent=2, sort_keys=True))
    self.assertNotEmpty(runtime_payload['devices'])


if __name__ == '__main__':
  absltest.main()
