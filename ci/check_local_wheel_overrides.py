from __future__ import annotations

import argparse
import json
from pathlib import Path

import python_version_repo_utils as repo_utils


EXPECTED_PACKAGES_BY_CUDA = {
    '12': (
        'jaxlib',
        'jax-cuda12-plugin',
        'jax-cuda12-pjrt',
    ),
    '13': (
        'jaxlib',
        'jax-cuda13-plugin',
        'jax-cuda13-pjrt',
    ),
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--cuda-version', required=True)
  parser.add_argument('--github-env')
  return parser.parse_args()


def normalize_cuda_version(cuda_version: str) -> str:
  normalized = cuda_version.split('.', 1)[0]
  if normalized not in EXPECTED_PACKAGES_BY_CUDA:
    raise ValueError(f'Unsupported CUDA version for nightly guard: {cuda_version}')
  return normalized


def wheel_version_from_filename(package_name: str, wheel_name: str) -> str:
  wheel_stem = wheel_name.removesuffix('.whl')
  expected_prefix = package_name.replace('-', '_') + '-'
  if not wheel_stem.startswith(expected_prefix):
    raise ValueError(
        'Wheel name does not match package name: '
        f'{package_name} vs {wheel_name}'
    )
  remainder = wheel_stem.removeprefix(expected_prefix)
  version, _, _ = remainder.partition('-')
  if not version:
    raise ValueError(f'Could not parse wheel version from {wheel_name}')
  return version


def write_github_env(path: Path, env_values: dict[str, str]) -> None:
  with path.open('a', encoding='utf-8') as file:
    for key, value in env_values.items():
      file.write(f'{key}={value}\n')


def main() -> None:
  args = parse_args()
  cuda_version = normalize_cuda_version(args.cuda_version)
  expected_packages = EXPECTED_PACKAGES_BY_CUDA[cuda_version]

  state = repo_utils.load_python_version_repo_state()
  overrides_label = state.py_version_values.get('LOCAL_WHEEL_OVERRIDES_LABEL')
  if not overrides_label:
    raise RuntimeError(
        'LOCAL_WHEEL_OVERRIDES_LABEL is not present in py_version.bzl. '
        'This nightly guard requires the XLA local-wheel override fix.'
    )

  overrides_path = repo_utils.resolve_python_version_repo_label(
      state,
      overrides_label,
  )
  overrides_payload = json.loads(overrides_path.read_text(encoding='utf-8'))
  override_entries = {
      entry['package']: entry
      for entry in overrides_payload.get('overrides', [])
  }

  missing_packages = [
      package_name
      for package_name in expected_packages
      if package_name not in override_entries
  ]
  if missing_packages:
    raise RuntimeError(
        'Missing expected local wheel overrides for '
        f'CUDA {cuda_version}: {missing_packages}. '
        f'Checked {overrides_path}.'
    )

  selected_versions = {}
  for package_name in expected_packages:
    wheel_name = override_entries[package_name]['wheel']
    selected_versions[package_name] = wheel_version_from_filename(
        package_name,
        wheel_name,
    )

  unique_versions = sorted(set(selected_versions.values()))
  if len(unique_versions) != 1:
    raise RuntimeError(
        'Expected a single nightly binary version from local wheel overrides, '
        f'but found {selected_versions}.'
    )

  expected_binary_version = unique_versions[0]
  if '.dev' not in expected_binary_version:
    raise RuntimeError(
        'Expected the local wheel overrides to select nightly binary wheels, '
        f'but resolved {expected_binary_version}. Full selection: '
        f'{selected_versions}'
    )
  plugin_package = expected_packages[1]
  pjrt_package = expected_packages[2]

  print('Nightly local wheel override guard passed.')
  print(f'  CUDA version: {cuda_version}')
  print(f'  local_wheel_overrides.json: {overrides_path}')
  print(f'  expected binary version: {expected_binary_version}')
  for package_name in expected_packages:
    entry = override_entries[package_name]
    marker = entry.get('marker') or '<none>'
    print(
        f'  {package_name}: wheel={entry["wheel"]}, marker={marker}'
    )

  if args.github_env:
    write_github_env(
        Path(args.github_env),
        {
            'JAXCI_EXPECTED_BINARY_VERSION': expected_binary_version,
            'JAXCI_EXPECTED_BINARY_PACKAGES': ','.join(expected_packages),
            'JAXCI_EXPECTED_PLUGIN_DISTRIBUTION': plugin_package,
            'JAXCI_EXPECTED_PJRT_DISTRIBUTION': pjrt_package,
            'JAXCI_LOCAL_WHEEL_OVERRIDES_LABEL': overrides_label,
            'JAXCI_LOCAL_WHEEL_OVERRIDES_PATH': str(overrides_path),
        },
    )


if __name__ == '__main__':
  main()
