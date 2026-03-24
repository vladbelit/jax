from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PythonVersionRepoState:
  output_base: Path
  python_version_repo_dir: Path
  py_version_bzl: Path
  py_version_values: dict[str, str]


def repo_root() -> Path:
  return Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      cmd,
      cwd=repo_root(),
      text=True,
      capture_output=True,
      check=False,
  )


def parse_py_version_bzl(py_version_bzl: Path) -> dict[str, str]:
  values: dict[str, str] = {}
  for line in py_version_bzl.read_text(encoding='utf-8').splitlines():
    if ' = "' not in line:
      continue
    key, _, remainder = line.partition(' = "')
    if not remainder.endswith('"'):
      continue
    values[key.strip()] = remainder[:-1]
  return values


def output_base() -> Path:
  output_base_result = run(['bazel', 'info', 'output_base'])
  if output_base_result.returncode != 0:
    raise RuntimeError(
        'bazel info output_base failed:\n'
        f'stdout:\n{output_base_result.stdout}\n'
        f'stderr:\n{output_base_result.stderr}'
    )
  return Path(output_base_result.stdout.strip())


def find_python_version_repo_dir(output_base_path: Path) -> Path:
  external_dir = output_base_path / 'external'
  matches = [
      path.parent
      for path in external_dir.rglob('py_version.bzl')
      if 'python_version_repo' in str(path.parent)
  ]
  if not matches:
    raise FileNotFoundError(
        f'Could not locate python_version_repo under {external_dir}'
    )
  matches.sort(key=lambda path: len(str(path)))
  return matches[0]


def load_python_version_repo_state() -> PythonVersionRepoState:
  output_base_path = output_base()
  python_version_repo_dir = find_python_version_repo_dir(output_base_path)
  py_version_bzl = python_version_repo_dir / 'py_version.bzl'
  py_version_values = parse_py_version_bzl(py_version_bzl)
  return PythonVersionRepoState(
      output_base=output_base_path,
      python_version_repo_dir=python_version_repo_dir,
      py_version_bzl=py_version_bzl,
      py_version_values=py_version_values,
  )


def resolve_python_version_repo_label(
    state: PythonVersionRepoState,
    label: str,
) -> Path:
  prefix = '@python_version_repo//:'
  if not label.startswith(prefix):
    raise ValueError(f'Unsupported python_version_repo label: {label}')
  return state.python_version_repo_dir / label.removeprefix(prefix)
