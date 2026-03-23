from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


TARGET_PACKAGES = (
    'jaxlib',
    'jax-cuda12-plugin',
    'jax-cuda12-pjrt',
    'jax-cuda13-plugin',
    'jax-cuda13-pjrt',
)
REQUIREMENT_RE = re.compile(
    r'^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^ ;]+)'
    r'(?:\s*;\s*(?P<marker>.+))?$'
)


@dataclass(frozen=True)
class RequirementEntry:
    requirement_line: str
    name: str
    version: str
    marker: str | None


@dataclass
class RequirementVariant:
    requirement_line: str
    target_platforms: list[str] = field(default_factory=list)


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


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


def find_python_version_repo_dir(output_base: Path) -> Path:
    external_dir = output_base / 'external'
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


def parse_requirement_line(requirement_line: str) -> RequirementEntry:
    match = REQUIREMENT_RE.match(requirement_line.strip())
    if match is None:
        raise ValueError(f'Unsupported requirement line: {requirement_line!r}')
    return RequirementEntry(
        requirement_line=requirement_line.strip(),
        name=match.group('name'),
        version=match.group('version'),
        marker=match.group('marker'),
    )


def extract_relevant_requirement_lines(merged_requirements: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in merged_requirements.read_text(encoding='utf-8').splitlines():
        stripped = raw_line.strip()
        if (
            not stripped
            or stripped.startswith('#')
            or stripped.startswith('--hash=')
        ):
            continue
        if stripped.endswith('\\'):
            stripped = stripped[:-1].rstrip()
        for package_name in TARGET_PACKAGES:
            if stripped.startswith(f'{package_name}=='):
                lines.append(stripped)
                break
    return lines


def overwrite_sort_key(requirement_line: str) -> tuple[int, str]:
    return (len(requirement_line.partition('==')[0]), requirement_line)


def marker_matches_platform(marker: str | None, platform: str) -> bool:
    if marker is None:
        return True
    normalized_marker = marker.strip()
    if normalized_marker == 'sys_platform == "linux"':
        return '_linux_' in platform
    raise ValueError(f'Unsupported marker: {marker!r}')


def build_requirement_variants(
    requirement_lines: list[str],
    platforms: tuple[str, str, str],
) -> dict[str, dict[str, RequirementVariant]]:
    grouped: dict[str, dict[str, RequirementVariant]] = {}

    for platform in platforms:
        requirements_dict: dict[str, RequirementEntry] = {}
        for requirement_line in sorted(requirement_lines, key=overwrite_sort_key):
            entry = parse_requirement_line(requirement_line)
            if entry.marker and not marker_matches_platform(entry.marker, platform):
                continue
            requirements_dict[entry.name] = entry

        for entry in requirements_dict.values():
            variants = grouped.setdefault(entry.name, {})
            variant = variants.setdefault(
                entry.requirement_line,
                RequirementVariant(requirement_line=entry.requirement_line),
            )
            variant.target_platforms.append(platform)

    return grouped


def select_requirement(
    variants: dict[str, RequirementVariant],
    platform_suffix: str | None,
) -> RequirementVariant | None:
    ordered = sorted(variants.values(), key=lambda item: item.requirement_line)
    candidates = [
        item
        for item in ordered
        if platform_suffix is None
        or any(target.endswith(platform_suffix) for target in item.target_platforms)
    ]
    if not candidates:
        return None
    return candidates[0]


def build_selection_report(
    requirement_lines: list[str],
    python_version: str,
) -> dict[str, object]:
    platform_tag = 'cp' + python_version.split('-', 1)[0].replace('.', '')
    platforms = (
        f'{platform_tag}_linux_x86_64',
        f'{platform_tag}_osx_x86_64',
        f'{platform_tag}_windows_x86_64',
    )
    variants_by_package = build_requirement_variants(requirement_lines, platforms)
    selected_none: dict[str, str] = {}
    selected_linux: dict[str, str] = {}
    variant_payload: dict[str, list[dict[str, object]]] = {}

    for package_name in TARGET_PACKAGES:
        variants = variants_by_package.get(package_name, {})
        variant_payload[package_name] = [
            {
                'requirement_line': item.requirement_line,
                'target_platforms': item.target_platforms,
            }
            for item in sorted(
                variants.values(),
                key=lambda item: item.requirement_line,
            )
        ]
        if not variants:
            continue

        none_selection = select_requirement(variants, platform_suffix=None)
        linux_selection = select_requirement(
            variants,
            platform_suffix='linux_x86_64',
        )
        if none_selection is not None:
            selected_none[package_name] = none_selection.requirement_line
        if linux_selection is not None:
            selected_linux[package_name] = linux_selection.requirement_line

    return {
        'requirement_lines': requirement_lines,
        'selected_with_platform_none': selected_none,
        'selected_with_linux_platform': selected_linux,
        'variants_by_package': variant_payload,
    }


def build_payload(phase: str) -> dict[str, object]:
    output_base_result = run(['bazel', 'info', 'output_base'])
    if output_base_result.returncode != 0:
        raise RuntimeError(
            'bazel info output_base failed:\n'
            f'stdout:\n{output_base_result.stdout}\n'
            f'stderr:\n{output_base_result.stderr}'
        )

    output_base = Path(output_base_result.stdout.strip())
    python_version_repo_dir = find_python_version_repo_dir(output_base)
    py_version_bzl = python_version_repo_dir / 'py_version.bzl'
    py_version_text = py_version_bzl.read_text(encoding='utf-8')
    py_version_values = parse_py_version_bzl(py_version_bzl)
    merged_requirements_name = Path(
        py_version_values['REQUIREMENTS_WITH_LOCAL_WHEELS'].split(':', 1)[1]
    ).name
    merged_requirements = python_version_repo_dir / merged_requirements_name
    merged_requirements_text = merged_requirements.read_text(encoding='utf-8')
    relevant_requirement_lines = extract_relevant_requirement_lines(
        merged_requirements
    )
    selection_report = build_selection_report(
        relevant_requirement_lines,
        py_version_values['HERMETIC_PYTHON_VERSION'],
    )

    return {
        'phase': phase,
        'dist_wheels': sorted(
            path.name for path in (repo_root() / 'dist').glob('*.whl')
        ),
        'output_base': str(output_base),
        'python_version_repo_dir': str(python_version_repo_dir),
        'py_version_values': py_version_values,
        'py_version_bzl_path': str(py_version_bzl),
        'py_version_bzl_sha256': sha256_text(py_version_text),
        'merged_requirements_path': str(merged_requirements),
        'merged_requirements_sha256': sha256_text(merged_requirements_text),
        'selection_report': selection_report,
    }


def compare_payloads(
    before: dict[str, object],
    after: dict[str, object],
) -> dict[str, object]:
    comparison = {
        'changed': False,
        'changes': {},
    }

    keys_to_compare = (
        'dist_wheels',
        'py_version_values',
        'py_version_bzl_sha256',
        'merged_requirements_sha256',
        'selection_report',
    )
    for key in keys_to_compare:
        if before.get(key) == after.get(key):
            continue
        comparison['changed'] = True
        comparison['changes'][key] = {
            'before': before.get(key),
            'after': after.get(key),
        }
    return comparison


def format_report(
    payload: dict[str, object],
    comparison: dict[str, object] | None,
) -> str:
    lines: list[str] = []
    lines.append(f"python_version_repo state dump ({payload['phase']})")
    lines.append('')
    lines.append('dist/ wheels:')
    for wheel_name in payload['dist_wheels']:
        lines.append(f'  {wheel_name}')
    if not payload['dist_wheels']:
        lines.append('  <none>')
    lines.append('')
    lines.append(f"output_base: {payload['output_base']}")
    lines.append(
        'python_version_repo_dir: '
        f"{payload['python_version_repo_dir']}"
    )
    lines.append('py_version.bzl values:')
    for key in sorted(payload['py_version_values']):
        lines.append(f"  {key}={payload['py_version_values'][key]}")
    lines.append(
        'py_version.bzl sha256: '
        f"{payload['py_version_bzl_sha256']}"
    )
    lines.append(
        'merged requirements: '
        f"{payload['merged_requirements_path']}"
    )
    lines.append(
        'merged requirements sha256: '
        f"{payload['merged_requirements_sha256']}"
    )
    lines.append('')
    lines.append('Relevant requirement lines:')
    for requirement_line in payload['selection_report']['requirement_lines']:
        lines.append(f'  {requirement_line}')
    if not payload['selection_report']['requirement_lines']:
        lines.append('  <none>')
    lines.append('')
    lines.append('Selected with platform=None:')
    for package_name in TARGET_PACKAGES:
        requirement_line = payload['selection_report'][
            'selected_with_platform_none'
        ].get(package_name)
        lines.append(f'  {package_name}: {requirement_line or "<none>"}')
    lines.append('')
    lines.append('Selected with Linux platform:')
    for package_name in TARGET_PACKAGES:
        requirement_line = payload['selection_report'][
            'selected_with_linux_platform'
        ].get(package_name)
        lines.append(f'  {package_name}: {requirement_line or "<none>"}')

    if comparison is not None:
        lines.append('')
        if comparison['changed']:
            lines.append('Comparison against previous snapshot: changed')
            for key, values in comparison['changes'].items():
                lines.append(f'  {key}:')
                lines.append(f"    before: {json.dumps(values['before'], sort_keys=True)}")
                lines.append(f"    after:  {json.dumps(values['after'], sort_keys=True)}")
        else:
            lines.append('Comparison against previous snapshot: no change')

    lines.append('')
    return '\n'.join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True)
    parser.add_argument('--snapshot-json')
    parser.add_argument('--compare-json')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args.phase)

    comparison = None
    if args.compare_json:
        compare_path = Path(args.compare_json)
        if compare_path.exists():
            previous_payload = json.loads(compare_path.read_text(encoding='utf-8'))
            comparison = compare_payloads(previous_payload, payload)
        else:
            comparison = {
                'changed': True,
                'changes': {
                    'compare_json': {
                        'before': '<missing>',
                        'after': args.compare_json,
                    }
                },
            }

    report = format_report(payload, comparison)
    sys.stdout.write(report)

    if args.snapshot_json:
        snapshot_path = Path(args.snapshot_json)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )


if __name__ == '__main__':
    main()
