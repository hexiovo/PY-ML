"""Build and package one of the supported Windows release profiles."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'packaging/profiles.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=sorted(config['profiles']),
                        default=config['default_profile'])
    parser.add_argument('--dry-run', action='store_true',
                        help='print the selected profile and build commands without running them')
    parser.add_argument('--work-dir', type=Path,
                        help='absolute, new directory for PyInstaller intermediates (outside the release output)')
    parser.add_argument('--release-dir', type=Path,
                        help='absolute, new output folder for this profile (folder name must match the spec)')
    parser.add_argument('--archive-path', type=Path,
                        help='absolute, new ZIP path (defaults to the versioned dist filename)')
    args = parser.parse_args()

    profile_name = args.profile
    profile = config['profiles'][profile_name]
    if args.release_dir is None:
        dist_dir = root / 'dist' / f'PYML-Workbench-{profile_name}'
    else:
        if not args.release_dir.is_absolute():
            raise SystemExit('--release-dir must be an absolute path.')
        dist_dir = args.release_dir.resolve()
    if dist_dir.name != f'PYML-Workbench-{profile_name}':
        raise SystemExit(f'--release-dir must end with PYML-Workbench-{profile_name}.')
    if args.archive_path is not None and not args.archive_path.is_absolute():
        raise SystemExit('--archive-path must be an absolute path.')
    archive_override = args.archive_path.resolve() if args.archive_path else None
    if archive_override is not None and not archive_override.is_relative_to((root / 'dist').resolve()):
        raise SystemExit('--archive-path must stay under the project dist directory.')
    if not dist_dir.resolve().is_relative_to((root / 'dist').resolve()):
        raise SystemExit('--release-dir must stay under the project dist directory.')
    archive_prefix = root / f'dist/PYML-Workbench-{{version}}-windows-x64-{profile_name}.zip'
    if args.work_dir is None:
        work_dir = root / 'build' / f'pyinstaller-{profile_name}'
    else:
        if not args.work_dir.is_absolute():
            raise SystemExit('--work-dir must be an absolute path.')
        work_dir = args.work_dir.resolve()
    build_command = [
        sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
        '--distpath', str(dist_dir.parent), '--workpath', str(work_dir),
        str(root / 'packaging/workbench.spec'),
    ]
    package_command = [sys.executable, str(root / 'packaging/release.py'),
                       '--profile', profile_name, '--release-dir', str(dist_dir)]
    if archive_override is not None:
        package_command += ['--archive-path', str(archive_override)]
    plan = {
        'profile': profile_name,
        'title': profile['title'],
        'summary': profile['summary'],
        'uv_extras': profile['extras'],
        'expected_output_folder': str(dist_dir),
        'expected_archive_pattern': str(archive_override or archive_prefix),
        'commands': [build_command, package_command],
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return 0

    if sys.version_info[:3] != (3, 12, 14):
        raise SystemExit('Build with the isolated Python 3.12.14 runtime.')
    if dist_dir.exists():
        raise SystemExit(f'Release folder already exists; preserve it and choose a clean output: {dist_dir}')
    version = metadata.version('pyml-workbench')
    archive = archive_override or root / f'dist/PYML-Workbench-{version}-windows-x64-{profile_name}.zip'
    if archive.exists():
        raise SystemExit(f'Release archive already exists; preserve it and choose a clean output: {archive}')
    if work_dir.exists():
        raise SystemExit(f'PyInstaller work directory already exists; preserve it and choose a clean output: {work_dir}')
    guide_pdf = root / f'output/pdf/PYML-Workbench-{version}-图文指南.pdf'
    if not guide_pdf.is_file():
        raise SystemExit(f'The README-linked PDF guide is missing: {guide_pdf}')
    if not (root / 'packaging/workbench.spec').is_file():
        raise SystemExit('Missing packaging/workbench.spec')

    environment = os.environ.copy()
    environment['PYML_PACKAGE_PROFILE'] = profile_name
    subprocess.run(build_command, cwd=root, env=environment, check=True)
    subprocess.run(package_command, cwd=root, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
