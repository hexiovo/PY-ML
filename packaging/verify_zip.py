"""Verify all archived file hashes/CRCs and launch a relocated frozen build."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default=metadata.version('pyml-workbench'))
    args = parser.parse_args()
    if not re.fullmatch(r'\d+\.\d+\.\d+', args.version):
        parser.error('version must use major.minor.patch digits')
    archive = root / f'dist/PYML-Workbench-{args.version}-windows-x64.zip'
    evidence = root / f'delivery/exe-{args.version}'
    (evidence / 'logs').mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert len(names) == len(set(names))
        assert all(name.startswith('PYML-Workbench/') and '..' not in Path(name).parts
            and not Path(name).is_absolute() for name in names)
        manifest_name = 'PYML-Workbench/SHA256SUMS.txt'
        guide_present = 'PYML-Workbench/_internal/pyml_workbench/resources/overall-guide.md' in names
        if tuple(map(int, args.version.split('.'))) >= (0, 4, 2):
            assert guide_present, 'offline guide missing from current release'
        expected = {}
        for line in bundle.read(manifest_name).decode('utf-8').splitlines():
            checksum, name = line.split('  ', 1)
            expected['PYML-Workbench/' + name] = checksum
        assert set(names) == set(expected) | {manifest_name}
        for name, checksum in expected.items():
            with bundle.open(name) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == checksum, name
        print(f'PASS ZIP SHA-256/CRC for {len(expected)} files', flush=True)
        with tempfile.TemporaryDirectory(prefix='PYML-便携分发-') as temporary:
            target = Path(temporary).resolve()
            assert target.parent == Path(tempfile.gettempdir()).resolve() and target.name.startswith('PYML-便携分发-')
            bundle.extractall(target)
            distribution = target / 'PYML-Workbench'
            completed = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(root / 'packaging/smoke_gui.py'),
                str(distribution), str(evidence / 'relocated-gui'), '--startup-only'] + (['--guide'] if guide_present else []),
                cwd=target, capture_output=True, text=True, encoding='utf-8', timeout=90)
            (evidence / 'logs/relocated-gui.log').write_text(completed.stdout + completed.stderr, encoding='utf-8')
            assert completed.returncode == 0, completed.stderr
            env = dict(os.environ)
            for key in ('PYTHONHOME', 'PYTHONPATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
                env.pop(key, None)
            env['PATH'] = os.pathsep.join([str(Path(os.environ['SystemRoot']) / 'System32'), os.environ['SystemRoot']])
            env['PYML_LOG_ROOT'] = str(target / 'worker-logs')
            worker = subprocess.run([str(distribution / 'PYML-Worker.exe'), 'inspect', '--model', str(target / 'missing.joblib')],
                cwd=target, env=env, capture_output=True, text=True, encoding='utf-8', timeout=30,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            event = json.loads(worker.stdout)
            assert worker.returncode == 1 and event['type'] == 'error' and len(event['error_id']) == 32
            with archive.open('rb') as stream:
                archive_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            report = {'overall': 'PASS', 'version': args.version, 'zip_sha256': archive_digest,
                'checked_file_count': len(expected), 'CRC_checked_during_read': True,
                'relocated_gui': 'visible Chinese window, external Unicode path, system-only PATH',
                'relocated_guide': 'actual frozen F1 entry -> offline guide; resource matches public document' if guide_present else 'not available in this version',
                'relocated_worker': 'structured error event from actual copied EXE',
                'source_or_python_env_dependency': False}
            (evidence / 'zip-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS relocated GUI/worker', flush=True)


if __name__ == '__main__':
    main()
