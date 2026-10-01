"""Check guide/source provenance in wheel/sdist and preserve old checksums."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import tarfile
import zipfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    version = metadata.version('pyml-workbench')
    document = (root / 'docs/overall-guide.md').read_bytes()
    wheel = root / f'dist/pyml_workbench-{version}-py3-none-any.whl'
    sdist = root / f'dist/pyml_workbench-{version}.tar.gz'
    with zipfile.ZipFile(wheel) as bundle:
        assert bundle.read('pyml_workbench/resources/overall-guide.md') == document
        for name in ('guide.py', 'gui.py', '__init__.py'):
            assert bundle.read('pyml_workbench/' + name) == (root / 'src/pyml_workbench' / name).read_bytes()
    with tarfile.open(sdist) as bundle:
        assert not any('/delivery/' in item.name for item in bundle.getmembers())
        for name in ('docs/overall-guide.md', 'version.md', 'README.md', 'packaging/verify_zip.py'):
            with bundle.extractfile(f'pyml_workbench-{version}/' + name) as stream:
                assert stream.read() == (root / name).read_bytes()
    checksum = root / 'dist/SHA256SUMS.txt'
    previous = checksum.read_text(encoding='utf-8').splitlines()
    own = {wheel.name, sdist.name}
    entries = [line for line in previous if line.split('  ', 1)[-1] not in own]
    hashes = {path.name: digest(path) for path in (wheel, sdist)}
    entries.extend(f'{value}  {name}' for name, value in hashes.items())
    checksum.write_text('\n'.join(entries) + '\n', encoding='utf-8')
    old_artifacts = {}
    for line in previous:
        value, name = line.split('  ', 1)
        if name not in own and '/' not in name:
            assert digest(root / 'dist' / name) == value, name
            old_artifacts[name] = value
    report = {'overall': 'PASS', 'version': version,
              'wheel_resource_and_changed_sources_match': True,
              'sdist_documents_match_current_source': True,
              'artifacts': hashes, 'existing_artifacts_preserved': old_artifacts}
    destination = root / f'delivery/exe-{version}/python-package-provenance.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS Python releases, current docs, and previous artifact hashes', flush=True)


if __name__ == '__main__':
    main()
