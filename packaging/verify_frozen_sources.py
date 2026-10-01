"""Bind the embedded GUI/worker bytecode to current source with build Python."""
import argparse
import hashlib
import json
from pathlib import Path
import types

from PyInstaller.archive.readers import CArchiveReader


def normalized(code):
    return code.replace(
        co_filename='<verified-source>',
        co_consts=tuple(normalized(value) if isinstance(value, types.CodeType) else value
                        for value in code.co_consts),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('distribution', type=Path)
    parser.add_argument('evidence', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    modules = ('pyml_workbench', 'pyml_workbench.gui', 'pyml_workbench.guide')
    checked = []
    for executable in ('PYML-Workbench.exe', 'PYML-Worker.exe'):
        archive = CArchiveReader(str(args.distribution / executable))
        pyz_name = next(name for name in archive.toc if name.endswith('.pyz'))
        embedded = archive.open_embedded_archive(pyz_name)
        for module in modules:
            source = root / ('src/pyml_workbench/__init__.py' if module == 'pyml_workbench'
                             else 'src/' + module.replace('.', '/') + '.py')
            current = compile(source.read_bytes(), str(source), 'exec', dont_inherit=True, optimize=0)
            actual = embedded.extract(module)
            assert normalized(actual) == normalized(current), (executable, module)
            checked.append({'executable': executable, 'module': module,
                            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    resource = args.distribution / '_internal/pyml_workbench/resources/overall-guide.md'
    assert resource.read_bytes() == (root / 'docs/overall-guide.md').read_bytes()
    report = {'overall': 'PASS', 'embedded_modules_match_current_source': checked,
              'guide_resource_sha256': hashlib.sha256(resource.read_bytes()).hexdigest()}
    args.evidence.mkdir(parents=True, exist_ok=True)
    (args.evidence / 'frozen-source-provenance.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('PASS embedded GUI/worker code matches current source (6 module checks)', flush=True)


if __name__ == '__main__':
    main()
