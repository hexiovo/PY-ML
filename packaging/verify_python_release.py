"""Verify current source and study skill in the Python release artifacts."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import tarfile
import zipfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    version=metadata.version('pyml-workbench')
    wheel=root/f'dist/pyml_workbench-{version}-py3-none-any.whl'
    sdist=root/f'dist/pyml_workbench-{version}.tar.gz'
    checked=[]
    with zipfile.ZipFile(wheel) as bundle:
        assert bundle.read('pyml_workbench/resources/overall-guide.md')==(root/'docs/overall-guide.md').read_bytes()
        for source in (root/'src/pyml_workbench').rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and source.suffix in {'.py','.json'}:
                name=source.relative_to(root/'src').as_posix()
                assert bundle.read(name)==source.read_bytes(),name
                checked.append(name)
        assert not any(name.startswith(('tests/','delivery/','output/','build/')) for name in bundle.namelist())
    with tarfile.open(sdist) as bundle:
        names=bundle.getnames()
        assert not any(any('/'+part+'/' in name for part in ['delivery','output','build','.venv','__pycache__']) for name in names)
        assert not any(name.endswith(('/packaging/plan.txt', '/packaging/guide-plan.txt')) for name in names)
        for source in [root/'pyproject.toml',root/'README.md',root/'version.md',root/'docs/overall-guide.md',*(p for p in (root/'skills/pyml-study-config').rglob('*') if p.is_file() and '__pycache__' not in p.parts)]:
            name=f'pyml_workbench-{version}/'+source.relative_to(root).as_posix()
            with bundle.extractfile(name) as stream: assert stream.read()==source.read_bytes(),name
    report={'overall':'PASS','version':version,'wheel_sources_checked':checked,'offline_guide_matches':True,'sdist_skill_and_documents_match':True,'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [wheel,sdist]}}
    if args.evidence:
        args.evidence.parent.mkdir(parents=True,exist_ok=True)
        args.evidence.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
