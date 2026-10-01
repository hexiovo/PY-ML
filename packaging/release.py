"""Add local guides/notices and produce a complete, verifiable folder ZIP."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys
import zipfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'dist/PYML-Workbench'
    assert (folder / 'PYML-Workbench.exe').is_file() and (folder / 'PYML-Worker.exe').is_file()
    version = metadata.version('pyml-workbench')
    # Guides are external frozen data, so refresh from the same canonical file
    # as public docs before calculating checksums and creating the archive.
    guide_resource = folder / '_internal/pyml_workbench/resources/overall-guide.md'
    assert guide_resource.is_file(), 'the build must include the offline guide resource'
    shutil.copy2(root / 'docs/overall-guide.md', guide_resource)
    for document in ('README.md', 'version.md'):
        shutil.copy2(root / document, folder / document)
    shutil.copytree(root / 'docs', folder / 'docs', dirs_exist_ok=True)
    (folder / '启动说明.txt').write_text(
        f'PY-ML 中文机器学习工作台 {version}\n\n'
        '双击 PYML-Workbench.exe 启动，无需安装 Python。\n'
        '请保留整个文件夹，包括 PYML-Worker.exe 与 _internal，不能只复制主 EXE。\n'
        '整体指南：主界面顶部“整体指南”、帮助菜单或 F1；支持离线目录和搜索。\n'
        '文档：docs/overall-guide.md；日志和诊断：文件菜单 → 导出诊断包。\n'
        '深度学习使用 CPU。第三方来源及许可证见 docs/third-party.md 与 LICENSES。\n', encoding='utf-8-sig')
    notices = folder / 'LICENSES'
    copied = []
    for distribution in metadata.distributions():
        name = distribution.metadata['Name']
        for file in distribution.files or ():
            if Path(str(file)).is_absolute() or '..' in Path(str(file)).parts:
                continue
            if any(marker in file.name.casefold() for marker in ('license', 'copying', 'notice', 'copyright')):
                source = Path(distribution.locate_file(file))
                if source.is_file() and source.suffix.lower() in {'', '.txt', '.md', '.rst', '.apache', '.bsd'}:
                    target = notices / name / Path(str(file))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    copied.append(target.relative_to(folder).as_posix())
    python_license = Path(sys.base_prefix) / 'LICENSE.txt'
    if python_license.is_file():
        (notices / 'Python').mkdir(parents=True, exist_ok=True)
        shutil.copy2(python_license, notices / 'Python/LICENSE.txt')
    packages = ['pyml-workbench', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'PySide6',
        'matplotlib', 'hmmlearn', 'torch', 'skorch', 'optuna', 'pymoo', 'PyInstaller', 'pyinstaller-hooks-contrib']
    build_info = {'version': version, 'python': sys.version.split()[0], 'architecture': 'Windows x64',
        'packages': {p: metadata.version(p) for p in packages},
        'licenses_copied': len(copied), 'kind': 'PyInstaller onedir, shared GUI and worker dependencies'}
    (folder / 'build-info.json').write_text(json.dumps(build_info, ensure_ascii=False, indent=2), encoding='utf-8')
    files = sorted(p for p in folder.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt')
    manifest = ''.join(f'{digest(p)}  {p.relative_to(folder).as_posix()}\n' for p in files)
    (folder / 'SHA256SUMS.txt').write_text(manifest, encoding='utf-8')
    archive = root / f'dist/PYML-Workbench-{version}-windows-x64.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                output.write(path, path.relative_to(folder.parent).as_posix())
    # Preserve previous artifact lines, replacing only this version's owned entries.
    checksum = root / 'dist/SHA256SUMS.txt'
    prior = checksum.read_text(encoding='utf-8') if checksum.exists() else ''
    entries = [folder / 'PYML-Workbench.exe', folder / 'PYML-Worker.exe', archive]
    own_names = {p.relative_to(root / 'dist').as_posix() for p in entries}
    lines = [line for line in prior.splitlines() if line.split('  ', 1)[-1] not in own_names]
    lines += [f'{digest(p)}  {p.relative_to(root / "dist").as_posix()}' for p in entries]
    checksum.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    report = {'version': version, 'folder_bytes': sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()),
        'file_count': len(files) + 1, 'zip': str(archive), 'zip_bytes': archive.stat().st_size,
        'sha256': {p.relative_to(root).as_posix(): digest(p) for p in entries}}
    destination = root / f'delivery/exe-{version}/release-result.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
