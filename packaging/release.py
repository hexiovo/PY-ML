"""Add local guides/notices and produce a complete, verifiable folder ZIP."""
from __future__ import annotations

import argparse
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
    config = json.loads((root / 'packaging/profiles.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=sorted(config['profiles']),
                        default=config['default_profile'])
    parser.add_argument('--release-dir', type=Path,
                        help='profile output folder; defaults to the original dist location')
    parser.add_argument('--archive-path', type=Path,
                        help='ZIP output path; defaults to the versioned dist filename')
    args = parser.parse_args()
    profile_name = args.profile
    profile = config['profiles'][profile_name]
    folder = (args.release_dir.resolve() if args.release_dir
              else root / 'dist' / f'PYML-Workbench-{profile_name}')
    archive_override = args.archive_path.resolve() if args.archive_path else None
    dist_root = (root / 'dist').resolve()
    if not folder.is_relative_to(dist_root):
        raise SystemExit('--release-dir must stay under the project dist directory')
    if archive_override is not None and not archive_override.is_relative_to(dist_root):
        raise SystemExit('--archive-path must stay under the project dist directory')
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
    guide_pdf = root / f'output/pdf/PYML-Workbench-{version}-图文指南.pdf'
    assert guide_pdf.is_file(), f'the README-linked PDF guide is missing: {guide_pdf}'
    guide_destination = folder / 'output/pdf' / guide_pdf.name
    guide_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(guide_pdf, guide_destination)
    (folder / '启动说明.txt').write_text(
        f'PY-ML 中文机器学习工作台 {version}\n\n'
        '双击 PYML-Workbench.exe 启动，无需安装 Python。\n'
        '请保留整个文件夹，包括 PYML-Worker.exe 与 _internal，不能只复制主 EXE。\n'
        '启动时会先显示“正在开启程序中…”；初始化完成后自动进入工作台。\n'
        '操作指南：主界面底部“操作指南”、帮助菜单或 F1；支持离线目录和搜索。\n'
        '文档：docs/overall-guide.md；日志和诊断：文件菜单 → 导出诊断包。\n'
        + ('标准版包含 71 个 scikit-learn 模型，不含 HMM 或深度学习扩展。'
         if profile_name == 'standard' else
         '完整扩展版中 N01 默认使用 CPU；仅在 CUDA PyTorch 可用时可选择 GPU，其他模型使用 CPU。')
        + '第三方来源及许可证见 docs/third-party.md 与 LICENSES。\n', encoding='utf-8-sig')
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
    from PyInstaller.utils.hooks.tcl_tk import tcltk_info
    for component, directory in (('Tcl', tcltk_info.tcl_data_dir), ('Tk', tcltk_info.tk_data_dir)):
        source = Path(directory) / 'license.terms'
        if component == 'Tcl' and not source.is_file():
            # Standalone Windows Python includes the Tcl terms in LICENSE.txt.
            source = python_license
        if not source.is_file():
            raise SystemExit(f'Missing {component} startup-splash license: {source}')
        target = notices / component / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    packages = ['pyml-workbench', *profile['runtime_distributions'], *config['build_distributions']]
    build_info = {'version': version, 'python': sys.version.split()[0], 'architecture': 'Windows x64',
        'packages': {p: metadata.version(p) for p in packages},
        'profile': profile_name, 'profile_title': profile['title'],
        'profile_extras': profile['extras'], 'licenses_copied': len(copied),
        'kind': 'PyInstaller onedir, shared GUI and worker dependencies',
        'startup_splash': {'enabled': True, 'scope': 'GUI only',
                           'renderer': 'DPI-scaled canvas text, shapes and indeterminate loading bar',
                           'dpi_awareness': 'PerMonitorV2',
                           'tcl': '.'.join(map(str, tcltk_info.tcl_version)),
                           'tk': '.'.join(map(str, tcltk_info.tk_version))}}
    (folder / 'build-info.json').write_text(json.dumps(build_info, ensure_ascii=False, indent=2), encoding='utf-8')
    runtime_files = {'pyml-workbench-operations.jsonl', '.pyml-workbench-operations.lock'}
    files = sorted(p for p in folder.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt'
                   and not (p.parent == folder and p.name in runtime_files))
    manifest = ''.join(f'{digest(p)}  {p.relative_to(folder).as_posix()}\n' for p in files)
    (folder / 'SHA256SUMS.txt').write_text(manifest, encoding='utf-8')
    archive = archive_override or root / f'dist/PYML-Workbench-{version}-windows-x64-{profile_name}.zip'
    if archive.exists():
        raise SystemExit(f'Release archive already exists; preserve it and choose a clean output: {archive}')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted([*files, folder / 'SHA256SUMS.txt']):
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
    destination = root / f'delivery/exe-{version}/{profile_name}-release-result.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
