"""Build both study skill artifacts directly from the current source tree."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

from release_metadata import source_version


def build_skill(root: Path, output: Path) -> tuple[Path, Path]:
    source = root / "skills/pyml-study-config"
    output.mkdir(parents=True, exist_ok=True)
    versioned = f"pyml-study-config-{source_version(root)}"
    archive = output / f"{versioned}.zip"
    portable = output / f"{versioned}-portable.md"
    paths = sorted(path for path in source.rglob("*") if path.is_file()
                   and "__pycache__" not in path.parts and path.suffix != ".pyc")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path, path.relative_to(source.parent).as_posix())
    document = ["# PYML 实验配置技能 · 单文件便携版", "", "运行脚本与模板文件见完整技能 ZIP。", ""]
    references = [path for path in paths if path.is_relative_to(source / "references")]
    for path in [source / "SKILL.md", *references]:
        document.extend(["---", "", f"## {path.relative_to(source).as_posix()}", "",
                         path.read_text(encoding="utf-8-sig").rstrip(), ""])
    portable.write_text("\n".join(document), encoding="utf-8")
    return archive, portable


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "dist")
    for path in build_skill(root, parser.parse_args().output):
        print(path)
