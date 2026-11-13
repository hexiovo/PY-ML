"""Convert a PYEDA result bundle and audit the resulting feature table."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import queue
import sys
import threading
from uuid import uuid4


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_demo_bundle(root):
    """Generate a small synthetic result bundle without any labels or raw signals."""
    (root / "intake").mkdir(parents=True)
    (root / "features").mkdir()
    features = ["tonic_mean", "phasic_rms", "relative_psd"]
    write_json(root / "intake/analysis_settings.json", {"features": features, "qualification": "synthetic"})
    write_json(root / "summary.json", {"qualification": "synthetic", "rows": 12})
    (root / "intake/feature_definitions.csv").write_text(
        "feature,unit,definition\ntonic_mean,a.u.,Synthetic tonic mean\n"
        "phasic_rms,a.u.,Synthetic phasic RMS\nrelative_psd,a.u.,Synthetic missing feature\n", encoding="utf-8"
    )
    table = root / "features/participant_task_role_features.csv"
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject_id", "task", "group_id", "group_type", "participant_label", "window", "role", *features])
        for person in range(1, 7):
            for window in range(1, 3):
                writer.writerow([f"S{person:02}", "demo", f"G{(person-1)//2 + 1}", "demo",
                                 f"P{person:02}", f"W{window}", "self", person + window / 10,
                                 "inf" if (person, window) == (1, 1) else window / 10, ""])
    artifacts = [{"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                 for path in sorted(root.rglob("*")) if path.is_file()]
    write_json(root / "manifest.json", {"artifacts": artifacts})


def audit_conversion(conversion, prepared):
    dictionary = json.loads((conversion / "feature_dictionary.json").read_text(encoding="utf-8"))
    features = [item["output_column"] for item in dictionary["features"]]
    identities = [f"meta__{name}" for name in dictionary["identity_columns"]]
    observed = dict.fromkeys(features, 0)
    seen = set()
    with (conversion / "dataset.csv").open(encoding="utf-8-sig", newline="") as data_stream, \
         (conversion / "valid_mask.csv").open(encoding="utf-8-sig", newline="") as mask_stream, \
         prepared.open("w", encoding="utf-8-sig", newline="") as output_stream:
        data = csv.DictReader(data_stream)
        mask = csv.DictReader(mask_stream)
        header = ["sample_id", *identities]
        if "meta__status" in (data.fieldnames or []):
            header.append("meta__status")
        if dictionary["targets"] or data.fieldnames != [*header, *features] or mask.fieldnames != ["sample_id", *features]:
            raise ValueError("特征、身份列或掩码表头与预期不一致")
        writer = csv.DictWriter(output_stream, fieldnames=[*header, *features])
        writer.writeheader()
        for row, flags in zip(data, mask, strict=True):
            sample = row["sample_id"]
            if not sample or sample in seen or sample != flags["sample_id"]:
                raise ValueError("数据表与掩码的样本键不唯一或行序不一致")
            seen.add(sample)
            for feature in features:
                flag = flags[feature]
                if flag not in {"0", "1"} or (flag == "1" and not math.isfinite(float(row[feature]))):
                    raise ValueError(f"掩码与特征值不一致：{feature}")
                observed[feature] += flag == "1"
                if flag == "0":
                    row[feature] = ""
            writer.writerow(row)
    return {
        "rows": len(seen), "features": len(features), "finite_cells": sum(observed.values()),
        "missing_cells_by_feature": {name: len(seen) - count for name, count in observed.items()},
        "all_missing_features": [name for name, count in observed.items() if count == 0],
        "missing_policy": "keep columns and replace flagged cells with blanks; no imputation",
    }


def run(plugin, source, output, profile, progress, cancel_event=None):
    plugin = plugin.expanduser().resolve(strict=True)
    manifest = json.loads((plugin / "pyml-plugin.json").read_text(encoding="utf-8-sig"))
    if manifest.get("source_project") != "PYEDA" or not (plugin / "pyeda_converter/core.py").is_file():
        raise ValueError("请选择转换仓库中的 pyeda 源码目录")
    sys.path.insert(0, str(plugin))
    from pyeda_converter import ConversionOptions, convert_bundle

    if source is not None:
        source = source.expanduser().resolve(strict=True)
    elif profile != "participant_task_role":
        raise ValueError("合成示例提供 participant_task_role 来源")
    output.mkdir(parents=True, exist_ok=False)
    try:
        synthetic = source is None
        if synthetic:
            source = output / "synthetic-pyeda"
            write_demo_bundle(source)
        progress("输入准备", 1, 1, str(source))
        conversion = output / "conversion"
        convert_bundle(source, conversion, ConversionOptions(source=profile, include_outcomes=False),
                       progress_callback=progress, cancel_event=cancel_event)
        progress("核对转换表和有效性掩码", 0, 1, "features.csv")
        qc = audit_conversion(conversion, output / "features.csv")
        write_json(output / "quality.json", {"status": "completed", "synthetic": synthetic, **qc})
        progress("转换与质控完成", 1, 1, str(output))
    except Exception as exc:
        state = "cancelled" if cancel_event is not None and cancel_event.is_set() else "failed"
        write_json(output / "quality.json", {"status": state, "error": str(exc)})
        raise
    return qc


def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, ttk

    window = tk.Tk()
    window.title("PYEDA 特征转换示例")
    window.minsize(640, 260)
    frame = ttk.Frame(window, padding=14)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)
    plugin, source, output = (tk.StringVar() for _ in range(3))
    output.set(str(Path.cwd()))

    def choose(variable):
        directory = filedialog.askdirectory()
        if directory:
            variable.set(directory)

    for index, (label, value) in enumerate([
        ("PYEDA 插件源码目录", plugin), ("结果包目录（空白运行合成示例）", source), ("输出根目录", output),
    ]):
        ttk.Label(frame, text=label).grid(row=index, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=value).grid(row=index, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="浏览", command=lambda value=value: choose(value)).grid(row=index, column=2)
    stage = tk.StringVar(value="请选择插件源码目录")
    bar = ttk.Progressbar(frame, maximum=100)
    bar.grid(row=4, columnspan=3, sticky="ew", pady=8)
    ttk.Label(frame, textvariable=stage, wraplength=620).grid(row=5, columnspan=3, sticky="w")
    events = queue.Queue()
    cancel = threading.Event()

    def poll():
        while not events.empty():
            item = events.get_nowait()
            if isinstance(item, str):
                stage.set(item)
                start.configure(state="normal")
                return
            name, current, total, detail = item
            stage.set(f"{name}  {current}/{total}  {detail}")
            bar.configure(value=100 * current / max(total, 1))
        window.after(100, poll)

    def begin():
        if not plugin.get().strip() or not output.get().strip():
            stage.set("请选择插件源码目录和输出根目录")
            return
        plugin_path = Path(plugin.get().strip())
        source_path = Path(source.get().strip()) if source.get().strip() else None
        run_path = Path(output.get().strip()) / f"pyeda-example-{uuid4().hex[:8]}"
        start.configure(state="disabled")
        cancel.clear()
        bar.configure(value=0)

        def worker():
            try:
                run(plugin_path, source_path, run_path, "participant_task_role", lambda *info: events.put(info), cancel)
            except Exception as exc:
                events.put(f"{'已取消' if cancel.is_set() else '运行失败'}：{exc}")
            else:
                events.put(f"已完成：{run_path}")

        threading.Thread(target=worker, daemon=True).start()
        window.after(100, poll)

    start = ttk.Button(frame, text="开始转换", command=begin)
    start.grid(row=3, column=0, sticky="w", pady=6)
    ttk.Button(frame, text="取消", command=cancel.set).grid(row=3, column=1, sticky="w")
    window.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", required=True, type=Path, help="PYEDA 插件源码目录")
    parser.add_argument("--input", type=Path, help="PYEDA 结果目录，省略则生成合成示例")
    parser.add_argument("--output", required=True, type=Path, help="新的输出目录")
    parser.add_argument("--source", default="participant_task_role",
                        choices=["participant_task_role", "participant_exposure", "records"])
    args = parser.parse_args()
    run(args.plugin, args.input, args.output, args.source,
        lambda stage, current, total, detail: print(f"{stage} {current}/{total} {detail}", flush=True))


if __name__ == "__main__":
    main()
