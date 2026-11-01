"""One-click conversion for this linked HyperEEG result bundle; no training."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid
import shutil
from progress import Progress, Cancelled

HERE = Path(__file__).resolve().parent
ACTIVE_PROGRESS = None
ACTIVE_PROCESS = None
OUTPUT_ROOT = HERE.parent
RUN_LOCK = None

def lock_output(root):
    import msvcrt
    handle = (root / '.conversion.lock').open('a+b')
    handle.seek(0)
    if not handle.read(1):
        handle.write(b'0'); handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        raise RuntimeError('此输出目录已有转换正在运行，请使用原窗口，避免重复占用内存。') from None
    return handle

def latest(payload):
    save(OUTPUT_ROOT / 'latest-run.json', payload)
    save(HERE / 'latest-run.json', payload)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()

def main():
    global ACTIVE_PROGRESS, ACTIVE_PROCESS, OUTPUT_ROOT, RUN_LOCK
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    cfg = read(HERE / 'settings.json')
    OUTPUT_ROOT = Path(cfg['output_root']).resolve()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_LOCK = lock_output(OUTPUT_ROOT)
    source = Path(cfg['input'])
    manifest = read(source / '结果清单.json')
    if manifest.get('storage_mode') != 'linked_native_results':
        raise ValueError('输入不是已核实的 HyperEEG 引用式结果包。')
    retained = Path(manifest['retained_run'])
    if not retained.is_dir(): raise FileNotFoundError(f'缺少完整原始运行目录：{retained}')
    plugin = Path(cfg['plugin'])
    native_manifest = read(plugin / 'pyml-plugin.json')
    if native_manifest['id'] != 'data-conversion': raise ValueError('插件身份不匹配')
    sys.path.insert(0, str(plugin))
    from hypereeg_converter.core import (Options, Table, read_csv, schema,
        convert_tables, merge_converted, write_package, numeric, Skipped, ConversionError, RECORD_METRICS)
    import pandas as pd
    from streaming import Accumulator
    run = OUTPUT_ROOT / 'runs' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    progress = ACTIVE_PROGRESS = Progress(run)
    progress.update(1, '检查输入与参数', 0, 1, current=source, force=True)
    print(f'输出目录：{run}', flush=True)
    latest({'run': str(run), 'status': 'running'})
    options = Options(**read(HERE / 'hypereeg-options.json'))
    save(run / 'hypereeg-options.json', asdict(options))
    input_hashes = {str(p): sha(p) for p in [source / '特征结果索引.mat', source / '结果清单.json']}
    request = {'input': str(source), 'output': str(run), 'out_root': str(retained.parent.parent),
               'bridge': str(plugin / 'hypereeg_converter' / 'matlab'), 'metric_names': RECORD_METRICS,
               'metric_prefixes': ['quantile_', 'absolute_power_', 'relative_power_', 'log_power_db_',
                                   'log_power_log10_', 'spectral_edge_', 'rqa_'],
               'memory_helper_root': cfg['memory_helper_root'],
               'max_artifact_load_gib': cfg['max_artifact_load_gib'], 'reserve_free_gib': cfg['reserve_free_gib']}
    save(run / 'export-request.json', request)
    quote = lambda p: str(p).replace("'", "''")
    expression = f"addpath('{quote(HERE)}'); export_linked('{quote(run / 'export-request.json')}')"
    progress.update(1, '检查输入与参数', 1, 1, current=source, force=True)
    progress.update(2, '启动 MATLAB，等待原生读取进度', current=run / 'matlab.log', force=True)
    print('正在用 MATLAB 读取原生表，保留原始数据；请等待。', flush=True)
    with (run / 'matlab.log').open('wb') as log:
        proc = subprocess.Popen([cfg['matlab'], '-wait', '-batch', expression], stdout=log, stderr=log,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        ACTIVE_PROCESS = proc
        while proc.poll() is None:
            try:
                native_progress = read(run / 'native_progress.json')
            except (FileNotFoundError, json.JSONDecodeError, PermissionError):
                pass
            else:
                progress.update(2, f"MATLAB 原生导出（任务 {native_progress['task']}/{native_progress['tasks']}）",
                                native_progress['overall_completed'], native_progress['overall_total'],
                                current=native_progress.get('current', ''),
                                **native_progress.get('counts', {}))
            progress.stop.wait(0.5)
    if proc.returncode: raise RuntimeError(f'MATLAB 读取失败，查看 {run / "matlab.log"}')
    ACTIVE_PROCESS = None
    records = [json.loads(line) for line in (run / 'native_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    audit, groups, examples, source_hashes = [], {}, {}, {}
    # Check committed artifact hashes before conversion; reject changed or missing sources.
    record_paths = {r['source'] for r in records if r['status'] != 'outside_scalar_profile'}
    progress.update(3, '核对源特征哈希', 0, len(record_paths), force=True)
    for index in sorted(run.glob('task_*/artifact_index.csv')):
        frame = pd.read_csv(index, dtype=str, keep_default_na=False)
        for row in frame.to_dict('records'):
            if row['path'] in record_paths:
                actual = sha(row['path'])
                if row['file_sha256'] and actual != row['file_sha256']:
                    raise ValueError(f'来源哈希与原生索引不一致：{row["path"]}')
                source_hashes[row['path']] = actual
                progress.update(3, '核对源特征哈希', len(source_hashes), len(record_paths), current=row['path'])
    progress.update(3, '核对源特征哈希', len(source_hashes), len(record_paths), force=True)
    for record_number, r in enumerate(records, 1):
        progress.update(4, '转换特征表', record_number - 1, len(records), current=r['source'],
                        已转换表=sum(x['status'] == 'converted' for x in audit),
                        未转换表=sum(x['status'] != 'converted' for x in audit))
        if r['status'] == 'failed': raise RuntimeError(f'原生载荷读取失败：{r}')
        quality = r.get('quality', {}).get('status', '')
        for tr in r.get('tables', []):
            record = {'task': r['task'], 'source': r['source'], 'table': tr['path'], 'status': tr['status']}
            if tr['status'] != 'exported':
                record['reason'] = tr.get('reason', '')
                audit.append(record); continue
            if quality in {'failed','skipped','unavailable','dependency_blocked'}:
                record.update(status='source_quality_excluded', reason=quality)
                audit.append(record); continue
            directory = Path(tr['directory'])
            native = read(directory / 'table.json')
            frame = read_csv(directory / 'table.csv')
            ids = r.get('source_ids', [])
            if isinstance(ids, str): ids = [ids]
            if r.get('scope') == 'record' and len(ids) == 1:
                frame['__source_id'] = ids[0]
            table = Table(frame, r['source'] + '#' + tr['path'], 'feature', native)
            try:
                samples, coordinates, values = schema(table, options)
                if native['path'] == 'Artifact.payload.periodicPeaks' and 'peak_index' in frame and 'peak_index' not in coordinates:
                    raise ConversionError('周期峰表需要peak_index坐标，当前插件尚未映射该轴；未自动聚合或展平。')
            except (Skipped, ConversionError) as exc:
                record.update(status='unsupported_mapping', reason=str(exc))
                audit.append(record); continue
            try:
                item = convert_tables([table], options)
            except Skipped as exc:
                record.update(status='unsupported_mapping', reason=str(exc))
                audit.append(record); continue
            # Exact native source path retained without inferring subject or condition from filenames.
            for sample in item.samples:
                sample['meta__source_path'] = r.get('source_path', '')
            key = (r['task'], samples)
            current_cells = sum(len(group.value.cells) for group in groups.values())
            if current_cells + len(item.cells) > cfg['max_output_cells']:
                raise MemoryError('输出标量格数量超过内存配置上限，已停止；没有删除样本或改为部分成功。')
            raw = {'file_source': str(directory / 'table.csv'), 'origin': table.origin,
                   'role': table.role, 'shape': [native['rows'], len(native['columns'])],
                   'native': native, 'source_sha256': source_hashes[r['source']]}
            groups.setdefault(key, Accumulator()).add(item, raw)
            record.update(status='converted', rows=len(item.samples), features=len(item.features))
            audit.append(record)
            del item, table, frame
    progress.update(4, '转换特征表', len(records), len(records), force=True,
                    已转换表=sum(x['status'] == 'converted' for x in audit),
                    未转换表=sum(x['status'] != 'converted' for x in audit))
    packages = []
    for number, ((task, sample_columns), accumulator) in enumerate(groups.items(), 1):
        merged = accumulator.finalize()
        dest = run / f'dataset_task_{task:02d}_{number:02d}'
        progress.update(5, f'写入数据包 {number}/{len(groups)}', current=dest, force=True)
        write_package(merged, dest, options, None)
        provenance = read(dest / 'provenance.json')
        for table_number, raw in enumerate(accumulator.raw_tables, 1):
            filename = f'source_tables/table_{table_number:04d}.csv'
            shutil.copyfile(raw['file_source'], dest / filename)
            provenance['tables'].append({k: v for k, v in dict(raw, file=filename).items() if k != 'file_source'})
        provenance['memory_policy'] = 'one source table at a time; native Artifact loaded once after memory admission; no raw DataFrame cache'
        save(dest / 'provenance.json', provenance)
        frame = read_csv(dest / 'dataset.csv')
        mask = read_csv(dest / 'valid_mask.csv')
        if frame['sample_id'].duplicated().any(): raise ValueError('输出样本ID重复')
        if frame['sample_id'].tolist() != mask['sample_id'].tolist(): raise ValueError('掩码样本顺序不一致')
        indexed = frame.set_index('sample_id')
        mask_indexed = mask.set_index('sample_id')
        count = 0
        total_cells = len(merged.samples) * len(merged.features)
        progress.update(5, f'逐值核验数据包 {number}/{len(groups)}', 0, total_cells, current=dest, force=True)
        for sample in merged.samples:
            for column in merged.features:
                value = merged.cells.get((sample['_key'], column), '')
                if indexed.at[sample['sample_id'], column] != value: raise ValueError('输出数值与原生表不一致')
                if mask_indexed.at[sample['sample_id'], column] != str(int(numeric(value))):
                    raise ValueError('有限性掩码不一致')
                count += 1
            progress.update(5, f'逐值核验数据包 {number}/{len(groups)}', count, total_cells, current=sample['sample_id'])
        progress.update(5, f'逐值核验数据包 {number}/{len(groups)}', count, total_cells, force=True)
        packages.append({'directory': str(dest), 'task': task, 'rows': len(frame),
                         'features': len(merged.features), 'cells_verified': count,
                         'sample_columns': sample_columns, 'dataset_sha256': sha(dest / 'dataset.csv')})
        print(f'转换成功：任务{task}，{len(frame)}行，{len(merged.features)}个特征。', flush=True)
    if not packages: raise ValueError('没有通过原生插件映射的数据；请查看逐表审计。')
    progress.update(6, '复核来源未变更', 0, len(input_hashes) + len(source_hashes), force=True)
    if any(sha(p) != h for p, h in input_hashes.items()): raise ValueError('转换期间输入索引发生变化')
    for verified, (p, h) in enumerate(source_hashes.items(), len(input_hashes) + 1):
        if sha(p) != h: raise ValueError(f'转换期间源特征发生变化：{p}')
        progress.update(6, '复核来源未变更', verified, len(input_hashes) + len(source_hashes), current=p)
    progress.update(6, '保存摘要', 0, 1, current=run, force=True)
    unsupported = [r for r in audit if r['status'] != 'converted']
    no_table = [r for r in records if r['status'] in {'unsupported_tensor_or_no_table', 'outside_scalar_profile'}]
    summary = {'status': 'completed_supported_tables', 'full_feature_coverage': not unsupported and not no_table,
               'packages': packages, 'native_artifacts': len(records), 'table_audit': audit,
               'unsupported_non_table_artifacts': no_table, 'unsupported_table_count': len(unsupported),
               'input_sha256': input_hashes, 'artifact_sha256': source_hashes,
               'runtime': sys.executable, 'pandas': pd.__version__, 'plugin_manifest': native_manifest,
               'plugin_source_sha256': sha(plugin / 'hypereeg_converter' / 'core.py'),
               'program_sha256': {p.name: sha(p) for p in [HERE / name for name in ['convert.py', 'export_linked.m', 'progress.py', 'export_loaded_table.m', 'streaming.py']]},
               'target_status': 'missing_not_inferred', 'training': 'NOT RUN', 'statistics': 'NOT RUN',
               'subject_identity_policy': 'Native source IDs retained. No subject, condition or target inferred from names.',
               'scientific_readiness': 'not_assessed_conversion_only'}
    save(run / 'batch_summary.json', summary)
    save(run / 'results-manifest.json', summary)
    latest({'run': str(run), 'status': summary['status'], 'packages': packages})
    text = '\n'.join(f"任务{x['task']}：{x['rows']}行 / {x['features']}个特征\n{x['directory']}\\dataset.csv" for x in packages)
    (run / '转换结果.txt').write_text('已转换插件支持的原生特征表。\n' + text +
        f'\n未映射表：{len(unsupported)}；未选择或无可转换标量表的产物：{len(no_table)}。完整状态见 batch_summary.json。\n'
        '被试身份和预测目标未推断；sample_id、meta__ 和源路径供追溯，不能当模型特征。\n'
        '整段与片段分开导出，不作为相互独立的训练样本；训练前按真实被试分组。\n', encoding='utf-8-sig')
    print(f'完成。逐值校验通过。未支持内容已记录。\n{run}', flush=True)
    progress.update(6, '保存摘要', 1, 1, current=run, force=True)
    progress.finish('completed_supported_tables', '本轮支持范围内已完成；其他范围见 batch_summary.json')
    if os.name == 'nt' and not args.no_open: os.startfile(str(run))

if __name__ == '__main__':
    try: main()
    except (Exception, KeyboardInterrupt) as exc:
        failure = traceback.format_exc()
        if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(ACTIVE_PROCESS.pid), '/T', '/F'], capture_output=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else: ACTIVE_PROCESS.terminate()
            ACTIVE_PROCESS.wait()
        status = 'cancelled' if isinstance(exc, (Cancelled, KeyboardInterrupt)) else 'failed'
        if ACTIVE_PROGRESS is not None:
            ACTIVE_PROGRESS.finish(status, failure.splitlines()[-1])
        (HERE / 'last-error.log').write_text(failure, encoding='utf-8')
        if ACTIVE_PROGRESS is not None:
            latest({'run': str(ACTIVE_PROGRESS.run), 'status': status, 'error': failure})
        print(failure, file=sys.stderr)
        sys.exit(1)
