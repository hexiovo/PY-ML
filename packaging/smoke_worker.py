"""Exercise actual frozen workers using temporary synthetic data, no source path."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('distribution', type=Path)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--search-only', action='store_true')
    args = parser.parse_args()
    worker = args.distribution.resolve() / 'PYML-Worker.exe'
    evidence = args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    checks = []
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
        env.pop(key, None)
    env['PATH'] = os.pathsep.join([str(Path(os.environ['SystemRoot']) / 'System32'), os.environ['SystemRoot']])
    with tempfile.TemporaryDirectory(prefix='PYML-中文EXE-') as temp:
        root = Path(temp)
        env['PYML_LOG_ROOT'] = str(root / 'logs')
        source = root / '数据.csv'
        with source.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['x', 'category', 'target', 'value'])
            for i in range(100):
                writer.writerow([i / 20, '甲' if i % 2 else '乙', i % 2, math.sin(i / 8)])

        def call(name, command, *, success=True):
            start = time.monotonic()
            result = subprocess.run([str(worker), *command], cwd=root, env=env,
                capture_output=True, text=True, encoding='utf-8', timeout=180,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            (evidence / f'{name}.stdout.jsonl').write_text(result.stdout, encoding='utf-8')
            (evidence / f'{name}.stderr.txt').write_text(result.stderr, encoding='utf-8')
            events = [json.loads(line) for line in result.stdout.splitlines()]
            assert result.returncode == (0 if success else 1), (name, result.stderr[-2500:])
            assert events and events[-1]['type'] == ('result' if success else 'error'), (name, events)
            checks.append({'name': name, 'returncode': result.returncode,
                'seconds': round(time.monotonic()-start, 3), 'event_count': len(events)})
            print(f'PASS {name}', flush=True)
            return events[-1]

        configs = [
            ('C01', 'classification', ['x', 'category'], 'target', {'max_iter': 100}, None),
            ('H01', 'sequence_modeling', ['value'], None, {'n_components': 2, 'n_iter': 3}, None),
            ('N01', 'classification', ['x'], 'target', {'hidden_size': 8, 'max_epochs': 2, 'batch_size': 16}, None),
            ('N06', 'regression', ['value'], 'value', {'hidden_size': 8, 'max_epochs': 2, 'batch_size': 16, 'patience': 2},
                {'window': 5, 'horizon': 1}),
        ]
        if args.search_only:
            configs = []
        # Window regression uses a separate feature so the target is not a feature.
        series = root / '序列.csv'
        with series.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['signal', 'target'])
            for i in range(120):
                writer.writerow([math.sin(i/8), math.sin((i+1)/8)])
        for model, task, features, target, parameters, sequence in configs:
            dataset = {'source_path': str(source), 'feature_columns': features, 'target_column': target}
            if model == 'N06':
                dataset.update(source_path=str(series), feature_columns=['signal'], target_column='target')
            config = {'dataset': dataset, 'task': task, 'model_id': model, 'parameters': parameters}
            if sequence:
                config['sequence'] = sequence
            path, session = root / f'{model}.json', root / f'{model}.joblib'
            path.write_text(json.dumps(config, ensure_ascii=False), encoding='utf-8')
            common = ['--config', str(path), '--session', str(session)]
            trained = call(f'{model}-train', ['train', *common])
            assert trained['state'] == 'trained' and 'test' not in trained['metrics']
            common += ['--session-id', trained['session_id']]
            call(f'{model}-freeze', ['freeze', *common])
            tested = call(f'{model}-test', ['test', *common])
            assert type(tested['test_evaluation_count']) is int and tested['test_evaluation_count'] == 1
            cached = call(f'{model}-cached', ['test', *common])
            assert cached['cached'] is True and cached['test_evaluation_count'] == 1
            exported = call(f'{model}-export', ['export', *common, '--output', str(root / f'{model}-output')])
            model_file = exported['artifact_paths']['model']
            call(f'{model}-inspect', ['inspect', '--model', model_file])
            if model in {'C01', 'H01'}:
                inferred = call(f'{model}-inference', ['inference', '--operation', 'predict',
                    '--model', model_file, '--data', str(source), '--output', str(root / f'{model}-pred.csv')])
                assert inferred['rows'] == 100

        requests = []
        for method in ('grid', 'random', 'annealing', 'tpe', 'genetic'):
            requests.append({'config': {'dataset': {'source_path': str(source), 'feature_columns': ['x'],
                'target_column': 'target'}, 'task': 'classification', 'model_id': 'C01',
                'parameters': {'max_iter': 80}}, 'spec': {'method': method,
                'space': {'fields': {'C': {'type': 'real', 'low': 0.1, 'high': 2.0, 'values': [0.1, 1.0]}}},
                'max_fits': 2, 'max_proposals': 20, 'timeout_seconds': 60}})
        request_path = root / 'requests.json'
        request_path.write_text(json.dumps(requests), encoding='utf-8')
        result = call('search-all-five', ['batch-run-request', '--history', str(root / 'history.db'),
            '--root', str(root / 'batch'), '--requests', str(request_path)])
        assert len(result['outcomes']) == 5
        assert all(item['winner_value'] is not None and item['actual_fit_count'] > 0
            and not item.get('error') for item in result['outcomes']), result
        error = call('diagnostic-error', ['inspect', '--model', str(root / 'missing.joblib')], success=False)
        assert len(error['error_id']) == 32 and 'Traceback' in error['traceback']
        logs = list((root / 'logs').rglob('*.jsonl*'))
        assert any(error['error_id'] in p.read_text(encoding='utf-8') for p in logs if p.is_file())
        report = {'overall': 'PASS', 'worker': str(worker), 'source_pythonpath_removed': True,
            'cwd': 'temporary Unicode directory outside source', 'checks': checks,
            'search_methods': ['grid', 'random', 'annealing', 'tpe', 'genetic'],
            'limitations': ['Representative models only; not 78 full fits.',
                'Target host is this Windows x64 machine; no clean VM test.']}
        (evidence / 'worker-result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'PASS total={len(checks)}', flush=True)


if __name__ == '__main__':
    main()
