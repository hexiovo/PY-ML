"""Verify new models using the shipped worker in an isolated Windows environment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('distribution', type=Path)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--model-id', action='append', help='limit model smoke to selected IDs')
    parser.add_argument('--cv-only', action='store_true', help='recheck the CV batch contract without repeating model fits')
    args = parser.parse_args()
    distribution = args.distribution.resolve()
    worker = distribution / 'PYML-Worker.exe'
    evidence = args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    checks, failures = [], []
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
        env.pop(key, None)
    env['PATH'] = os.pathsep.join([str(Path(os.environ['SystemRoot']) / 'System32'), os.environ['SystemRoot']])
    with tempfile.TemporaryDirectory(prefix='PYML-扩展模型-') as directory:
        root = Path(directory)
        env['PYML_LOG_ROOT'] = str(root / 'logs')
        env['NUMBA_CACHE_DIR'] = str(root / 'numba-cache')
        env['APPDATA'] = str(root / 'appdata')
        env['LOCALAPPDATA'] = str(root / 'local-appdata')
        env['PYML_OPERATION_LOG_PATH'] = str(root / 'operations.jsonl')
        source = root / '合成数据.csv'
        rng = random.Random(20261027)
        with source.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['a', 'b', 'c', 'label', 'target'])
            for index in range(180):
                a, b, c = (rng.gauss(0, 1) for _ in range(3))
                writer.writerow([a, b, c, '甲' if a + b > 0 else '乙',
                                 10 + index * .02 + math.sin(index * math.pi / 6) + a * .2])
        future = root / '未来步.csv'
        future.write_text('step\n1\n2\n3\n4\n5\n6\n7\n8\n', encoding='utf-8')

        def call(name, command, *, timeout=240):
            started = time.monotonic()
            completed = subprocess.run([str(worker), *command], cwd=root, env=env,
                capture_output=True, text=True, encoding='utf-8', timeout=timeout,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            (evidence / f'{name}.stdout.jsonl').write_text(completed.stdout, encoding='utf-8')
            (evidence / f'{name}.stderr.txt').write_text(completed.stderr, encoding='utf-8')
            events = [json.loads(line) for line in completed.stdout.splitlines()]
            if completed.returncode or not events or events[-1].get('type') != 'result':
                message = events[-1] if events else completed.stderr[-4000:]
                raise AssertionError(f'{name}: exit={completed.returncode}; {message}')
            checks.append({'name': name, 'seconds': round(time.monotonic() - started, 3),
                           'returncode': completed.returncode, 'event_count': len(events)})
            print(f'OK {name}', flush=True)
            return events[-1]

        cases = []
        for model in ('C28', 'C29', 'C30', 'C31', 'C25', 'C26', 'C27'):
            parameters = {'member_parameters': {'C16': {'n_estimators': 10}}} if model in {'C29', 'C30'} else {}
            if model in {'C25', 'C26'}:
                parameters = {'n_estimators': 5}
            elif model == 'C27':
                parameters = {'iterations': 5}
            cases.append((model, 'classification', parameters))
        for model in ('R27', 'R28', 'R29', 'R30', 'R31', 'R32', 'R33', 'R34', 'R35', 'R36', 'R24', 'R25', 'R26'):
            parameters = {'member_parameters': {'R14': {'n_estimators': 10}}} if model in {'R28', 'R29'} else {}
            if model in {'R24', 'R25'}:
                parameters = {'n_estimators': 5}
            elif model == 'R26':
                parameters = {'iterations': 5}
            cases.append((model, 'regression', parameters))
        cases.extend((model, 'clustering', {}) for model in ('K11', 'K12', 'K13'))
        cases.extend([('D11', 'dimensionality reduction', {'n_epochs': 10, 'n_neighbors': 5}),
                      ('D12', 'dimensionality reduction', {'max_epochs': 2})])
        cases.extend((model, 'regression', {}) for model in ('T01', 'T02', 'T03', 'T04', 'T05'))
        cases.extend((model, 'regression', {'max_epochs': 2, 'hidden_size': 8, 'patience': 1})
                     for model in ('N07', 'N08', 'N09'))
        selected = set(args.model_id or ())
        lifecycle = {'C30', 'R29', 'D11', 'D12', 'T01', 'T03', 'T04', 'T05', 'N07', 'N08', 'N09'}
        for model, task, parameters in cases:
            if args.cv_only:
                continue
            if selected and model not in selected:
                continue
            try:
                dataset = {'source_path': str(source), 'feature_columns': ['a', 'b', 'c']}
                if task in {'classification', 'regression'}:
                    dataset['target_column'] = 'label' if task == 'classification' else 'target'
                config = {'dataset': dataset, 'task': task, 'model_id': model, 'parameters': parameters}
                if model.startswith('T'):
                    dataset['feature_columns'] = None
                    config['split'] = {'strategy': 'time'}
                elif model.startswith('N'):
                    config['sequence'] = {'window': 5, 'horizon': 1}
                path, session = root / f'{model}.json', root / f'{model}.joblib'
                path.write_text(json.dumps(config, ensure_ascii=False), encoding='utf-8')
                common = ['--config', str(path), '--session', str(session)]
                trained = call(f'{model}-train', ['train', *common])
                assert trained['state'] == 'trained' and 'test' not in trained['metrics']
                assert trained['test_evaluation_count'] == 0
                if model not in lifecycle:
                    continue
                common += ['--session-id', trained['session_id']]
                call(f'{model}-freeze', ['freeze', *common])
                tested = call(f'{model}-test', ['test', *common])
                assert tested['test_evaluation_count'] == 1
                cached = call(f'{model}-cached', ['test', *common])
                assert cached['cached'] is True and cached['test_evaluation_count'] == 1
                exported = call(f'{model}-export', ['export', *common, '--output', str(root / f'{model}-output')])
                model_file = exported['artifact_paths']['model']
                call(f'{model}-inspect', ['inspect', '--model', model_file])
                operation = 'transform' if task == 'dimensionality reduction' else 'predict'
                inferred = call(f'{model}-inference', ['inference', '--operation', operation,
                    '--model', model_file, '--data', str(future if model.startswith('T') else source),
                    '--output', str(root / f'{model}-prediction.csv')])
                expected = 8 if model.startswith('T') else 175 if model.startswith('N') else 180
                assert inferred['rows'] == expected, (model, inferred)
            except Exception as exc:
                failures.append({'model_id': model, 'error': str(exc)})
                print(f'FAILED {model}: {exc}', flush=True)

        if not selected:
            request = [{'config': {'dataset': {'source_path': str(source), 'feature_columns': ['a', 'b', 'c'],
                        'target_column': 'label'}, 'task': 'classification', 'model_id': 'C01',
                        'evaluation': {'folds': 3}},
                        'spec': {'max_fits': 4, 'objective': {'metric': 'log_loss'}}}]
            path = root / 'cv-request.json'
            path.write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
            try:
                result = call('cv-batch', ['batch-run-request', '--history', str(root / 'cv.sqlite'),
                    '--root', str(root / 'cv-batch'), '--requests', str(path)])
                outcome = result['outcomes'][0]
                assert outcome['actual_fit_count'] == 4 and outcome['winner_value'] is not None, outcome
                job_id = outcome['job_id']
                call('cv-freeze', ['batch-freeze', '--history', str(root / 'cv.sqlite'), '--job-id', job_id])
                final = call('cv-finalize', ['batch-finalize', '--history', str(root / 'cv.sqlite'), '--job-id', job_id])
                assert final['final_selection']['test_evaluation_count'] == 1
            except Exception as exc:
                failures.append({'model_id': 'cross_validation', 'error': str(exc)})
                print(f'FAILED cross-validation: {exc}', flush=True)

        report = {'result': 'OK' if not failures else 'FAILED',
                  'worker': str(worker), 'worker_sha256': hashlib.sha256(worker.read_bytes()).hexdigest(),
                  'models_requested': [] if args.cv_only else [model for model, _, _ in cases if not selected or model in selected],
                  'checks': checks, 'failures': failures, 'seed': 20261027,
                  'environment': 'external temporary Unicode directory, no Python paths, system-only PATH',
                  'scope': '27 new models plus 6 boosting models; representative full lifecycles and cross-validation'}
        (evidence / 'expansion-worker-result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{report["result"]}: {len(checks)} subprocess checks, {len(failures)} failures', flush=True)
        return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
