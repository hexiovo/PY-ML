"""Measured progress snapshots; fixed display, with estimates from observed throughput."""
import json
import os
from pathlib import Path
import sys
import threading
import time

class Cancelled(Exception):
    pass

class Progress:
    def __init__(self, run):
        self.run = Path(run)
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.last_emit = 0.0
        self.segment_started = self.started
        self.segment_key = None
        self.segment_done = 0
        self.state = {'status': 'running', 'stage': 1, 'stage_count': 6,
                      'label': '检查输入', 'completed': 0, 'total': None, 'current': ''}
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def update(self, stage, label, completed=0, total=None, current='', force=False, **counts):
        if (self.run / 'cancel.request').exists():
            raise Cancelled('用户已停止转换，部分输出保留。')
        if completed < 0 or (total is not None and not 0 <= completed <= total):
            raise ValueError('Invalid measured progress counters')
        with self.lock:
            key = (stage, label, total)
            changed = key != self.segment_key
            if changed:
                self.segment_key = key
                self.segment_started = time.monotonic()
                self.segment_done = completed
            self.state.update(stage=stage, label=label, completed=completed, total=total,
                              current=str(current), counts=counts)
            self._emit(force or changed)

    def _emit(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_emit < 1.0:
            return
        self.last_emit = now
        state = dict(self.state, elapsed_seconds=round(now - self.started, 1))
        total, done = state['total'], state['completed']
        delta = done - self.segment_done
        duration = now - self.segment_started
        eta = None
        if state['status'] == 'completed_supported_tables':
            eta = 0
        elif state['status'] == 'running' and total and delta > 0 and duration >= 3 and done < total:
            eta = round(duration / delta * (total - done), 1)
        state.update(eta_seconds=eta, eta_scope='current_step', eta_basis='observed_throughput')
        (self.run / 'progress.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        measured = f'{done}/{total}' if total else '处理中'
        line = f"阶段 {state['stage']}/6 {state['label']} | 已用 {state['elapsed_seconds']:.0f}s | 预计当前步骤剩余 {eta if eta is not None else '估计中'}s"
        if os.environ.get('PYML_PROGRESS_SILENT') != '1' and sys.stdout and sys.stdout.isatty():
            print('\r' + line.ljust(160), end='', flush=True)
        with (self.run / 'progress.log').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(state, ensure_ascii=False) + '\n')

    def _heartbeat(self):
        while not self.stop.wait(5):
            with self.lock:
                self._emit(True)

    def finish(self, status, message):
        self.stop.set()
        self.thread.join(timeout=1)
        with self.lock:
            self.state.update(status=status, current=message)
            self._emit(True)
