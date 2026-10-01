from __future__ import annotations

from datetime import timedelta
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
from unittest import mock

from pyml_workbench.diagnostics import (
    DEFAULT_ACTIVE_BYTES,
    DEFAULT_BACKUP_COUNT,
    DEFAULT_PROCESS_BYTES,
    DEFAULT_RETENTION_DAYS,
    DEFAULT_ROOT_BYTES,
    LogSession,
    _utc_now,
    cleanup_owned_logs,
    owned_log_files,
    read_records,
)


def _hold_session_child(
    root: str,
    session_id: str,
    ready: object,
    finish: object,
    crash: object,
) -> None:
    logger = LogSession(root, session_id=session_id, role="worker", active_bytes=4096)
    payload = "x" * 3000
    logger.record_event("child_started", message=f"first:{payload}")
    logger.record_event("child_started", message=f"second:{payload}")
    ready.set()  # type: ignore[attr-defined]
    finish.wait(30)  # type: ignore[attr-defined]
    if crash.is_set():  # type: ignore[attr-defined]
        os._exit(19)
    logger.close()


class DiagnosticsTests(unittest.TestCase):
    def test_constants_match_approved_bounds(self):
        self.assertEqual(DEFAULT_ACTIVE_BYTES, 2 * 1024 * 1024)
        self.assertEqual(DEFAULT_BACKUP_COUNT, 4)
        self.assertEqual(DEFAULT_PROCESS_BYTES, 10 * 1024 * 1024)
        self.assertEqual(DEFAULT_ROOT_BYTES, 100 * 1024 * 1024)
        self.assertEqual(DEFAULT_RETENTION_DAYS, 30)

    def test_import_has_no_file_or_process_hook_side_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            log_root = Path(temp) / "must-not-be-created"
            script = r"""
import json, os, sys, threading, warnings
before = (sys.excepthook, threading.excepthook, warnings.showwarning)
import pyml_workbench.diagnostics
after = (sys.excepthook, threading.excepthook, warnings.showwarning)
optional = [name for name in ("torch", "matplotlib", "PyQt5", "PySide6", "PySide6.QtWidgets") if name in sys.modules]
print(json.dumps({"same_hooks": all(a is b for a, b in zip(before, after)), "optional": optional}))
"""
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
            env["PYML_LOG_ROOT"] = str(log_root)
            result = subprocess.run(
                [sys.executable, "-B", "-c", script],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                timeout=20,
            )
            observed = json.loads(result.stdout)
            self.assertTrue(observed["same_hooks"])
            self.assertEqual(observed["optional"], [])
            self.assertFalse(log_root.exists())

    def test_error_event_keeps_full_trace_unicode_context_and_error_id(self):
        with tempfile.TemporaryDirectory() as temp:
            logger = LogSession(temp, session_id="unit-error", role="worker")
            self.assertTrue(logger.available, logger.last_error)
            try:
                try:
                    raise KeyError("列名不存在")
                except KeyError as cause:
                    raise RuntimeError("训练阶段失败：目标列") from cause
            except RuntimeError as exc:
                record = logger.record_error(
                    exc,
                    context={"stage": "fit", "model": "C01", "job_id": "job-7", "dataset": "must-not-log"},
                )
            records = read_records(temp, session_id="unit-error", error_ids=[record.error_id])
            logger.close()

        self.assertEqual(len(records), 1)
        event = records[0]
        self.assertEqual(event["event"], "error")
        self.assertEqual(event["error_id"], record.error_id)
        self.assertEqual(event["error_type"], "RuntimeError")
        self.assertIn("训练阶段失败：目标列", event["message"])
        self.assertIn("列名不存在", event["traceback"])
        self.assertIn("The above exception was the direct cause", event["traceback"])
        self.assertEqual(event["stage"], "fit")
        self.assertEqual(event["model"], "C01")
        self.assertEqual(event["job_id"], "job-7")
        self.assertEqual(record.to_dict(), event)
        self.assertEqual(set(event), {
            "schema", "timestamp_utc", "level", "event", "session_id", "process_id",
            "process_nonce", "role", "version", "stage", "model", "job_id", "error_id",
            "error_type", "message", "traceback",
        })
        self.assertNotIn("dataset", event)

    def test_owned_registry_ignores_unregistered_files_and_read_filter_works(self):
        with tempfile.TemporaryDirectory() as temp:
            logger = LogSession(temp, session_id="unit-registry", role="gui")
            logger.record_event("startup", message="ready")
            stray = logger.session_dir / "unregistered.jsonl"  # type: ignore[operator]
            stray.write_text('{"message":"not owned"}\n', encoding="utf-8")
            files = owned_log_files(temp)
            self.assertTrue(all("unregistered" not in path.name for path in files))
            self.assertEqual(len([path for path in files if ".jsonl" in path.name]), 5)
            self.assertEqual(len(read_records(temp, session_id="unit-registry", event="startup")), 1)
            self.assertEqual(read_records(temp, session_id="unit-registry", error_ids=[]), [])
            logger.close()

    def test_rotation_keeps_five_files_and_per_process_size_under_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            logger = LogSession(temp, session_id="unit-roll", role="worker")
            payload = "x" * 1_900_000
            for index in range(12):
                logger.record_external_error(
                    "TestError",
                    f"large record {index}",
                    f"{index}:{payload}",
                    error_id=f"{index + 1:032x}",
                )
                self.assertTrue(logger.available, logger.last_error)
            paths = [path for path in owned_log_files(temp, "unit-roll") if ".jsonl" in path.name]
            self.assertEqual(len(paths), DEFAULT_BACKUP_COUNT + 1)
            sizes = [path.stat().st_size for path in paths]
            self.assertTrue(all(size <= DEFAULT_ACTIVE_BYTES for size in sizes), sizes)
            self.assertLessEqual(sum(sizes), DEFAULT_PROCESS_BYTES)
            self.assertEqual(sum(size > 0 for size in sizes), DEFAULT_BACKUP_COUNT + 1)
            logger.close()

    def test_write_failure_does_not_replace_exception_or_pollute_stdout(self):
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as temp:
            logger = LogSession(temp, session_id="unit-failure", role="gui")
            original = RuntimeError("original business exception")
            output = StringIO()
            with redirect_stdout(output):
                with self.assertRaises(RuntimeError) as raised:
                    try:
                        raise original
                    except RuntimeError as caught:
                        with mock.patch.object(logger, "_append_line", side_effect=OSError("disk full")):
                            record = logger.record_error(caught)
                        self.assertIs(caught, original)
                        raise
            self.assertIs(raised.exception, original)
            self.assertEqual(type(original), RuntimeError)
            self.assertEqual(record.error_type, "RuntimeError")
            self.assertIsNotNone(record.error_id)
            self.assertFalse(logger.available)
            self.assertIn("disk full", logger.last_error or "")
            self.assertEqual(output.getvalue(), "")
            logger.close()

    def test_concurrent_threads_write_complete_jsonl_records(self):
        with tempfile.TemporaryDirectory() as temp:
            logger = LogSession(temp, session_id="unit-threads", role="worker")

            def write_group(group: int) -> None:
                for index in range(12):
                    logger.record_event("thread_event", message=f"{group}:{index}")

            threads = [threading.Thread(target=write_group, args=(index,)) for index in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(15)
                self.assertFalse(thread.is_alive())
            rows = read_records(temp, session_id="unit-threads", event="thread_event")
            logger.close()
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({row["message"] for row in rows}), 48)

    def test_retention_budget_protects_active_and_only_deletes_registered_files(self):
        with tempfile.TemporaryDirectory() as temp:
            logger = LogSession(temp, session_id="unit-retention", role="gui")
            logger.record_event("older", message="preserve while active")
            foreign = Path(temp) / "user-file.bin"
            foreign.write_bytes(b"leave this file alone")
            foreign_before = foreign.read_bytes()
            old_time = time.time() - 40 * 24 * 60 * 60
            for path in owned_log_files(temp, "unit-retention"):
                os.utime(path, (old_time, old_time))
            result = cleanup_owned_logs(temp, now=_utc_now(), max_root_bytes=1)
            self.assertIn("unit-retention", result.retained_active_sessions)
            self.assertTrue(owned_log_files(temp, "unit-retention"))
            logger.close()
            result = cleanup_owned_logs(temp, now=_utc_now(), max_root_bytes=1)
            self.assertIn("unit-retention", result.removed_sessions)
            self.assertEqual(foreign.read_bytes(), foreign_before)
            self.assertTrue(foreign.exists())

    def test_two_child_sessions_are_protected_then_crash_orphan_is_reclaimed(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as temp:
            signals: list[tuple[object, object, object, object]] = []
            children: list[multiprocessing.Process] = []
            try:
                for session_id in ("child-one", "child-two"):
                    ready = context.Event()
                    finish = context.Event()
                    crash = context.Event()
                    process = context.Process(
                        target=_hold_session_child,
                        args=(temp, session_id, ready, finish, crash),
                    )
                    process.start()
                    self.assertTrue(ready.wait(20), f"{session_id} did not acquire its log lock")
                    children.append(process)
                    signals.append((ready, finish, crash, process))

                future = _utc_now() + timedelta(days=DEFAULT_RETENTION_DAYS + 2)
                first = cleanup_owned_logs(temp, now=future)
                self.assertIn("child-one", first.retained_active_sessions)
                self.assertIn("child-two", first.retained_active_sessions)
                first_files = owned_log_files(temp, "child-one")
                second_files = owned_log_files(temp, "child-two")
                self.assertTrue(first_files)
                self.assertTrue(second_files)
                for file_set in (first_files, second_files):
                    nonempty = [path for path in file_set if ".jsonl" in path.name and path.stat().st_size]
                    self.assertGreaterEqual(len(nonempty), 2)

                signals[0][2].set()  # type: ignore[attr-defined]
                signals[0][1].set()  # type: ignore[attr-defined]
                children[0].join(20)
                self.assertFalse(children[0].is_alive())
                self.assertEqual(children[0].exitcode, 19)
                after_crash = cleanup_owned_logs(temp, now=future)
                self.assertIn("child-one", after_crash.removed_sessions)
                self.assertTrue(owned_log_files(temp, "child-two"))
                self.assertEqual(owned_log_files(temp, "child-one"), ())

                signals[1][1].set()  # type: ignore[attr-defined]
                children[1].join(20)
                self.assertFalse(children[1].is_alive())
                after_close = cleanup_owned_logs(temp, now=future)
                self.assertIn("child-two", after_close.removed_sessions)
                self.assertEqual(owned_log_files(temp, "child-two"), ())
            finally:
                for signal in signals:
                    signal[1].set()  # type: ignore[attr-defined]
                    signal[2].set()  # type: ignore[attr-defined]
                for process in children:
                    process.join(5)
                    if process.is_alive():
                        process.terminate()
                        process.join(5)


if __name__ == "__main__":
    unittest.main()
