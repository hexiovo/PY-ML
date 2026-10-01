import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from pyml_workbench import diagnostics_archive as archive
from pyml_workbench.diagnostics import LogSession, owned_log_files


SESSION_A = "a" * 32
SESSION_B = "b" * 32
ERROR_A = "1" * 32
ERROR_B = "2" * 32
NONCE = "c" * 32
LOG_NAME = f"process-321-{NONCE}.jsonl"


def _record(
    *,
    session_id=SESSION_A,
    error_id=ERROR_A,
    message="failed",
    traceback_text="Traceback (most recent call last):\nValueError: failed",
    event="error",
):
    return {
        "schema": 1,
        "timestamp_utc": "2026-10-01T12:00:00+00:00",
        "level": "ERROR" if event == "error" else "INFO",
        "event": event,
        "session_id": session_id,
        "process_id": 321,
        "process_nonce": NONCE,
        "role": "gui",
        "version": "0.4.dev0",
        "stage": "fit",
        "model": "C01",
        "job_id": None,
        "error_id": error_id,
        "error_type": "ValueError" if event == "error" else None,
        "message": message,
        "traceback": traceback_text if event == "error" else None,
    }


class ArchiveFixture:
    def __init__(self, root: Path):
        self.root = root
        self.sessions: dict[str, list[dict]] = {}
        self.paths: dict[str, tuple[Path, ...]] = {}

    def add_session(self, session_id: str, records: list[dict]) -> Path:
        session_dir = self.root / session_id
        session_dir.mkdir(parents=True)
        log_path = session_dir / LOG_NAME
        log_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
            encoding="utf-8",
        )
        log_paths = [log_path]
        for index in range(1, 5):
            rotated = session_dir / f"{LOG_NAME}.{index}"
            rotated.write_bytes(b"")
            log_paths.append(rotated)
        (session_dir / "manifest.tmp").write_text(
            "TEMPORARY_PRIVATE_SENTINEL", encoding="utf-8"
        )
        manifest = {
            "schema": 1,
            "owner": "pyml-workbench",
            "session_id": session_id,
            "process": {"pid": 321, "nonce": NONCE},
            "role": "gui",
            "created_files": [
                {"path": "active.lock", "kind": "lock"},
                {"path": "manifest.json", "kind": "manifest"},
                {"path": "manifest.tmp", "kind": "temp"},
                {"path": LOG_NAME, "kind": "jsonl"},
                *[
                    {"path": f"{LOG_NAME}.{index}", "kind": "jsonl"}
                    for index in range(1, 5)
                ],
            ],
            "created_at_utc": "2026-10-01T12:00:00+00:00",
            "updated_at_utc": "2026-10-01T12:00:00+00:00",
        }
        manifest_path = session_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.sessions[session_id] = records
        self.paths[session_id] = (*log_paths, manifest_path)
        return session_dir

    def diagnostics_module(self):
        module = types.ModuleType("pyml_workbench.diagnostics")

        def read_records(log_root, *, session_id=None, error_ids=None, **_filters):
            session_ids = (session_id,) if session_id is not None else tuple(self.sessions)
            rows = [
                row
                for selected_id in session_ids
                for row in self.sessions.get(selected_id, ())
            ]
            if error_ids is not None:
                selected = set(error_ids)
                rows = [row for row in rows if row.get("error_id") in selected]
            return rows

        def owned_log_files(log_root, session_id=None):
            session_ids = (session_id,) if session_id is not None else tuple(self.sessions)
            return tuple(path for selected_id in session_ids for path in self.paths.get(selected_id, ()))

        module.read_records = read_records
        module.owned_log_files = owned_log_files
        return module


class DiagnosticsArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.log_root = self.root / "logs"
        self.log_root.mkdir()
        self.fixture = ArchiveFixture(self.log_root)
        self.diagnostics_patch = patch.dict(
            sys.modules,
            {"pyml_workbench.diagnostics": self.fixture.diagnostics_module()},
        )
        self.diagnostics_patch.start()
        self.addCleanup(self.diagnostics_patch.stop)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _export(self, name="diagnostics.zip", **kwargs):
        return archive.export_diagnostics_zip(
            self.log_root,
            self.root / name,
            app_version="0.4.dev0",
            **kwargs,
        )

    def _source_hashes(self):
        return {
            str(path.relative_to(self.log_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.log_root.rglob("*")
            if path.is_file()
        }

    def test_exports_allowlisted_sanitized_archive_and_preserves_sources(self):
        message = (
            "password=hunter2, api_key=sk-12345678901234567890 "
            'and JSON {"password": "jsonsecret"} '
            "and https://alice:urlysecret@example.test/resource "
            "and -----BEGIN PRIVATE KEY-----pemsecret-----END PRIVATE KEY----- "
            "at C:\\Users\\Alice\\input.csv and /home/alice/private.csv; "
            "D:\\Data Files\\My Project\\train.csv; keep marker"
        )
        traceback_text = (
            'File "F:\\桌面\\程序\\PY-ML\\src\\gui.py", line 8\n'
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789"
        )
        self.fixture.add_session(
            SESSION_A,
            [_record(message=message, traceback_text=traceback_text)],
        )
        source_hashes = self._source_hashes()

        result = self._export(session_ids=[SESSION_A])

        self.assertEqual(result, self.root / "diagnostics.zip")
        with zipfile.ZipFile(result) as package:
            names = set(package.namelist())
            self.assertEqual(
                names,
                {
                    "manifest.json",
                    "summary.json",
                    "tracebacks.json",
                    f"logs/{SESSION_A}/{LOG_NAME}",
                    *[f"logs/{SESSION_A}/{LOG_NAME}.{index}" for index in range(1, 5)],
                },
            )
            members = {name: package.read(name) for name in names}
        exported_text = b"\n".join(members.values()).decode("utf-8")
        self.assertIn(ERROR_A, exported_text)
        for secret in (
            "hunter2",
            "jsonsecret",
            "urlysecret",
            "pemsecret",
            "sk-12345678901234567890",
            "abcdefghijklmnopqrstuvwxyz0123456789",
            "C:\\Users\\Alice",
            "/home/alice",
            "Data Files",
            "train.csv",
            "F:\\桌面\\程序\\PY-ML",
            "TEMPORARY_PRIVATE_SENTINEL",
        ):
            self.assertNotIn(secret, exported_text)
        self.assertIn("keep marker", exported_text)
        self.assertIn("[REDACTED]", exported_text)
        self.assertIn("[LOCAL_PATH]", exported_text)
        summary = json.loads(members["summary.json"])
        self.assertEqual(summary["schema"], 1)
        self.assertEqual([item["error_id"] for item in summary["errors"]], [ERROR_A])
        manifest = json.loads(members["manifest.json"])
        self.assertEqual(
            set(manifest), {"schema", "app_version", "python_version", "platform"}
        )
        self.assertLessEqual(result.stat().st_size, archive.MAX_ARCHIVE_BYTES)
        self.assertEqual(source_hashes, self._source_hashes())

    def test_error_selection_resolves_only_owned_session_and_only_selected_traceback(self):
        self.fixture.add_session(
            SESSION_A,
            [
                _record(error_id=ERROR_A, traceback_text="selected traceback"),
                _record(
                    error_id=ERROR_B,
                    message="second error",
                    traceback_text="second traceback sentinel",
                ),
            ],
        )
        self.fixture.add_session(
            SESSION_B,
            [
                _record(
                    session_id=SESSION_B,
                    error_id="3" * 32,
                    message="other session",
                    traceback_text="other session sentinel",
                )
            ],
        )

        result = self._export(error_ids=[ERROR_A])

        with zipfile.ZipFile(result) as package:
            names = package.namelist()
            self.assertTrue(all(f"logs/{SESSION_A}/" in name for name in names if name.startswith("logs/")))
            self.assertFalse(any(name.startswith(f"logs/{SESSION_B}/") for name in names))
            summary = json.loads(package.read("summary.json"))
            tracebacks = json.loads(package.read("tracebacks.json"))
            log_text = package.read(f"logs/{SESSION_A}/{LOG_NAME}").decode("utf-8")
        self.assertEqual([item["error_id"] for item in summary["errors"]], [ERROR_A])
        self.assertEqual([item["error_id"] for item in tracebacks["errors"]], [ERROR_A])
        self.assertIn("selected traceback", log_text)
        self.assertNotIn("second traceback sentinel", log_text)
        self.assertNotIn("other session sentinel", log_text)

    def test_selected_session_without_errors_exports_empty_summary(self):
        self.fixture.add_session(
            SESSION_A,
            [_record(error_id=None, event="event", message="cancelled", traceback_text="")],
        )

        result = self._export(session_ids=[SESSION_A])

        with zipfile.ZipFile(result) as package:
            summary = json.loads(package.read("summary.json"))
            tracebacks = json.loads(package.read("tracebacks.json"))
        self.assertEqual(summary, {"schema": 1, "errors": []})
        self.assertEqual(tracebacks, {"schema": 1, "errors": []})

    def test_export_requires_explicit_nonempty_selection(self):
        self.fixture.add_session(SESSION_A, [_record()])
        for kwargs in ({}, {"session_ids": []}, {"error_ids": []}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(archive.DiagnosticArchiveError):
                    self._export("rejected.zip", **kwargs)
                self.assertFalse((self.root / "rejected.zip").exists())

    def test_missing_or_foreign_error_is_rejected_without_creating_output(self):
        self.fixture.add_session(SESSION_A, [_record()])

        with self.assertRaises(archive.DiagnosticArchiveError):
            self._export("missing.zip", error_ids=["f" * 32])

        self.assertFalse((self.root / "missing.zip").exists())

    def test_malformed_manifest_keys_and_traversal_are_rejected(self):
        session_dir = self.fixture.add_session(SESSION_A, [_record()])
        manifest_path = session_dir / "manifest.json"
        original = json.loads(manifest_path.read_text(encoding="utf-8"))

        malformed = dict(original, unexpected="private")
        manifest_path.write_text(json.dumps(malformed), encoding="utf-8")
        with self.assertRaises(archive.DiagnosticArchiveError):
            self._export("malformed.zip", session_ids=[SESSION_A])
        self.assertFalse((self.root / "malformed.zip").exists())

        traversal = dict(original)
        traversal["created_files"] = list(original["created_files"])
        traversal["created_files"][3] = {"path": "../outside.jsonl", "kind": "jsonl"}
        manifest_path.write_text(json.dumps(traversal), encoding="utf-8")
        with self.assertRaises(archive.DiagnosticArchiveError):
            self._export("traversal.zip", session_ids=[SESSION_A])
        self.assertFalse((self.root / "traversal.zip").exists())

    def test_registered_symlink_and_unregistered_caller_path_are_rejected(self):
        session_dir = self.fixture.add_session(SESSION_A, [_record()])
        outside = self.root / "outside.jsonl"
        outside.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
        link_path = session_dir / LOG_NAME
        link_path.unlink()
        try:
            os.symlink(outside, link_path)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks are unavailable in this test environment")

        with self.assertRaises(archive.DiagnosticArchiveError):
            self._export("linked.zip", session_ids=[SESSION_A])
        self.assertFalse((self.root / "linked.zip").exists())

    def test_reparse_detection_fails_closed_even_when_host_cannot_create_symlinks(self):
        session_dir = self.fixture.add_session(SESSION_A, [_record()])
        log_path = session_dir / LOG_NAME
        real_check = archive._is_link_or_reparse

        def report_registered_log_as_reparse(path):
            if Path(path) == log_path:
                return True
            return real_check(Path(path))

        with patch.object(
            archive,
            "_is_link_or_reparse",
            side_effect=report_registered_log_as_reparse,
        ):
            with self.assertRaises(archive.DiagnosticArchiveError):
                self._export("reparse.zip", session_ids=[SESSION_A])

        self.assertFalse((self.root / "reparse.zip").exists())

    def test_unknown_diagnostic_jsonl_fields_are_rejected(self):
        session_dir = self.fixture.add_session(SESSION_A, [_record()])
        log_path = session_dir / LOG_NAME
        row = _record()
        row["exception_locals"] = {"password": "must never be exported"}
        log_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

        with self.assertRaises(archive.DiagnosticArchiveError):
            self._export("unknown-field.zip", session_ids=[SESSION_A])

        self.assertFalse((self.root / "unknown-field.zip").exists())

    def test_real_log_session_registry_and_wire_schema_integrate(self):
        session_id = "d" * 32
        session = LogSession(self.log_root, session_id=session_id, role="gui")
        self.assertTrue(session.available, session.last_error)
        record = session.record_external_error(
            "ValueError",
            "password=integration-secret at C:\\Users\\Alice\\dataset.csv",
            traceback_text=(
                'Traceback (most recent call last):\n'
                '  File "F:\\work\\pyml\\gui.py", line 7\n'
                "ValueError: password=integration-secret"
            ),
            context={"stage": "integration", "model": "C01"},
        )
        session.close()

        # Use the real registry after the fixture module has been removed.
        self.diagnostics_patch.stop()
        owned = owned_log_files(self.log_root, session_id)
        self.assertEqual(len(owned), 6)  # five registered JSONL files plus manifest
        result = archive.export_diagnostics_zip(
            self.log_root,
            self.root / "real-session.zip",
            error_ids=[record.error_id],
            app_version="0.4.dev0",
        )

        with zipfile.ZipFile(result) as package:
            names = package.namelist()
            summary = json.loads(package.read("summary.json"))
            tracebacks = json.loads(package.read("tracebacks.json"))
            text = b"\n".join(package.read(name) for name in names).decode("utf-8")
        self.assertEqual([item["error_id"] for item in summary["errors"]], [record.error_id])
        self.assertEqual([item["error_id"] for item in tracebacks["errors"]], [record.error_id])
        self.assertNotIn("integration-secret", text)
        self.assertNotIn("C:\\Users\\Alice", text)
        self.assertNotIn("F:\\work\\pyml", text)
        self.assertIn(record.error_id, text)

    def test_existing_destination_is_rejected_without_overwrite(self):
        self.fixture.add_session(SESSION_A, [_record()])
        destination = self.root / "existing.zip"
        destination.write_bytes(b"keep me")
        before = destination.read_bytes()

        with self.assertRaises(archive.DiagnosticArchiveError):
            self._export("existing.zip", session_ids=[SESSION_A])

        self.assertEqual(destination.read_bytes(), before)

    def test_archive_input_limit_rejects_oversized_log_before_output(self):
        self.fixture.add_session(
            SESSION_A,
            [_record(message="x" * 4_000, traceback_text="y" * 4_000)],
        )

        with patch.object(archive, "MAX_ARCHIVE_BYTES", 2_000):
            with self.assertRaises(archive.DiagnosticArchiveError):
                self._export("oversized.zip", session_ids=[SESSION_A])

        self.assertFalse((self.root / "oversized.zip").exists())


if __name__ == "__main__":
    unittest.main()
