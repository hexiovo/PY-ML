"""Explicit Windows taskbar metadata for the workbench's native window."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path
import subprocess
import sys
import uuid


APP_ID = "hexiovo.PYML.Workbench"


def configure_taskbar_window(hwnd: int, icon_path: Path) -> None:
    """Bind the taskbar icon and relaunch target before the window is shown.

    A process AppUserModelID controls grouping, but does not provide Explorer
    with an icon/relaunch source when launching through a separate wrapper.
    """
    if sys.platform != "win32":
        return

    class GUID(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * 16)]

        def __init__(self, value: str):
            super().__init__()
            ctypes.memmove(ctypes.byref(self), uuid.UUID(value).bytes_le, 16)

    class PROPERTYKEY(ctypes.Structure):
        _fields_ = [("fmtid", GUID), ("pid", wintypes.DWORD)]

    class PROPVARIANT(ctypes.Structure):
        _fields_ = [
            ("vt", ctypes.c_ushort),
            ("reserved", ctypes.c_ushort * 3),
            ("value", ctypes.c_void_p),
            ("padding", ctypes.c_ulonglong),
        ]

    def checked(result: int) -> None:
        if result < 0:
            raise OSError(result, "无法设置 Windows 任务栏属性")

    ole32 = ctypes.WinDLL("ole32")
    shell32 = ctypes.WinDLL("shell32")
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    ole32.CoInitializeEx.restype = ctypes.c_long
    initialized = ole32.CoInitializeEx(None, 2)
    # Qt may already have initialized COM with a different apartment model.
    if initialized != -2147417850:  # RPC_E_CHANGED_MODE
        checked(initialized)
    store = ctypes.c_void_p()
    try:
        get_store = shell32.SHGetPropertyStoreForWindow
        get_store.argtypes = [wintypes.HWND, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
        get_store.restype = ctypes.c_long
        checked(get_store(hwnd, ctypes.byref(GUID("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")), ctypes.byref(store)))
        vtable = ctypes.cast(store, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        set_value = ctypes.WINFUNCTYPE(
            ctypes.c_long, ctypes.c_void_p,
            ctypes.POINTER(PROPERTYKEY), ctypes.POINTER(PROPVARIANT),
        )(vtable[6])
        executable = Path(sys.executable).resolve()
        if getattr(sys, "frozen", False):
            command = [str(executable)]
        else:
            windowed_python = executable.with_name("pythonw.exe")
            command = [str(windowed_python if windowed_python.is_file() else executable),
                       "-m", "pyml_workbench"]
        values = (
            (2, subprocess.list2cmdline(command)),
            (3, f"{icon_path.resolve()},0"),
            (4, "PY-ML 机器学习工作台"),
            (5, APP_ID),
        )
        for pid, value in values:
            buffer = ctypes.create_unicode_buffer(value)
            variant = PROPVARIANT()
            variant.vt = 31  # VT_LPWSTR; borrowed buffer, not COM-allocated.
            variant.value = ctypes.cast(buffer, ctypes.c_void_p).value
            key = PROPERTYKEY(GUID("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), pid)
            checked(set_value(store, ctypes.byref(key), ctypes.byref(variant)))
        commit = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p)(vtable[7])
        checked(commit(store))
    finally:
        if store.value:
            vtable = ctypes.cast(store, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(vtable[2])(store)
        if initialized >= 0:
            ole32.CoUninitialize()
