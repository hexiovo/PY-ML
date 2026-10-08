"""Select a Python worker or its sibling frozen executable."""
from __future__ import annotations

from pathlib import Path
import importlib
import sys


def gpu_capability() -> dict[str, object]:
    """Probe optional CUDA support on demand; ordinary imports never load torch."""
    result: dict[str, object] = {
        "available": False,
        "device": "cpu",
        "reason": "未安装 PyTorch，当前使用 CPU。",
        "torch_available": False,
        "cuda_built": False,
        "device_count": 0,
        "device_name": None,
    }
    try:
        torch = importlib.import_module("torch")
        result["torch_available"] = True
        result["cuda_built"] = bool(getattr(torch.version, "cuda", None))
        if not result["cuda_built"]:
            result["reason"] = "当前 PyTorch 为 CPU 版本，无法使用 CUDA GPU。"
        elif not torch.cuda.is_available():
            result["reason"] = "未检测到可用的 CUDA GPU，请检查显卡及驱动。"
        else:
            count = int(torch.cuda.device_count())
            if count > 0:
                result.update(
                    available=True,
                    device="cuda",
                    reason="CUDA GPU 可用（仅 N01 神经网络分类支持）。",
                    device_count=count,
                    device_name=str(torch.cuda.get_device_name(0)),
                )
            else:
                result["reason"] = "未检测到可用的 CUDA GPU。"
    except ImportError:
        pass
    except Exception as exc:
        result["reason"] = f"GPU 检测失败，当前使用 CPU：{exc}"
    return result


def worker_command(arguments: list[str]) -> tuple[str, list[str]]:
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).with_name("PYML-Worker.exe")
        if not executable.is_file():
            raise FileNotFoundError(
                f"后台程序缺失：{executable}。请解压并保留完整的程序文件夹。"
            )
        return str(executable), list(arguments)
    return sys.executable, ["-X", "utf8", "-m", "pyml_workbench._worker", *arguments]
