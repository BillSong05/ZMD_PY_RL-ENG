"""MuJoCo import helpers for Windows environments with plugin DLL issues."""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys


_CORE_VERIFIED: bool | None = None


def _is_plugin_init_error(error: BaseException) -> bool:
    return platform.system() == "Windows" and "WinError 1114" in str(error)


def _verify_mujoco_core() -> None:
    global _CORE_VERIFIED
    if _CORE_VERIFIED is True:
        return
    if _CORE_VERIFIED is False:
        raise ImportError("MuJoCo core smoke test failed")

    code = r"""
import importlib
import os
import sys

orig_walk = os.walk

def walk_without_mujoco_plugins(path, *args, **kwargs):
    normalized = str(path).replace("\\", "/")
    if normalized.endswith("/mujoco/plugin"):
        return []
    return orig_walk(path, *args, **kwargs)

os.walk = walk_without_mujoco_plugins
try:
    mujoco = importlib.import_module("mujoco")
finally:
    os.walk = orig_walk

xml = '<mujoco><worldbody><body><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
    )
    _CORE_VERIFIED = result.returncode == 0
    if not _CORE_VERIFIED:
        raise ImportError(
            "MuJoCo imports but cannot compile a minimal model in this Python "
            "environment. Use a clean python.org CPython environment or repair "
            "the Windows runtime/GPU driver stack."
        )


def import_mujoco(verify_core: bool = False):
    """Import MuJoCo, retrying without bundled plugins on Windows if needed.

    Some Windows Python environments can load ``mujoco.dll`` but fail while
    auto-loading optional bundled plugin DLLs. The core simulator does not need
    those plugins for this project, so we retry with the plugin scan disabled.
    """
    if platform.system() != "Windows":
        mujoco = importlib.import_module("mujoco")
        if verify_core:
            _verify_mujoco_core()
        return mujoco

    # On Windows, a failed bundled-plugin load can leave the process in a bad
    # native-DLL state. Skip those optional plugins up front; this project uses
    # MuJoCo core APIs only.
    original_walk = os.walk

    def walk_without_mujoco_plugins(path, *args, **kwargs):
        normalized = str(path).replace("\\", "/")
        if normalized.endswith("/mujoco/plugin"):
            return []
        return original_walk(path, *args, **kwargs)

    try:
        os.walk = walk_without_mujoco_plugins
        mujoco = importlib.import_module("mujoco")
        if verify_core:
            _verify_mujoco_core()
        return mujoco
    finally:
        os.walk = original_walk
