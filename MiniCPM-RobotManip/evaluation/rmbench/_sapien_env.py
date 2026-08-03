"""Runtime fixes applied before the first ``import sapien`` in any process.

SAPIEN 3.0.0b1 only probes ``LD_LIBRARY_PATH`` plus ``/usr/lib{,64,/x86_64-linux-gnu}`` for
``libvulkan.so.1``. On many Linux installs the loader lives under ``/lib/x86_64-linux-gnu`` or
``$CONDA_PREFIX/lib`` instead, which triggers a noisy fallback to SAPIEN's bundled libvulkan
even when a system/conda loader is available.

Also drops a conda-base ``__EGL_VENDOR_LIBRARY_DIRS`` override that points at Mesa-only EGL
vendors and makes SAPIEN fail with "failed to find a rendering device" on NVIDIA hosts.
"""

from __future__ import annotations

import os

_PREPARED = False

# Same filename SAPIEN's ``_vulkan_tricks._ensure_libvulkan`` checks for.
_VULKAN_SONAME = "libvulkan.so.1"

# Paths beyond SAPIEN's built-in probe list (conda prefix + Debian multiarch lib dir).
_EXTRA_LIB_DIRS = (
    "/lib/x86_64-linux-gnu",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
)


def _conda_lib_dir() -> str | None:
    prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if not prefix:
        # ``conda run`` / debugpy often invoke $CONDA_PREFIX/bin/python without activating.
        import sys

        prefix = getattr(sys, "prefix", "")
    if prefix:
        lib = os.path.join(prefix, "lib")
        if os.path.isdir(lib):
            return lib
    return None


def _libvulkan_search_dirs() -> list[str]:
    dirs: list[str] = []
    seen: set[str] = set()

    def _add(path: str) -> None:
        path = path.strip()
        if path and path not in seen:
            seen.add(path)
            dirs.append(path)

    for part in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        _add(part)
    conda_lib = _conda_lib_dir()
    if conda_lib:
        _add(conda_lib)
    for path in _EXTRA_LIB_DIRS:
        _add(path)
    return dirs


def _find_libvulkan_dir() -> str | None:
    for path in _libvulkan_search_dirs():
        if os.path.isfile(os.path.join(path, _VULKAN_SONAME)):
            return path
    return None


def _prepend_ld_library_path(path: str) -> None:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in existing.split(":") if p]
    if path in parts:
        return
    os.environ["LD_LIBRARY_PATH"] = f"{path}:{existing}" if existing else path


def prepare_sapien_runtime() -> None:
    """Idempotent: safe to call from every SAPIEN entry point before ``import sapien``."""
    global _PREPARED
    if _PREPARED:
        return
    _PREPARED = True

    # Conda base activate hook can stack a Mesa-only EGL vendor dir on top of memvla.
    os.environ.pop("__EGL_VENDOR_LIBRARY_DIRS", None)

    vulkan_dir = _find_libvulkan_dir()
    if vulkan_dir is not None:
        _prepend_ld_library_path(vulkan_dir)
