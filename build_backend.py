"""scikit-build-core, with Homebrew's Linux toolchain dropped from PATH first.

A Homebrew toolchain on PATH (e.g. on the Raspberry Pi) makes the system gcc pick up
Homebrew's ld and libgomp, which need a newer glibc than the system's and fail to link.
"""

import os

path = os.environ.get("PATH", "").split(os.pathsep)
os.environ["PATH"] = os.pathsep.join(p for p in path if "linuxbrew" not in p)

from scikit_build_core.build import (  # noqa: E402
    build_editable,
    build_sdist,
    build_wheel,
    get_requires_for_build_editable,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_editable,
    prepare_metadata_for_build_wheel,
)

__all__ = [
    "build_editable",
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_editable",
    "prepare_metadata_for_build_wheel",
]
