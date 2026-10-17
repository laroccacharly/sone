# /// script
# requires-python = ">=3.12"
# ///
"""Fetch the latest llama.cpp into vendor/ and build llama/sone-vision into build/.

Both directories are git-ignored. Re-running pulls llama.cpp's newest commit and rebuilds.
Needs git, cmake and a C++ compiler (ninja is used when available).
"""

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_DIR = ROOT / "vendor" / "llama.cpp"
BUILD_DIR = ROOT / "build"
BINARY = BUILD_DIR / "sone-vision"


def run(*command: str, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def fetch_llama() -> None:
    if not LLAMA_DIR.exists():
        run("git", "clone", "--depth", "1", LLAMA_REPO, str(LLAMA_DIR))
    else:
        run("git", "-C", str(LLAMA_DIR), "fetch", "--depth", "1", "origin", "HEAD")
        run("git", "-C", str(LLAMA_DIR), "reset", "--hard", "FETCH_HEAD")
    run("git", "-C", str(LLAMA_DIR), "log", "--oneline", "-1")


def build_env() -> dict[str, str]:
    # A Homebrew toolchain on PATH (e.g. on the Raspberry Pi) makes the system gcc pick up
    # Homebrew's ld and libgomp, which need a newer glibc than the system's and fail to link.
    path = os.environ.get("PATH", "").split(os.pathsep)
    return {**os.environ, "PATH": os.pathsep.join(p for p in path if "linuxbrew" not in p)}


def build() -> None:
    env = build_env()
    generator = ["-G", "Ninja"] if shutil.which("ninja", path=env["PATH"]) else []
    run(
        "cmake", "-S", str(ROOT / "llama"), "-B", str(BUILD_DIR), *generator,
        "-DCMAKE_BUILD_TYPE=Release", env=env,
    )
    run("cmake", "--build", str(BUILD_DIR), "--target", "sone-vision", "-j", str(os.cpu_count()), env=env)
    print(f"built {BINARY}")


if __name__ == "__main__":
    fetch_llama()
    build()
