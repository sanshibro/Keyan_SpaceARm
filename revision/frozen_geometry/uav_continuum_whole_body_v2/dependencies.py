"""Dependency loader with support for the repository-local Ubuntu packages."""

import os
import sys
from pathlib import Path


def enable_local_dependencies():
    override = os.environ.get("WHOLE_BODY_V2_DEPS_ROOT", "")
    candidates = [Path(override)] if override else []
    here = Path(__file__).resolve()
    candidates.extend(parent / ".v2_deps" / "root" for parent in here.parents)
    for root in candidates:
        py = root / "usr" / "lib" / "python3" / "dist-packages"
        lib = root / "usr" / "lib" / "x86_64-linux-gnu"
        if py.is_dir() and str(py) not in sys.path:
            sys.path.insert(0, str(py))
        if lib.is_dir():
            old = os.environ.get("LD_LIBRARY_PATH", "")
            if str(lib) not in old.split(":"):
                os.environ["LD_LIBRARY_PATH"] = str(lib) + ((":" + old) if old else "")
        if py.is_dir():
            return str(root)
    return ""


def import_scipy():
    try:
        import scipy  # noqa: F401
    except ImportError:
        enable_local_dependencies()
    import scipy
    return scipy
