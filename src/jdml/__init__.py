from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("JustDataML")  # set from the git tag at build time
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = ["__version__", "fit", "Model"]


def __getattr__(name):  # load the heavy ML imports only when the Python API is used
    if name in ("fit", "Model"):
        from . import api
        return getattr(api, name)
    raise AttributeError(f"module 'jdml' has no attribute {name!r}")
