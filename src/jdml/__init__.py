from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("JustDataML")  # set from the git tag at build time
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"
