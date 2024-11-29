from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aeon_sleap_processing")
except PackageNotFoundError:
    # package is not installed
    pass
