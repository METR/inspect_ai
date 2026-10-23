"""
Sandbox container tool code for inspect_ai.

Contains tools for bash, and editor functionality.
"""

from inspect_sandbox_tools._util.constants import PKG_NAME

__all__ = ["__version__"]


def __getattr__(name: str) -> str:
    # Looked up on first use: importing importlib.metadata is slow, and the CLI
    # forwarding an `exec` request to the server imports this package but never
    # needs the version.
    if name == "__version__":
        from importlib.metadata import version as importlib_version

        return importlib_version(PKG_NAME)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
