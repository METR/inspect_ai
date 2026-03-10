# Socket Protection Design

## Problem

An agent can crash the sandbox tools server by deleting or replacing the Unix socket at `/tmp/sandbox-tools.sock`. Even with privilege separation (server running as root), the socket sits in `/tmp` where any user can delete files.

## Solution

Move the socket into a protected subdirectory: `/tmp/inspect-sandbox-tools/sandbox-tools.sock`.

When the server runs as root, the directory is owned by `root:root` with mode `0o755`. The agent can enter the directory and connect to the socket (`0o666`), but cannot delete files inside (requires write permission on the directory). Server log files also move into this directory.

When the server is not root, the directory is owned by the current user with mode `0o755`. No real protection (agent is the same user), but the path is consistent.

## Changes

### `constants.py`

Socket path changes from `/tmp/sandbox-tools.sock` to `/tmp/inspect-sandbox-tools/sandbox-tools.sock`.

### `server.py`

Before binding the socket:
1. Create `/tmp/inspect-sandbox-tools/` if it doesn't exist
2. Set ownership and permissions (`root:root`, `0o755` when root; current user, `0o755` otherwise)
3. Remove stale socket (same as today)

Directory creation is idempotent — skip ownership/permission changes if directory already exists with correct owner.

### `main.py`

- Move log file paths into the protected directory
- Catch `OSError` around `SOCKET_PATH.unlink()` in `_can_connect_to_socket()` — unprivileged clients can't delete sockets in a root-owned directory, and shouldn't try to start a new server anyway
