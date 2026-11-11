"""Path to the sandbox tools injected into sandbox environments.

The tools ship as a PyInstaller --onedir bundle (a launcher executable plus an
``_internal`` directory), so what is injected is a directory tree rather than a single
file. ``SANDBOX_TOOLS_DIR`` is where the tree is extracted, and ``SANDBOX_CLI`` is the
launcher inside it.

We choose /var/tmp as the injection location since:
  1) it is accessible in all major linux distributions
  2) all users can create entries there, which supports rootless sandboxes
  3) it is unlikely to be cleared during an evaluation
     (https://en.wikipedia.org/wiki/Filesystem_Hierarchy_Standard)
  4) it is unlikely to be accidentally stumbled upon by an LLM solving a
     task that requires interacting with temp files

We additionally choose a dot-prefixed random hash sub-directory to reduce
accidental discovery. When Inspect can run commands in the sandbox as root, it
installs the tree as root and restricts it to 0700. A root-owned 0700 tree
prevents access by other, non-root users, but not by a process running in the
sandbox as root.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .environment import SandboxEnvironment

# Also defined in inspect_ai.tool._sandbox_tools_utils._build_config — keep in sync.
SANDBOX_TOOLS_BASE_NAME = "inspect-sandbox-tools"

SANDBOX_TOOLS_DIR = "/var/tmp/.da7be258e003d428"

SANDBOX_CLI = f"{SANDBOX_TOOLS_DIR}/{SANDBOX_TOOLS_BASE_NAME}"

TOOLS_DIR_REFUSED = "INSPECT_SANDBOX_TOOLS_DIRECTORY_REFUSED"
"""All the check prints to stdout when it refuses the launch; why goes to stderr."""

# Runs the command after `$1` (the parent of SANDBOX_TOOLS_DIR) and `$2` (its name)
# only if the directory is still a real one, owned by the user running this, mode
# 0700 (and owned by uid 0 when root runs it, see `tools_cli_command`). In a
# subshell, with shell builtins and `stat` (two program starts), so the command
# keeps the caller's PATH and working directory. The final `exec`
# looks the path up again, which is safe once the check passed: only the owner can
# move that directory out of its parent, which the full check required to be
# sticky or closed to others. A refusal exits 0 because k8s_sandbox
# raises, hiding the output, when an exec fails right after a container restart,
# which is when refusals happen. A check that cannot run (`stat`, a fork or the
# path's `printf` failing, as under a full process table) is no refusal: it exits
# non-zero (125, or the shell's own status), runs nothing, and fails as an exec does.
#
# The subshell prints the launcher's physical path, from which the launch sets the
# environment the PyInstaller bootloader would set and then re-exec itself to apply
# (the onedir restart in pyi_main.c of 6.22.3, which built the bundled CLI), so the
# CLI starts as one program, not two. `_PYI_ARCHIVE_FILE` has to be the launcher's
# real path, as /proc/self/exe resolves it, or the bootloader restarts anyway with
# the bundle's library path saved as the caller's. An inherited
# `PYINSTALLER_RESET_ENVIRONMENT` would restart it too, and an inherited
# `_PYI_LINUX_PROCESS_NAME` rename it. The launch sets no other variable, and
# `_cli/main.py` drops these for every child it starts. The assignment's status is
# its command substitution's (POSIX), so `case` reads the subshell's exit code. bash
# expands a substitution it cannot start (no pipe to read it from) to nothing, with
# status 0: so a pass also needs the path, and an empty `stat` is a failure too.
# `test_a_launch_starts_the_bundled_cli_once` checks that the bundled glibc amd64
# bootloader still takes them; the other three bundles share its release. The
# builder Dockerfiles don't pin PyInstaller yet: they are inputs to the bundles'
# source digest, so the next tools release pins `pyinstaller==6.22.3` in both.
_TOOLS_DIR_CHECK = (
    '_PYI_ARCHIVE_FILE=$(PATH=@PATH@; cd -P -- "$1" && p=${PWD%/} && cd -P -- "$p/$2"'
    ' || exit 1; s=$(stat -c @FORMAT@ .) && [ -n "$s" ] || exit 2;'
    ' [ "$PWD" = "$p/$2" ] && [ -O . ] && [ "$s" = @WANT@ ] || exit 1;'
    f' printf %s "$PWD/{SANDBOX_TOOLS_BASE_NAME}" || exit 2)'
    f"; case $?$_PYI_ARCHIVE_FILE in 0/*) ;; 1) echo {TOOLS_DIR_REFUSED}; exit 0;;"
    " *) exit 125;; esac;"
    " shift 2; unset PYINSTALLER_RESET_ENVIRONMENT _PYI_LINUX_PROCESS_NAME;"
    ' [ -z "$LD_LIBRARY_PATH" ] || export LD_LIBRARY_PATH_ORIG="$LD_LIBRARY_PATH";'
    " export _PYI_ARCHIVE_FILE _PYI_PARENT_PROCESS_LEVEL=-1"
    ' LD_LIBRARY_PATH="${_PYI_ARCHIVE_FILE%/*}/_internal'
    '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; exec "$@"'
)


def tools_cli_command(*args: str, user: str | None) -> list[str]:
    """Command that runs the sandbox tools CLI with ``args`` after a check.

    The full install check (``_sandbox_tools_installed``) runs once per sandbox.
    Of what it verified, only the directory itself can change for the worse: a tree
    that disappears (with a restarted container) can be re-created at the same path
    by anyone who can write to its parent. So each launch checks, in the shell
    that then runs the CLI, that the directory is still a real one owned by the
    tools user with mode 0700. Only that user or root can create such a directory
    or write into it, and nobody else can move one into place: moving a directory
    into the parent needs write access to it, and within the parent, which the full
    check required to be sticky or closed to others, only the entry's owner can
    rename it. If the directory fails the check, the command prints only
    ``TOOLS_DIR_REFUSED``, and ``SandboxJSONRPCTransport`` fails the call with an
    error naming a replaced sandbox. Nothing installs the tools again on that sandbox
    object, so later calls fail the same way while the directory stays missing. A
    check that cannot run exits non-zero instead, running nothing.

    ``user`` is the user the command runs as. For ``"root"`` the directory must also
    be owned by uid 0, as the full check required, so a provider that stops running
    root's commands as uid 0 cannot launch a tree owned by the uid it uses instead.
    """
    from ._privileged import SHELL_PATH, SYSTEM_PATH

    parent, _, name = SANDBOX_TOOLS_DIR.rpartition("/")
    stat_format, want = ("%u:%a", "0:700") if user == "root" else ("%a", "700")
    check = (
        _TOOLS_DIR_CHECK.replace("@PATH@", SYSTEM_PATH)
        .replace("@FORMAT@", stat_format)
        .replace("@WANT@", want)
    )
    return [SHELL_PATH, "-c", check, "sh", parent, name, SANDBOX_CLI, *args]


def is_tools_cli_command(cmd: list[str]) -> bool:
    """Whether ``cmd`` runs the sandbox tools CLI, directly or after the check."""
    return cmd[:1] == [SANDBOX_CLI] or any(
        cmd[:3] == tools_cli_command(user=user)[:3] for user in ("root", None)
    )


def tools_user_param(
    sandbox: "SandboxEnvironment", user: str | None
) -> str | dict[str, object] | None:
    """Value of a sandbox tool request's ``user`` param.

    An explicit ``user`` wins; otherwise the sandbox's default exec identity, which
    the tools switch to (a no-op when they already run as it). None when neither is
    known, so the tools keep their own identity.
    """
    if user:
        return user
    if sandbox._tools_default_user:
        return sandbox._tools_default_user._asdict()
    return None
