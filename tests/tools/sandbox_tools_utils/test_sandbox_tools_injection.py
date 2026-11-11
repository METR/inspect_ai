"""Tests for sandbox tools injection."""

import contextlib
import json
import os
import platform
import shutil
import subprocess
import sys
import warnings
from contextlib import asynccontextmanager
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import AsyncIterator, BinaryIO, Iterator

import anyio
import pytest
from test_helpers.sandbox import (
    ROOT_AMBIGUOUS,
    ROOT_UNUSABLE,
    ROOT_USABLE,
    CannedSandbox,
    ExecPolicy,
    FrameworkDirectoryCall,
    framework_directory_call,
    is_root_probe,
    root_probe_result,
)

from inspect_ai import Task, eval
from inspect_ai._util import logger as inspect_logger
from inspect_ai.dataset import Sample
from inspect_ai.event import LoggerEvent
from inspect_ai.event._sandbox import SandboxEvent
from inspect_ai.log._transcript import Transcript, init_transcript
from inspect_ai.model._call_tools import tool_call_error
from inspect_ai.solver import generate
from inspect_ai.tool._sandbox_tools_utils import sandbox as sandbox_tools
from inspect_ai.util._sandbox import _cli as cli_module
from inspect_ai.util._sandbox._cli import (
    SANDBOX_CLI,
    SANDBOX_TOOLS_DIR,
    TOOLS_DIR_REFUSED,
    is_tools_cli_command,
    tools_cli_command,
)
from inspect_ai.util._sandbox._framework_directory import (
    _MISSING_MARKER,
    _STAT_ENTRY,
    _UNAVAILABLE_MARKER,
    _USER_MISMATCH_MARKER,
    _VERIFIED_MARKER,
    _VIOLATION_MARKER,
    FrameworkDirectoryError,
)
from inspect_ai.util._sandbox._json_rpc_transport import SandboxJSONRPCTransport
from inspect_ai.util._sandbox._privileged import SHELL_PATH, SYSTEM_PATH
from inspect_ai.util._sandbox.context import init_sandbox_environments_sample
from inspect_ai.util._sandbox.environment import (
    RootAccess,
    SandboxDefaultUser,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    SandboxUnavailableError,
    SandboxUserUnsupportedError,
)
from inspect_ai.util._sandbox.events import (
    SandboxEnvironmentProxy,
    SandboxTimeoutError,
)
from inspect_ai.util._sandbox.local import LocalSandboxEnvironment
from inspect_ai.util._sandbox.recon import Architecture, SupportedContainerOSInfo
from inspect_ai.util._subprocess import ExecResult

OK = ExecResult(success=True, returncode=0, stdout="", stderr="")
"""Result of an ordinary (non-helper) command."""

VERIFIED = ExecResult(
    success=True, returncode=0, stdout="", stderr=f"{_VERIFIED_MARKER}\n"
)
"""Helper result: verification passed and the wrapped command (if any) succeeded."""

REGULAR_FILE = ExecResult(
    success=True,
    returncode=0,
    stdout="81ed\n",  # stat -c %f of a 0755 regular file
    stderr=f"{_VERIFIED_MARKER}\n",
)
"""Helper result for the detector: verified, and the launcher is a regular file."""


def violation(message: str) -> ExecResult[str]:
    return ExecResult(
        success=False,
        returncode=3,
        stdout="",
        stderr=f"{_VIOLATION_MARKER}: {message}\n",
    )


MISSING = ExecResult(
    success=False,
    returncode=4,
    stdout="",
    stderr=f"{_MISSING_MARKER}: {SANDBOX_TOOLS_DIR} does not exist\n",
)
UNAVAILABLE = ExecResult(
    success=False,
    returncode=5,
    stdout="",
    stderr=f"{_UNAVAILABLE_MARKER}: cannot stat parent directory /var/tmp: stat: not found\n",
)
NO_ROOT = ExecResult(
    success=False,
    returncode=126,
    stdout="",
    stderr="unable to find user root: no matching entries in passwd file\n",
)
"""A provider that reports an unusable user through the exit status."""
NOT_ROOT = ExecResult(
    success=False,
    returncode=6,
    stdout="",
    stderr=f"{_USER_MISMATCH_MARKER}: running as uid 1000, expected uid 0\n",
)
"""A provider that accepted user="root" but ran the helper as someone else."""


def _tools_dir_call(cmd: list[str]) -> FrameworkDirectoryCall | None:
    call = framework_directory_call(cmd)
    return call if call is not None and call.path == SANDBOX_TOOLS_DIR else None


def is_framework_dir_call(cmd: list[str]) -> bool:
    return _tools_dir_call(cmd) is not None


def wrapped_command(cmd: list[str]) -> list[str]:
    """The command a framework-directory call execs after verification."""
    call = _tools_dir_call(cmd)
    assert call is not None
    return list(call.cmd)


def helper_flags(cmd: list[str]) -> FrameworkDirectoryCall:
    """The fixed arguments a framework-directory call passes ahead of the path."""
    call = _tools_dir_call(cmd)
    assert call is not None
    return call


DEFAULT_USER = ExecResult(
    success=True,
    returncode=0,
    stdout="Uid: 1111\t1111\t1111\t1111\nGid: 1111\t1111\t1111\t1111\nGroups: 1111 \nHOME: /home/nonroot\nHOME_SET: 1\n",
    stderr="",
)
"""Result of the default-user identity probe (compose `user: nonroot`)."""
NONROOT = SandboxDefaultUser(uid=1111, gid=1111, groups=[1111], home="/home/nonroot")


def is_identity_probe(cmd: list[str]) -> bool:
    return cmd[:2] == ["/bin/sh", "-c"] and "Groups:" in cmd[2]


TAR_XZF_STDIN = "tar xzf - || { cat >/dev/null; exit 1; }"
TAR_XF_STDIN = "tar xf - || { cat >/dev/null; exit 1; }"


def helper_ok(cmd: list[str], user: str | None) -> ExecResult[str]:
    """Every helper call verifies; every other command succeeds (root is usable)."""
    if is_framework_dir_call(cmd):
        return VERIFIED
    if is_identity_probe(cmd):
        return DEFAULT_USER
    if is_root_probe(cmd):
        return root_probe_result()
    return OK


@pytest.fixture
def _warn_once_messages() -> Iterator[list[str]]:
    # warn_once dedupes via a module-level list; clear it and yield it so the test
    # can assert on what was emitted (caplog is unreliable once an earlier test has
    # set propagate=False on the inspect_ai logger).
    inspect_logger._warned.clear()
    yield inspect_logger._warned
    inspect_logger._warned.clear()


def root_access_warned(messages: list[str]) -> bool:
    return sandbox_tools._AMBIGUOUS_ROOT_ACCESS_WARNING in messages


@pytest.fixture
def stub_artifact(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Stub OS detection and artifact lookup; record the user extraction ran as."""
    recorded: dict[str, object] = {"extracted_as": None, "extracted": False}

    async def fake_detect_sandbox_os(
        _sandbox: SandboxEnvironment,
    ) -> SupportedContainerOSInfo:
        return {"architecture": "amd64", "libc": "glibc"}

    @asynccontextmanager
    async def fake_open_executable_for_arch(
        _arch: Architecture,
        _musl: bool,
    ) -> AsyncIterator[tuple[str, BinaryIO]]:
        yield "inspect-sandbox-tools", BytesIO(b"binary")

    async def fake_extract_tools_tree(
        _sandbox: SandboxEnvironment,
        _name: str,
        _gz_bytes: bytes,
        user: str | None,
    ) -> None:
        recorded["extracted"] = True
        recorded["extracted_as"] = user

    monkeypatch.setattr(sandbox_tools, "detect_sandbox_os", fake_detect_sandbox_os)
    monkeypatch.setattr(
        sandbox_tools, "_open_executable_for_arch", fake_open_executable_for_arch
    )
    monkeypatch.setattr(sandbox_tools, "_extract_tools_tree", fake_extract_tools_tree)
    return recorded


@pytest.mark.parametrize(
    "root_failure",
    [
        pytest.param(
            RuntimeError("runuser: may not be used by non-root users"),
            id="provider-raises",
        ),
        pytest.param(NO_ROOT, id="provider-fails-with-status"),
    ],
)
async def test_inject_falls_back_with_warning_when_root_probe_gives_no_verdict(
    stub_artifact: dict[str, object],
    _warn_once_messages: list[str],
    root_failure: Exception | ExecResult[str],
) -> None:
    """With no recorded decision the probe runs on first use; no verdict warns.

    A probe with no verdict still selects the default user, but with a warning.
    """

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if user == "root":
            if isinstance(root_failure, Exception):
                raise root_failure
            return root_failure
        return helper_ok(cmd, user)

    sandbox = CannedSandbox(policy)
    await sandbox_tools._inject_container_tools_code(sandbox)

    assert sandbox._root_access is not None
    assert sandbox._root_access.state == "ambiguous"
    assert sandbox._tools_user is None
    assert sandbox._tools_default_user is None
    assert stub_artifact["extracted_as"] is None
    # Root was probed exactly once, and never with a bare mkdir.
    root_calls = [cmd for cmd, user in sandbox.exec_calls if user == "root"]
    assert len(root_calls) == 1 and is_root_probe(root_calls[0])
    assert (tools_cli_command("start-server", user=None), None) in sandbox.exec_calls
    assert root_access_warned(_warn_once_messages)


async def test_detector_skips_root_probe_after_rootless_injection(
    stub_artifact: dict[str, object],
) -> None:
    """Once a rootless install has run, later tool calls do not re-probe root."""

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if user == "root":
            raise RuntimeError("runuser: may not be used by non-root users")
        return REGULAR_FILE if is_framework_dir_call(cmd) else OK

    sandbox = CannedSandbox(policy)
    await sandbox_tools._inject_container_tools_code(sandbox)
    assert sandbox._tools_user is None

    sandbox.exec_calls.clear()
    assert await sandbox_tools._sandbox_tools_installed(sandbox) is True
    assert [user for _, user in sandbox.exec_calls] == [None]


async def test_root_access_is_probed_once_on_first_use_without_sample_init() -> None:
    """With no recorded decision the first detection probes, records, and adopts."""

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if is_identity_probe(cmd):
            return DEFAULT_USER
        return REGULAR_FILE if is_framework_dir_call(cmd) else root_probe_result()

    sandbox = CannedSandbox(policy)
    assert sandbox._root_access is None
    assert await sandbox_tools._sandbox_tools_installed(sandbox) is True
    assert sandbox._root_access is not None
    assert sandbox._root_access.state == "usable"
    assert sandbox._tools_user == "root"
    assert sum(is_root_probe(cmd) for cmd, _ in sandbox.exec_calls) == 1

    assert await sandbox_tools._sandbox_tools_installed(sandbox) is True
    assert sum(is_root_probe(cmd) for cmd, _ in sandbox.exec_calls) == 1


@pytest.mark.parametrize(
    "root_failure, expected",
    [
        pytest.param(
            SandboxUserUnsupportedError("cannot execute as root"),
            "cannot execute as root",
            id="unsupported-user",
        ),
        pytest.param(
            RuntimeError("docker exec: transient failure"),
            "transient failure",
            id="provider-raises",
        ),
        pytest.param(
            NO_ROOT, "no matching entries in passwd", id="provider-fails-with-status"
        ),
        pytest.param(
            NOT_ROOT, "did not run as the requested user", id="provider-runs-other-uid"
        ),
    ],
)
async def test_inject_errors_on_root_failure_after_usable_verdict(
    stub_artifact: dict[str, object],
    _warn_once_messages: list[str],
    root_failure: Exception | ExecResult[str],
    expected: str,
) -> None:
    """Once root is known usable, a root failure is an error, never a rootless install."""

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if user == "root":
            if isinstance(root_failure, Exception):
                raise root_failure
            return root_failure
        return helper_ok(cmd, user)

    sandbox = CannedSandbox(policy)
    sandbox._root_access = ROOT_USABLE
    with pytest.raises(sandbox_tools.SandboxInjectionError, match=expected):
        await sandbox_tools._inject_container_tools_code(sandbox)

    assert sandbox._tools_user is None
    assert sandbox._tools_user_resolved is False
    assert stub_artifact["extracted"] is False
    assert not any(
        user is None and is_framework_dir_call(cmd) for cmd, user in sandbox.exec_calls
    )
    assert not any(is_tools_cli_command(cmd) for cmd, _ in sandbox.exec_calls)
    assert _warn_once_messages == []


async def test_inject_installs_rootless_quietly_when_root_unusable(
    stub_artifact: dict[str, object], _warn_once_messages: list[str]
) -> None:
    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        assert user is None, "no root exec at all once root is known unusable"
        return helper_ok(cmd, user)

    sandbox = CannedSandbox(policy)
    sandbox._root_access = ROOT_UNUSABLE
    await sandbox_tools._inject_container_tools_code(sandbox)

    assert sandbox._tools_user is None
    assert sandbox._tools_user_resolved is True
    assert stub_artifact["extracted_as"] is None
    assert (tools_cli_command("start-server", user=None), None) in sandbox.exec_calls
    assert not any(is_root_probe(cmd) for cmd, _ in sandbox.exec_calls)
    assert _warn_once_messages == []


async def test_inject_warns_once_per_process_when_root_access_ambiguous(
    stub_artifact: dict[str, object], _warn_once_messages: list[str]
) -> None:
    """An ambiguous decision installs as the default user, warned once, not per sandbox."""

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        assert user is None, "an ambiguous decision must not probe root again"
        return helper_ok(cmd, user)

    for _ in range(2):
        sandbox = CannedSandbox(policy)
        sandbox._root_access = ROOT_AMBIGUOUS
        await sandbox_tools._inject_container_tools_code(sandbox)
        assert sandbox._tools_user is None
        assert sandbox._tools_user_resolved is True
        assert (
            tools_cli_command("start-server", user=None),
            None,
        ) in sandbox.exec_calls

    assert _warn_once_messages == [sandbox_tools._AMBIGUOUS_ROOT_ACCESS_WARNING]


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            SandboxUnavailableError("container is not running"), id="unavailable"
        ),
        pytest.param(TimeoutError("root probe timed out"), id="timeout"),
    ],
)
async def test_tools_surface_a_failed_probe_instead_of_falling_back(
    stub_artifact: dict[str, object], _warn_once_messages: list[str], error: Exception
) -> None:
    """A probe that could not run decided nothing: the tools refuse to pick a user."""
    sandbox = CannedSandbox(helper_ok)
    sandbox._root_access = RootAccess(
        "failed", f"root probe did not complete: {error}", error
    )

    assert await sandbox_tools._sandbox_tools_installed(sandbox) is False
    with pytest.raises(
        sandbox_tools.SandboxInjectionError, match="root probe did not complete"
    ) as excinfo:
        await sandbox_tools._inject_container_tools_code(sandbox)

    assert str(error) in str(excinfo.value)
    assert excinfo.value.cause is error and excinfo.value.__cause__ is error
    assert sandbox.exec_calls == []
    assert sandbox._tools_user is None
    assert sandbox._tools_user_resolved is False
    assert stub_artifact["extracted"] is False
    assert _warn_once_messages == []


async def test_exec_remote_surfaces_failed_probe_without_marking_tools_injected() -> (
    None
):
    sandbox = CannedSandbox(helper_ok)
    sandbox._root_access = RootAccess(
        "failed", "root probe did not complete", SandboxUnavailableError("gone")
    )
    with pytest.raises(sandbox_tools.SandboxInjectionError):
        await sandbox.exec_remote(["true"], stream=False)
    assert sandbox._tools_injected is False
    assert sandbox.exec_calls == []


async def test_inject_uses_root_and_verifies_before_start(
    stub_artifact: dict[str, object],
) -> None:
    sandbox = CannedSandbox(helper_ok)
    await sandbox_tools._inject_container_tools_code(sandbox)

    assert sandbox._tools_user == "root"
    assert sandbox._tools_default_user == NONROOT
    assert [u for cmd, u in sandbox.exec_calls if is_identity_probe(cmd)] == [None]
    assert stub_artifact["extracted_as"] == "root"
    # Verified before extraction and again immediately before the launcher starts,
    # the latter without creating anything.
    start = sandbox.exec_calls.index(
        (tools_cli_command("start-server", user="root"), "root")
    )
    verifications = [
        i
        for i, (cmd, user) in enumerate(sandbox.exec_calls)
        if is_framework_dir_call(cmd) and user == "root"
    ]
    assert len(verifications) >= 2
    assert verifications[-1] == start - 1
    assert not helper_flags(sandbox.exec_calls[start - 1][0]).create
    # Every root-side check insists the script really ran as uid 0, and none asks
    # for a wrong-mode root-owned directory to be repaired.
    root_flags = [
        helper_flags(cmd)
        for cmd, user in sandbox.exec_calls
        if is_framework_dir_call(cmd) and user == "root"
    ]
    assert all(flags.expected_uid == "0" for flags in root_flags)
    assert not any(flags.repair for flags in root_flags)
    # The tools tree stays private to the tools user.
    assert all(flags.mode == "700" for flags in root_flags)
    # No path-based chmod: the directory is created 0700 and verified, not repaired.
    assert not any(cmd[:1] == ["chmod"] for cmd, _ in sandbox.exec_calls)


async def test_inject_fails_when_the_server_start_is_refused(
    stub_artifact: dict[str, object],
) -> None:
    """The tree verified a moment before changed before the server started from it."""

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if cmd == tools_cli_command("start-server", user="root"):
            return REFUSED
        return helper_ok(cmd, user)

    with pytest.raises(sandbox_tools.SandboxInjectionError, match="Failed to start"):
        await sandbox_tools._inject_container_tools_code(CannedSandbox(policy))


async def test_inject_falls_back_when_provider_runs_root_as_default_user(
    stub_artifact: dict[str, object], _warn_once_messages: list[str]
) -> None:
    """A provider that ignores `user` must yield a rootless install, not a fake root one.

    The probe itself runs as the default user then, which is a definitive verdict:
    no warning, and root is never asked for again.
    """

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if is_root_probe(cmd):
            return root_probe_result(uid="1000")
        assert user is None, "root must not be used once the probe ran as another uid"
        return helper_ok(cmd, user)

    sandbox = CannedSandbox(policy)
    await sandbox_tools._inject_container_tools_code(sandbox)

    assert sandbox._root_access is not None
    assert sandbox._root_access.state == "unusable"
    assert sandbox._tools_user is None
    assert sandbox._tools_user_resolved is True
    assert stub_artifact["extracted_as"] is None
    assert (tools_cli_command("start-server", user=None), None) in sandbox.exec_calls
    assert _warn_once_messages == []
    # Default-user checks carry no uid expectation (the host cannot know it). Only
    # the install step repairs a wrong-mode directory the default user owns; the
    # detector and the pre-launch re-check never do.
    default_flags = [
        helper_flags(cmd)
        for cmd, user in sandbox.exec_calls
        if is_framework_dir_call(cmd) and user is None
    ]
    assert all(flags.expected_uid == "" for flags in default_flags)
    assert [flags.repair for flags in default_flags if flags.create] == [True]
    assert not any(flags.repair for flags in default_flags if not flags.create)


@pytest.mark.parametrize(
    "result, expected",
    [
        pytest.param(
            violation(f"{SANDBOX_TOOLS_DIR} is owned by uid 1111, expected uid 0"),
            "owned by uid 1111, expected uid 0",
            id="planted-directory",
        ),
        pytest.param(UNAVAILABLE, "stat: not found", id="cannot-verify"),
    ],
)
async def test_inject_aborts_on_root_verdict_without_downgrading(
    stub_artifact: dict[str, object], result: ExecResult[str], expected: str
) -> None:
    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if user == "root" and is_framework_dir_call(cmd):
            return result
        return helper_ok(cmd, user)

    sandbox = CannedSandbox(policy)
    with pytest.raises(sandbox_tools.SandboxInjectionError) as excinfo:
        await sandbox_tools._inject_container_tools_code(sandbox)

    assert expected in str(excinfo.value)
    assert sandbox._tools_user is None
    assert stub_artifact["extracted"] is False
    # Never tried the default user, never started a launcher.
    assert all(user == "root" for _, user in sandbox.exec_calls)
    assert not any(is_tools_cli_command(cmd) for cmd, _ in sandbox.exec_calls)


async def test_inject_aborts_on_rootless_contract_violation(
    stub_artifact: dict[str, object],
) -> None:
    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if is_framework_dir_call(cmd):
            return violation(f"{SANDBOX_TOOLS_DIR} is a symbolic link")
        return OK

    sandbox = CannedSandbox(policy)
    sandbox._root_access = ROOT_UNUSABLE
    with pytest.raises(sandbox_tools.SandboxInjectionError, match="symbolic link"):
        await sandbox_tools._inject_container_tools_code(sandbox)

    assert stub_artifact["extracted"] is False
    assert not any(is_tools_cli_command(cmd) for cmd, _ in sandbox.exec_calls)


@pytest.mark.parametrize(
    "result, expected",
    [
        pytest.param(
            violation(f"{SANDBOX_TOOLS_DIR} is a symbolic link"),
            "symbolic link",
            id="swapped-for-symlink",
        ),
        pytest.param(MISSING, "does not exist", id="removed"),
    ],
)
async def test_inject_aborts_when_reverification_before_start_fails(
    stub_artifact: dict[str, object], result: ExecResult[str], expected: str
) -> None:
    calls = {"n": 0}

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if is_framework_dir_call(cmd):
            calls["n"] += 1
            if calls["n"] == 2:  # between extraction and launch
                return result
            return VERIFIED
        return helper_ok(cmd, user)

    sandbox = CannedSandbox(policy)
    with pytest.raises(sandbox_tools.SandboxInjectionError, match=expected):
        await sandbox_tools._inject_container_tools_code(sandbox)

    assert stub_artifact["extracted"] is True
    assert not any(is_tools_cli_command(cmd) for cmd, _ in sandbox.exec_calls)


async def test_inject_names_a_message_less_exception(
    stub_artifact: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken stdin write raises with no message; the error still says what happened."""

    async def broken_extract(*_args: object) -> None:
        raise anyio.BrokenResourceError()

    monkeypatch.setattr(sandbox_tools, "_extract_tools_tree", broken_extract)
    with pytest.raises(
        sandbox_tools.SandboxInjectionError, match="BrokenResourceError"
    ):
        await sandbox_tools._inject_container_tools_code(CannedSandbox(helper_ok))


async def test_extract_streams_archive_to_tar_inside_verified_directory() -> None:
    sandbox = CannedSandbox(helper_ok)
    await sandbox_tools._extract_tools_tree(sandbox, "name", b"gz", "root")

    [(tar_cmd, user)] = sandbox.exec_calls
    assert user == "root"
    assert wrapped_command(tar_cmd) == ["sh", "-c", TAR_XZF_STDIN]
    assert sandbox.inputs == [b"gz"]
    assert sandbox.written == []


async def test_extract_falls_back_to_plain_tar_inside_verified_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sandbox_tools, "_uncompressed_tar_bytes", lambda name, gz: b"tar"
    )

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if is_framework_dir_call(cmd) and TAR_XZF_STDIN in cmd:
            return ExecResult(
                success=False,
                returncode=2,
                stdout="",
                stderr=f"{_VERIFIED_MARKER}\ntar: gzip: not found\n",
            )
        return helper_ok(cmd, user)

    sandbox = CannedSandbox(policy)
    await sandbox_tools._extract_tools_tree(sandbox, "name", b"gz", None)

    assert [wrapped_command(cmd) for cmd, _ in sandbox.exec_calls] == [
        ["sh", "-c", TAR_XZF_STDIN],
        ["sh", "-c", TAR_XF_STDIN],
    ]
    assert sandbox.inputs == [b"gz", b"tar"]
    assert sandbox.written == []


async def test_extract_propagates_helper_verdict() -> None:
    sandbox = CannedSandbox(
        lambda cmd, user: violation(f"{SANDBOX_TOOLS_DIR} is a symbolic link")
    )
    with pytest.raises(FrameworkDirectoryError, match="is a symbolic link"):
        await sandbox_tools._extract_tools_tree(sandbox, "name", b"gz", "root")
    assert len(sandbox.exec_calls) == 1
    assert sandbox.written == []


async def test_extract_reports_failure_of_both_tar_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sandbox_tools, "_uncompressed_tar_bytes", lambda name, gz: b"tar"
    )
    sandbox = CannedSandbox.returning(
        ExecResult(
            success=False,
            returncode=2,
            stdout="",
            stderr=f"{_VERIFIED_MARKER}\ntar: short read\n",
        )
    )
    with pytest.raises(RuntimeError, match="Failed to extract sandbox tools"):
        await sandbox_tools._extract_tools_tree(sandbox, "name", b"gz", "root")
    assert len(sandbox.exec_calls) == 2


async def test_detector_checks_as_known_tools_user() -> None:
    sandbox = CannedSandbox(lambda cmd, user: REGULAR_FILE)
    sandbox._tools_user = "root"
    assert await sandbox_tools._sandbox_tools_installed(sandbox) is True
    # Checked as the tools user, inside the verified directory, by relative name.
    [(cmd, user)] = sandbox.exec_calls
    assert user == "root"
    assert wrapped_command(cmd) == [
        "sh",
        "-c",
        _STAT_ENTRY,
        "sh",
        "inspect-sandbox-tools",
    ]


@pytest.mark.parametrize(
    "stdout, expected",
    [
        pytest.param("81ed\n", True, id="regular-0755"),
        pytest.param("8180\n", True, id="regular-0600"),
        pytest.param("a1ff\n", False, id="symlink"),
        pytest.param("41ed\n", False, id="directory"),
        pytest.param("missing\n", False, id="missing"),
    ],
)
async def test_detector_reads_launcher_type_from_raw_mode(
    stdout: str, expected: bool
) -> None:
    """Only a regular file at the launcher name counts as installed."""
    sandbox = CannedSandbox(
        lambda cmd, user: ExecResult(
            success=True, returncode=0, stdout=stdout, stderr=f"{_VERIFIED_MARKER}\n"
        )
    )
    sandbox._tools_user = "root"
    assert await sandbox_tools._sandbox_tools_installed(sandbox) is expected


async def test_detector_adopts_existing_root_installation() -> None:
    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if is_identity_probe(cmd):
            return DEFAULT_USER
        assert user == "root", "default user must not be consulted when root works"
        return REGULAR_FILE

    sandbox = CannedSandbox(policy)
    sandbox._root_access = ROOT_USABLE
    assert await sandbox_tools._sandbox_tools_installed(sandbox) is True
    assert sandbox._tools_user == "root"
    assert sandbox._tools_default_user == NONROOT


async def test_detector_adopts_default_user_install_when_root_unusable(
    _warn_once_messages: list[str],
) -> None:
    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        assert user is None, "root must never be consulted once known unusable"
        return REGULAR_FILE

    sandbox = CannedSandbox(policy)
    sandbox._root_access = ROOT_UNUSABLE
    assert await sandbox_tools._sandbox_tools_installed(sandbox) is True
    assert sandbox._tools_user is None
    assert sandbox._tools_user_resolved is True
    assert [user for _, user in sandbox.exec_calls] == [None]

    # The adopted rootless install is remembered.
    assert await sandbox_tools._sandbox_tools_installed(sandbox) is True
    assert [user for _, user in sandbox.exec_calls] == [None]
    assert _warn_once_messages == []


async def test_detector_adopts_default_user_install_with_warning_when_ambiguous(
    _warn_once_messages: list[str],
) -> None:
    """An ambiguous decision is applied like "unusable", plus the warning.

    Root is not probed again either.
    """
    sandbox = CannedSandbox(lambda cmd, user: REGULAR_FILE)
    sandbox._root_access = ROOT_AMBIGUOUS
    assert await sandbox_tools._sandbox_tools_installed(sandbox) is True
    assert sandbox._tools_user is None
    assert sandbox._tools_user_resolved is True

    assert await sandbox_tools._sandbox_tools_installed(sandbox) is True
    assert [user for _, user in sandbox.exec_calls] == [None]
    assert _warn_once_messages == [sandbox_tools._AMBIGUOUS_ROOT_ACCESS_WARNING]


async def test_detector_does_not_warn_for_an_ambiguous_sandbox_it_only_visits(
    _warn_once_messages: list[str],
) -> None:
    """Detection alone records no user, so a sandbox with no install is not warned."""
    sandbox = CannedSandbox(lambda cmd, user: MISSING)
    sandbox._root_access = ROOT_AMBIGUOUS
    assert await sandbox_tools._sandbox_tools_installed(sandbox) is False
    assert sandbox._tools_user_resolved is False
    assert _warn_once_messages == []


async def test_root_failure_after_usable_verdict_never_consults_default_user(
    stub_artifact: dict[str, object],
) -> None:
    """A planted tree is never looked at once root is known usable.

    Root-capable sandbox: the agent plants a 0700 tree under its own uid and root
    then fails transiently. Detection reports "not installed" without consulting
    the default user's view, and the injection that follows fails loud on the
    planted tree instead of installing as that uid.
    """
    calls = {"root": 0}

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if user == "root":
            calls["root"] += 1
            if calls["root"] == 1:
                raise RuntimeError("docker exec: transient failure")
            return violation(
                f"{SANDBOX_TOOLS_DIR} is owned by uid 1111, expected uid 0"
            )
        return REGULAR_FILE if is_framework_dir_call(cmd) else helper_ok(cmd, user)

    sandbox = CannedSandbox(policy)
    sandbox._root_access = ROOT_USABLE
    assert await sandbox_tools._sandbox_tools_installed(sandbox) is False
    assert [user for _, user in sandbox.exec_calls] == ["root"]

    with pytest.raises(
        sandbox_tools.SandboxInjectionError, match="owned by uid 1111, expected uid 0"
    ):
        await sandbox_tools._inject_container_tools_code(sandbox)
    assert stub_artifact["extracted"] is False
    assert sandbox._tools_user is None
    assert not any(
        user is None and is_framework_dir_call(cmd) for cmd, user in sandbox.exec_calls
    )
    assert not any(is_tools_cli_command(cmd) for cmd, _ in sandbox.exec_calls)


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(
            violation(f"{SANDBOX_TOOLS_DIR} is owned by uid 1111, expected uid 0"),
            id="wrong-owner",
        ),
        pytest.param(
            violation(f"{SANDBOX_TOOLS_DIR} has mode 755, expected 700"),
            id="wrong-mode",
        ),
        pytest.param(MISSING, id="not-installed-yet"),
        pytest.param(UNAVAILABLE, id="cannot-verify"),
        pytest.param(
            ExecResult(
                success=True,
                returncode=0,
                stdout="missing\n",
                stderr=f"{_VERIFIED_MARKER}\n",
            ),
            id="launcher-missing",
        ),
        pytest.param(
            ExecResult(
                success=True,
                returncode=0,
                stdout="a1ff\n",  # stat -c %f of a symlink
                stderr=f"{_VERIFIED_MARKER}\n",
            ),
            id="launcher-is-symlink",
        ),
    ],
)
async def test_detector_reports_not_installed_and_does_not_downgrade(
    result: ExecResult[str],
) -> None:
    sandbox = CannedSandbox(lambda cmd, user: result)
    sandbox._root_access = ROOT_USABLE
    assert await sandbox_tools._sandbox_tools_installed(sandbox) is False
    # Root is usable, so the default user's view is never consulted.
    assert [user for _, user in sandbox.exec_calls] == ["root"]
    assert sandbox._tools_user is None


async def test_detector_treats_provider_exception_as_not_installed() -> None:
    def raising(cmd: list[str], user: str | None) -> ExecResult[str]:
        raise ConnectionError("sandbox gone")

    sandbox = CannedSandbox(raising)
    sandbox._root_access = ROOT_USABLE
    assert await sandbox_tools._sandbox_tools_installed(sandbox) is False
    assert [user for _, user in sandbox.exec_calls] == ["root"]


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(
            ExecResult(
                success=False,
                returncode=1,
                stdout="",
                stderr=f"{_VERIFIED_MARKER}\nstat: cannot statx 'inspect-sandbox-tools': Input/output error\n",
            ),
            id="stat-fails",
        ),
        pytest.param(
            ExecResult(
                success=True,
                returncode=0,
                stdout="regular file\n",
                stderr=f"{_VERIFIED_MARKER}\n",
            ),
            id="stat-prints-no-mode",
        ),
    ],
)
async def test_detector_treats_unreadable_launcher_as_not_installed(
    result: ExecResult[str],
) -> None:
    """A launcher whose type cannot be read counts as not installed, as before.

    ``stat_in_framework_directory`` raises here where the old inline ``stat`` made
    the check return False; the detector's broad catch keeps the outcome the same
    (injection re-extracts) and the tools user is not pinned by the failure.
    """
    sandbox = CannedSandbox(lambda cmd, user: result)
    sandbox._root_access = ROOT_USABLE
    assert await sandbox_tools._sandbox_tools_installed(sandbox) is False
    assert sandbox._tools_user is None
    assert sandbox._tools_user_resolved is False

    # Same verdict once the tools user is already known.
    sandbox._tools_user = "root"
    assert await sandbox_tools._sandbox_tools_installed(sandbox) is False


async def test_detector_records_no_transcript_events() -> None:
    """The install check must not add its script to the transcript."""
    transcript = Transcript()
    init_transcript(transcript)
    inner = CannedSandbox(
        lambda cmd, user: DEFAULT_USER if is_identity_probe(cmd) else REGULAR_FILE
    )
    proxy = SandboxEnvironmentProxy(inner)
    proxy._root_access = ROOT_USABLE

    assert await sandbox_tools._sandbox_tools_installed(proxy) is True
    assert inner.exec_calls, "the probe must still run"
    assert [e for e in transcript.events if isinstance(e, SandboxEvent)] == []
    # Event recording is back on for whatever the tool runs next.
    await proxy.exec(["echo", "hi"])
    [event] = [e for e in transcript.events if isinstance(e, SandboxEvent)]
    assert event.cmd == "echo hi"


ANSWER = '{"jsonrpc": "2.0", "id": 1, "result": "ok"}'
LAUNCH = tools_cli_command("exec", user="root")
"""A request's launch of the CLI when root is usable."""
REFUSED = ExecResult(
    True, 0, f"{TOOLS_DIR_REFUSED}\n", f"sh: cd: can't cd to {SANDBOX_TOOLS_DIR}\n"
)
"""What the CLI launch prints when its directory check fails."""


def installed(cmd: list[str], user: str | None) -> ExecResult[str]:
    """`helper_ok` with the launcher in place, and a CLI that answers `ANSWER`."""
    if cmd == LAUNCH:
        return ExecResult(True, 0, ANSWER, "")
    return REGULAR_FILE if is_framework_dir_call(cmd) else helper_ok(cmd, user)


async def _sample(**policies: ExecPolicy) -> dict[str, SandboxEnvironment]:
    """A sample's sandboxes, set up as an eval sets them up."""
    init_transcript(Transcript())
    return await init_sandbox_environments_sample(
        provider(**policies), "task", None, files={}, setup=None, metadata={}
    )


def _launches(sandbox: SandboxEnvironment) -> int:
    calls = sandbox.as_type(CannedSandbox).exec_calls
    return sum(cmd == LAUNCH for cmd, _ in calls)


async def test_tools_check_the_install_once_per_sandbox() -> None:
    """Calls after a sandbox's first skip the full check.

    `bash_session()` and `text_editor()` look their sandbox up before every
    request: by name, or unnamed, the first that holds the tools.
    """
    environments = await _sample(default=installed, other=installed)
    for name, sandbox in environments.items():
        for _ in range(2):
            target = await sandbox_tools.sandbox_with_injected_tools(
                sandbox_name=None if name == "default" else name
            )
            await SandboxJSONRPCTransport(target, SANDBOX_CLI)(
                "m", {}, False, user="root"
            )
        calls = sandbox.as_type(CannedSandbox).exec_calls
        assert sum(is_framework_dir_call(cmd) for cmd, _ in calls) == 1, name
        assert _launches(sandbox) == 2


class ContainerRestartedError(Exception):
    """Like k8s_sandbox's, raised from an exec that failed after a restart."""


@pytest.mark.parametrize(
    "dying_exec",
    [
        pytest.param(ExecResult(False, 255, "", ""), id="exit-255"),
        pytest.param(ContainerRestartedError("container restarted"), id="raised"),
    ],
)
async def test_a_sandbox_replaced_after_install_fails_every_tools_call(
    stub_artifact: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    dying_exec: ExecResult[str] | Exception,
) -> None:
    """A tools tree gone after it was installed means the sandbox was replaced.

    As when a sample's VM dies: the exec in flight fails, and the next ones run in
    a fresh container with neither the tools nor the sample's files. Installing the
    tools there would let the sample carry on until scoring fails, so every tools
    call fails the sample instead, saying why.
    """
    tree, extractions = False, 0
    dying: list[ExecResult[str] | Exception] = []

    async def extract(*_: object) -> None:
        nonlocal tree, extractions
        tree, extractions = True, extractions + 1

    monkeypatch.setattr(sandbox_tools, "_extract_tools_tree", extract)

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if cmd == LAUNCH and dying:
            result = dying.pop()
            if isinstance(result, Exception):
                raise result
            return result
        if cmd == LAUNCH and not tree:
            return REFUSED
        if is_framework_dir_call(cmd) and wrapped_command(cmd) and not tree:
            return MISSING
        return installed(cmd, user)

    sandbox = (await _sample(default=policy))["default"]
    calls = sandbox.as_type(CannedSandbox).exec_calls
    transport = SandboxJSONRPCTransport(sandbox, SANDBOX_CLI)
    assert await sandbox_tools.sandbox_with_injected_tools() is sandbox
    assert await transport("m", {}, False, user="root") == ANSWER
    checks = sum(is_framework_dir_call(cmd) for cmd, _ in calls)

    dying.append(dying_exec)  # the VM dies under this call
    with pytest.raises((RuntimeError, ContainerRestartedError)):
        await transport("m", {}, False, user="root")
    tree = False  # and the next calls reach a fresh container

    for _ in range(2):
        assert await sandbox_tools.sandbox_with_injected_tools() is sandbox
        with pytest.raises(RuntimeError, match="sandbox was most likely replaced"):
            await transport("m", {}, False, user="root")
    monkeypatch.setenv("INSPECT_EXEC_REMOTE_SERVER_WAIT", "1")  # as Hawk runs
    with pytest.raises(RuntimeError, match="or its VM died") as raised:
        await sandbox.exec_remote(["true"], stream=False)
    assert "can't cd to" in str(raised.value)  # the check's reason
    inputs = sandbox.as_type(CannedSandbox).inputs
    starts = [i for i in inputs if "exec_remote_start" in str(i)]
    assert len(starts) == 1  # not sent again
    assert not any("exec_remote_kill" in str(i) for i in inputs)  # nothing ran
    # Not a tool error shown to the model: it fails the sample.
    assert tool_call_error(raised.value, "bash") is None
    assert extractions == 1
    assert sum(is_framework_dir_call(cmd) for cmd, _ in calls) == checks


async def test_exec_remote_waits_for_an_injection_in_progress(
    stub_artifact: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A check that passes mid-injection does not let `exec_remote` launch into it.

    `bash_session()` and `text_editor()` run the check outside the injection lock
    to pick their sandbox, so it can see (and record) a launcher still being written.
    """
    tree, release = anyio.Event(), anyio.Event()

    async def extract(*_: object) -> None:
        tree.set()  # the launcher is in place; the rest of the tree is not
        await release.wait()

    monkeypatch.setattr(sandbox_tools, "_extract_tools_tree", extract)

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if is_framework_dir_call(cmd) and wrapped_command(cmd) and not tree.is_set():
            return MISSING
        return installed(cmd, user)

    sandbox = CannedSandbox(policy)
    sandbox._root_access = ROOT_USABLE

    async def call_exec_remote() -> None:
        with contextlib.suppress(Exception):  # only when it launches matters here
            await sandbox.exec_remote(["true"], stream=False)

    async with anyio.create_task_group() as tg:
        tg.start_soon(
            partial(sandbox_tools.sandbox_with_injected_tools, sandbox=sandbox)
        )
        await tree.wait()
        assert await sandbox_tools._sandbox_tools_installed(sandbox) is True
        tg.start_soon(call_exec_remote)
        await anyio.wait_all_tasks_blocked()
        assert _launches(sandbox) == 0
        release.set()
    assert _launches(sandbox) >= 1


@pytest.fixture
def tools_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in CLI where `tools_cli_command` launches it, owned by the test user.

    The caller's `PATH` starts with a `stat` that passes any mode check.
    """
    if sys.platform != "linux":
        pytest.skip("the launch check requires GNU/BusyBox stat")
    tools = tmp_path / "parent" / "tools"
    tools.mkdir(parents=True, mode=0o700)
    cli = tools / "inspect-sandbox-tools"
    cli.write_text('#!/bin/sh\npwd -P\nprintf "[%s]" "$@"\necho\necho "$PATH"\n')
    cli.chmod(0o700)
    (tmp_path / "hostile").mkdir()
    (tmp_path / "hostile" / "stat").write_text("#!/bin/sh\necho 700\n")
    (tmp_path / "hostile" / "stat").chmod(0o755)
    _point_tools_cli_at(monkeypatch, tools)
    return tools


def _point_tools_cli_at(monkeypatch: pytest.MonkeyPatch, tools: Path) -> None:
    monkeypatch.setattr(cli_module, "SANDBOX_TOOLS_DIR", str(tools))
    monkeypatch.setattr(cli_module, "SANDBOX_CLI", f"{tools}/inspect-sandbox-tools")


def _caller_path(cwd: Path) -> str:
    return f"{cwd / 'hostile'}:{os.environ['PATH']}"


def _run(cmd: list[str], cwd: Path) -> "subprocess.CompletedProcess[str]":
    env = {**os.environ, "PATH": _caller_path(cwd)}
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def test_tools_cli_command_runs_the_cli_in_the_users_own_directory(
    tools_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI gets its arguments, the caller's directory and `PATH`; a linked parent works."""
    (tmp_path / "link").symlink_to(tools_tree.parent)
    for tools in (tools_tree, tmp_path / "link" / tools_tree.name):
        _point_tools_cli_at(monkeypatch, tools)
        result = _run(tools_cli_command("exec", "a b", user=None), cwd=tmp_path)
        assert result.stdout.splitlines() == [
            str(tmp_path.resolve()),
            "[exec][a b]",
            _caller_path(tmp_path),
        ]


@pytest.mark.parametrize(
    "entry", ["missing", "symlink", "writable", "other-owner", "root-not-root"]
)
def test_tools_cli_command_refuses_a_directory_that_is_not_the_users_own(
    tools_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """A tree re-created by someone else, writable by others, or a link, never runs.

    Others could put a launcher of their own in a writable one, even one the
    user owns (root, moved into place). The caller's `stat` is not trusted. Only a
    root run can tell the owner check apart: other users' 0700 trees can't be entered.
    """
    user = None
    if entry == "missing":
        shutil.rmtree(tools_tree)
    elif entry == "symlink":
        tools_tree.rename(tmp_path / "elsewhere")
        tools_tree.symlink_to(tmp_path / "elsewhere")
    elif entry == "writable":
        tools_tree.chmod(0o777)
    elif entry == "root-not-root":
        # Meant for root but run as someone else, as by a provider that stopped
        # running root's commands as uid 0: that user's own tree is refused.
        user = "root"
        if os.geteuid() == 0:
            os.chown(tools_tree, 65534, 65534)
    elif os.geteuid() == 0:
        # As the tools user is, typically: root can enter anyone's 0700 tree.
        os.chown(tools_tree, 65534, 65534)
    else:
        _point_tools_cli_at(monkeypatch, Path("/usr/bin"))
    result = _run(tools_cli_command("exec", user=user), cwd=tmp_path)
    assert (result.returncode, result.stdout) == (0, f"{TOOLS_DIR_REFUSED}\n")
    if entry == "missing":
        assert str(tools_tree) in result.stderr  # the reason, for the error


# A caller's environment the bootloader would act on: a forced fresh start, the
# process name a restart passes to itself, and a restart's markers gone stale.
_CLEARED_PYI = {"PYINSTALLER_RESET_ENVIRONMENT": "1", "_PYI_LINUX_PROCESS_NAME": "x"}
_INHERITED_PYI = _CLEARED_PYI | {
    "_PYI_ARCHIVE_FILE": "/stale/inspect-sandbox-tools",
    "_PYI_PARENT_PROCESS_LEVEL": "0",
}


@pytest.mark.parametrize("library_path", [None, "", "/opt/lib"])
def test_a_launch_exports_what_the_bootloader_sets_when_it_restarts(
    tools_tree: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    library_path: str | None,
) -> None:
    """A launch sets what the bootloader would set before re-executing itself.

    The archive is named by its real path, through a linked parent too. A caller's
    non-empty `LD_LIBRARY_PATH` follows the bundle's and is kept for the CLI's children.
    The caller's own variables reach the CLI as they were.
    """
    (tools_tree / "inspect-sandbox-tools").write_text("#!/bin/sh\nexec env\n")
    (tmp_path / "link").symlink_to(tools_tree.parent)
    _point_tools_cli_at(monkeypatch, tmp_path / "link" / tools_tree.name)
    # Names as short as the launch's shell might use for its own.
    env = _caller_env(library_path, inherited=True) | {"d": "mine", "p": "mine"}

    result = subprocess.run(
        tools_cli_command("exec", user=None),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    got = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    real = tools_tree.resolve()
    assert got["_PYI_ARCHIVE_FILE"] == f"{real}/inspect-sandbox-tools"
    assert got["_PYI_PARENT_PROCESS_LEVEL"] == "-1"
    assert got["LD_LIBRARY_PATH"] == ":".join(
        part for part in (f"{real}/_internal", library_path) if part
    )
    assert got.get("LD_LIBRARY_PATH_ORIG") == (library_path or None)
    assert not got.keys() & _CLEARED_PYI.keys()
    assert (got["d"], got["p"]) == ("mine", "mine")


@pytest.fixture
def bundled_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """This build's own glibc x86_64 tools tree, where `tools_cli_command` launches it.

    Its server keeps its state in the tree, not where the runner's points.
    """
    bundle = sandbox_tools._binaries_dir() / sandbox_tools._get_executable_name(
        "amd64", dev=False, musl=False
    )
    host = (sys.platform, platform.machine(), platform.libc_ver()[0])
    if host != ("linux", "x86_64", "glibc") or not bundle.exists():
        pytest.skip("runs this build's glibc x86_64 sandbox tools")
    monkeypatch.delenv("INSPECT_SANDBOX_TOOLS_DIR", raising=False)
    tools = tmp_path / "parent" / "tools"
    tools.mkdir(parents=True, mode=0o700)
    subprocess.run(["tar", "xzf", str(bundle), "-C", str(tools)], check=True)
    _point_tools_cli_at(monkeypatch, tools)
    return tools


def _caller_env(library_path: str | None, inherited: bool) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LD_LIBRARY_PATH")}
    if library_path is not None:
        env["LD_LIBRARY_PATH"] = library_path
    return env | (_INHERITED_PYI if inherited else {})


@pytest.mark.parametrize("inherited", [False, True], ids=["clean", "inherited"])
@pytest.mark.parametrize("via", ["parent", "link"])
def test_a_launch_starts_the_bundled_cli_once(
    bundled_tools: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    via: str,
    inherited: bool,
) -> None:
    """This build's CLI starts as one program, not two.

    Its bootloader takes the launch's environment for that of its own restart, so it
    does not exec itself again, and loads the bundle's libraries, not the host's.
    """
    if via == "link":
        (tmp_path / "link").symlink_to(bundled_tools.parent)
        _point_tools_cli_at(monkeypatch, tmp_path / "link" / bundled_tools.name)
    request = json.dumps({"jsonrpc": "2.0", "method": "version", "id": 1})

    # glibc's loader logs each program it starts and each library it loads.
    result = subprocess.run(
        tools_cli_command("exec", request, user=None),
        env=_caller_env(None, inherited) | {"LD_DEBUG": "files"},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert "result" in json.loads(result.stdout), result.stderr[-2000:]
    starts = [
        line
        for line in result.stderr.splitlines()
        if line.endswith(f"transferring control: {cli_module.SANDBOX_CLI}")
    ]
    assert len(starts) == 1
    real = bundled_tools.resolve()
    assert f"calling init: {real}/_internal/libz.so.1\n" in result.stderr


@pytest.mark.parametrize("library_path", [None, "/opt/lib"])
def test_a_command_the_bundled_cli_runs_keeps_the_callers_library_path(
    bundled_tools: Path, library_path: str | None
) -> None:
    """Never the bundle's: its libraries would shadow the sandbox's own."""
    env = _caller_env(library_path, inherited=False)

    def launch(*args: str) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(
            tools_cli_command(*args, user=None),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    request = {
        "jsonrpc": "2.0",
        "method": "exec_remote_start_wait",
        "id": 1,
        "params": {
            "command": 'printf %s "${LD_LIBRARY_PATH-unset}"',
            "request_id": "0" * 32,
            "wait_seconds": 5,
        },
    }
    try:
        assert launch("start-server").returncode == 0
        result = launch("exec", json.dumps(request))
    finally:
        stopped = launch("stop-server")

    assert stopped.returncode == 0
    output = json.loads(result.stdout)["result"]["result"]["stdout"]
    assert output == (library_path or "unset")


async def test_tools_inject_into_a_provider_that_skips_the_base_init(
    stub_artifact: dict[str, object],
) -> None:
    """Several providers never call `SandboxEnvironment.__init__`."""

    class BareSandbox(CannedSandbox):
        def __init__(self) -> None:
            self.policy = installed
            self.exec_calls = []
            self.inputs = []
            self.envs = []
            self.concurrency = []
            self.timeouts = []
            self.written = []

    sandbox = BareSandbox()
    assert await sandbox_tools.sandbox_with_injected_tools(sandbox=sandbox) is sandbox
    execs = len(sandbox.exec_calls)
    assert await sandbox_tools.sandbox_with_injected_tools(sandbox=sandbox) is sandbox
    assert len(sandbox.exec_calls) == execs


@pytest.mark.parametrize(
    "result",
    [
        ExecResult(success=False, returncode=1, stdout="", stderr="sh: boom"),
        ExecResult(success=True, returncode=0, stdout="garbage", stderr=""),
    ],
)
async def test_inject_fails_when_default_user_unknown(
    stub_artifact: dict[str, object], result: ExecResult[str]
) -> None:
    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        return result if is_identity_probe(cmd) else helper_ok(cmd, user)

    with pytest.raises(sandbox_tools.SandboxInjectionError, match="default user"):
        await sandbox_tools._inject_container_tools_code(CannedSandbox(policy))


def test_parse_default_user() -> None:
    parse = sandbox_tools._parse_default_user
    assert parse(
        "Uid: 0\t0\t0\t0\nGid: 0\t0\t0\t0\nGroups: \nHOME: /root\nHOME_SET: 1\n"
    ) == (SandboxDefaultUser(uid=0, gid=0, groups=[], home="/root"))
    assert parse(
        "Uid: 1000 1000 1000 1000\nGid: 5 5 5 5\nGroups: 4 20 1000\nHOME: /\nHOME_SET: 1\n"
    ) == (SandboxDefaultUser(uid=1000, gid=5, groups=[4, 20, 1000], home="/"))
    assert (
        parse("Uid: 5 5 5 5\nGid: 5 5 5 5\nGroups: 5\nHOME: \nHOME_SET: 1\n").home == ""
    )
    assert (
        parse("Uid: 5 5 5 5\nGid: 5 5 5 5\nGroups: 5\nHOME: \nHOME_SET: \n").home
        is None
    )
    # a login banner precedes the probe output and must not shadow it
    assert (
        parse(
            "HOME: /banner\nUid: 5 5 5 5\nGid: 5 5 5 5\nGroups: 5\nHOME: /real\nHOME_SET: 1\n"
        ).home
        == "/real"
    )


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(RuntimeError("docker exec: transient failure"), id="exception"),
        pytest.param(
            ExecResult(success=False, returncode=1, stdout="", stderr="boom"),
            id="failed",
        ),
        pytest.param(
            ExecResult(success=True, returncode=0, stdout="garbage\n", stderr=""),
            id="unparsable",
        ),
    ],
)
async def test_detector_fails_loud_when_identity_probe_fails(
    failure: Exception | ExecResult[str],
) -> None:
    """An unreadable default identity on a healthy root install is an error.

    Not a reinjection; nothing is cached, so the next call retries the probe.
    """
    probes = {"n": 0}

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if is_identity_probe(cmd):
            probes["n"] += 1
            if probes["n"] == 1:
                if isinstance(failure, Exception):
                    raise failure
                return failure
            return DEFAULT_USER
        return REGULAR_FILE

    sandbox = CannedSandbox(policy)
    sandbox._root_access = ROOT_USABLE
    with pytest.raises(sandbox_tools.SandboxDefaultUserError, match="default user"):
        await sandbox_tools._sandbox_tools_installed(sandbox)
    assert sandbox._tools_user is None
    assert sandbox._tools_user_resolved is False
    assert sandbox._tools_default_user is None

    assert await sandbox_tools._sandbox_tools_installed(sandbox) is True
    assert sandbox._tools_user == "root"
    assert sandbox._tools_default_user == NONROOT


@pytest.mark.parametrize(
    "probe",
    [
        pytest.param(root_probe_result("0000000000000000"), id="cap_drop-all"),
        pytest.param(root_probe_result("0000000000000040"), id="setgid-without-setuid"),
        pytest.param(root_probe_result(setgroups="deny"), id="setgroups-denied"),
        pytest.param(root_probe_result(uid="1000"), id="not-root"),
        pytest.param(
            SandboxUserUnsupportedError("cannot execute as root"),
            id="unsupported-user",
        ),
    ],
)
async def test_inject_falls_back_quietly_when_root_cannot_switch_users(
    stub_artifact: dict[str, object],
    _warn_once_messages: list[str],
    probe: ExecResult[str] | SandboxUserUnsupportedError,
) -> None:
    """Root that cannot switch identity is a definitive verdict: rootless, no warning."""

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if is_root_probe(cmd):
            if isinstance(probe, SandboxUserUnsupportedError):
                raise probe
            return probe
        return helper_ok(cmd, user)

    sandbox = CannedSandbox(policy)
    await sandbox_tools._inject_container_tools_code(sandbox)

    assert sandbox._root_access is not None
    assert sandbox._root_access.state == "unusable"
    assert sandbox._tools_user is None
    assert sandbox._tools_default_user is None
    assert stub_artifact["extracted_as"] is None
    assert not any(
        is_framework_dir_call(cmd) and user == "root"
        for cmd, user in sandbox.exec_calls
    )
    assert _warn_once_messages == []


@pytest.mark.parametrize(
    "probe",
    [
        pytest.param(
            ExecResult(success=False, returncode=1, stdout="", stderr="exec failed"),
            id="probe-failed",
        ),
        pytest.param(
            ExecResult(success=True, returncode=0, stdout="allow\n", stderr=""),
            id="probe-output-short",
        ),
        pytest.param(
            ExecResult(
                success=True,
                returncode=0,
                stdout="Welcome: to the VM\nUid: 0 0 0 0\nCapEff: 000001ffffffffff\n",
                stderr="",
            ),
            id="probe-missing-key-with-noise",
        ),
        pytest.param(root_probe_result("not-hex"), id="probe-unparsable"),
    ],
)
async def test_inject_falls_back_with_warning_when_probe_output_has_no_verdict(
    stub_artifact: dict[str, object],
    _warn_once_messages: list[str],
    probe: ExecResult[str],
) -> None:
    """Probe output that cannot be read as a verdict is ambiguous, not "no root"."""

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        return probe if is_root_probe(cmd) else helper_ok(cmd, user)

    sandbox = CannedSandbox(policy)
    await sandbox_tools._inject_container_tools_code(sandbox)

    assert sandbox._root_access is not None
    assert sandbox._root_access.state == "ambiguous"
    assert "KeyError" not in sandbox._root_access.reason
    assert sandbox._tools_user is None
    assert stub_artifact["extracted_as"] is None
    assert root_access_warned(_warn_once_messages)


async def test_root_probe_tolerates_login_shell_noise(
    stub_artifact: dict[str, object],
) -> None:
    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        if is_root_probe(cmd):
            return root_probe_result(noise="Welcome to the VM\n")
        return helper_ok(cmd, user)

    sandbox = CannedSandbox(policy)
    await sandbox_tools._inject_container_tools_code(sandbox)
    assert sandbox._root_access is not None
    assert sandbox._root_access.state == "usable"
    assert sandbox._tools_user == "root"


# ---------------------------------------------------------------------------
# The root-access decision itself: probe, verdict, recording at sample init
# ---------------------------------------------------------------------------


async def test_probe_runs_the_fixed_script_as_root_with_pinned_path() -> None:
    sandbox = CannedSandbox.returning(root_probe_result())
    access = await sandbox_tools._probe_root_access(sandbox)

    assert access.state == "usable"
    [(cmd, user)] = sandbox.exec_calls
    assert user == "root"
    assert cmd[:2] == [SHELL_PATH, "-c"] and is_root_probe(cmd)
    assert sandbox.envs == [{"PATH": SYSTEM_PATH}]


async def test_probe_uses_the_provider_timeout_and_its_retries() -> None:
    """The probe hands the provider its timeout and keeps the provider's retries.

    A provider timeout excludes time spent queued behind other sandbox commands on
    a busy host, which a deadline around the whole exec would count. A retry rides
    out a transient hang, which would otherwise fail the tools for the sample.
    """
    sandbox = CannedSandbox.returning(root_probe_result())
    await sandbox_tools._probe_root_access(sandbox)
    assert sandbox.timeouts == [(sandbox_tools._ROOT_ACCESS_PROBE_TIMEOUT, True)]


@pytest.mark.parametrize(
    "probe, state, reason",
    [
        pytest.param(root_probe_result(), "usable", "can switch users", id="usable"),
        pytest.param(
            root_probe_result(noise="Welcome: to the VM\n"),
            "usable",
            "can switch users",
            id="usable-despite-login-banner",
        ),
        pytest.param(
            root_probe_result(uid="1000"),
            "unusable",
            "run as uid 1000",
            id="runs-as-another-uid",
        ),
        pytest.param(
            ExecResult(
                success=False,
                returncode=1,
                stdout=root_probe_result(uid="1000").stdout,
                stderr="",
            ),
            "unusable",
            "run as uid 1000",
            id="verdict-despite-failing-status",
        ),
        pytest.param(
            root_probe_result("0000000000000000"),
            "unusable",
            "cannot switch users",
            id="cap_drop-all",
        ),
        pytest.param(
            root_probe_result("0000000000000040"),
            "unusable",
            "cannot switch users",
            id="setgid-without-setuid",
        ),
        pytest.param(
            root_probe_result(setgroups="deny"),
            "unusable",
            "setgroups deny",
            id="setgroups-denied",
        ),
        pytest.param(NO_ROOT, "ambiguous", "exit status 126", id="status-no-output"),
        pytest.param(
            ExecResult(
                success=True, returncode=0, stdout="setgroups: allow\n", stderr=""
            ),
            "ambiguous",
            "no verdict",
            id="fields-missing",
        ),
        pytest.param(
            root_probe_result("not-hex"),
            "ambiguous",
            "could not be parsed",
            id="unparsable-caps",
        ),
        pytest.param(
            root_probe_result(uid="garbled"),
            "ambiguous",
            "could not be parsed",
            id="unparsable-uid",
        ),
        pytest.param(
            root_probe_result(setgroups="garbled"),
            "ambiguous",
            "could not be parsed",
            id="unparsable-setgroups",
        ),
        pytest.param(
            ExecResult(
                success=True,
                returncode=0,
                stdout="Uid:\nCapEff: 000001ffffffffff\nsetgroups: allow\n",
                stderr="",
            ),
            "ambiguous",
            "could not be parsed",
            id="empty-uid",
        ),
    ],
)
async def test_probe_verdict(probe: ExecResult[str], state: str, reason: str) -> None:
    access = await sandbox_tools._probe_root_access(CannedSandbox.returning(probe))
    assert access.state == state
    assert reason in access.reason
    assert access.error is None


@pytest.mark.parametrize(
    "error, state",
    [
        pytest.param(
            SandboxUserUnsupportedError("cannot execute as root"),
            "unusable",
            id="unsupported-user",
        ),
        pytest.param(
            RuntimeError("runuser: may not be used by non-root users"),
            "ambiguous",
            id="provider-raises",
        ),
        pytest.param(ConnectionError("sandbox gone"), "ambiguous", id="other-error"),
        pytest.param(
            SandboxUnavailableError("container is not running"),
            "failed",
            id="unavailable",
        ),
        pytest.param(TimeoutError("exec timed out"), "failed", id="timeout"),
        pytest.param(
            SandboxTimeoutError("exec timed out"), "failed", id="sandbox-timeout"
        ),
    ],
)
async def test_probe_exception(error: Exception, state: str) -> None:
    def raising(cmd: list[str], user: str | None) -> ExecResult[str]:
        raise error

    access = await sandbox_tools._probe_root_access(CannedSandbox(raising))
    assert access.state == state
    assert access.error is error
    assert str(error) in access.reason


async def test_probe_cancellation_propagates_and_records_nothing() -> None:
    started = anyio.Event()

    class BlockingSandbox(CannedSandbox):
        async def exec(
            self,
            cmd: list[str],
            input: str | bytes | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            user: str | None = None,
            timeout: int | None = None,
            timeout_retry: bool = True,
            concurrency: bool = True,
        ) -> ExecResult[str]:
            started.set()
            await anyio.sleep_forever()
            raise AssertionError("unreachable")

    sandbox = BlockingSandbox(lambda cmd, user: OK)
    async with anyio.create_task_group() as tg:
        tg.start_soon(sandbox_tools.resolve_root_access, sandbox)
        await started.wait()
        tg.cancel_scope.cancel()
    assert sandbox._root_access is None


@pytest.mark.parametrize("wrapped", [False, True])
async def test_local_root_probe_is_quiet_through_wrappers(
    monkeypatch: pytest.MonkeyPatch,
    wrapped: bool,
    _warn_once_messages: list[str],
) -> None:
    local = LocalSandboxEnvironment()
    sandbox: SandboxEnvironment = local
    if wrapped:
        wrapper = CannedSandbox.returning(OK)
        monkeypatch.setattr(wrapper, "exec", local.exec)
        sandbox = SandboxEnvironmentProxy(wrapper)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            access = await sandbox_tools.resolve_root_access(sandbox)
            if os.name != "posix" or os.geteuid() != 0:
                assert await sandbox_tools._tools_user_for(sandbox) is None
    finally:
        local.directory.cleanup()

    assert sandbox._root_access is access
    assert not [w for w in caught if issubclass(w.category, UserWarning)], caught
    assert _warn_once_messages == []
    if os.name != "posix" or os.geteuid() != 0:
        assert access.state == "unusable"
        assert isinstance(access.error, SandboxUserUnsupportedError)
    elif sys.platform == "linux":
        # root, but possibly with capabilities dropped: definitive either way
        assert access.state in ("usable", "unusable")
    else:
        # no /proc, so the probe reports no identity
        assert access.state == "ambiguous"


def test_local_sandbox_samples_start_without_warnings(
    tmp_path: Path, _warn_once_messages: list[str]
) -> None:
    """An eval on the built-in local sandbox that never uses the tools stays silent.

    The root check runs at every sample's init, so a warning it set off would repeat
    for every sample. Python warnings are captured directly and logged warnings are
    read back from each sample's transcript, not only Inspect's ``warn_once`` list.
    """
    task = Task(
        dataset=[Sample(input="first"), Sample(input="second")],
        solver=generate(),
        sandbox="local",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = eval(task, model="mockllm/model", log_dir=str(tmp_path))[0]

    assert log.status == "success", log.error
    assert log.samples is not None and len(log.samples) == 2
    assert not [w for w in caught if issubclass(w.category, UserWarning)], caught
    logged = [
        event
        for sample in log.samples
        for event in sample.events
        if isinstance(event, LoggerEvent)
        and event.message.level in ("warning", "error", "critical")
    ]
    assert not logged, logged
    assert _warn_once_messages == []


async def test_resolve_records_once_and_never_probes_again() -> None:
    sandbox = CannedSandbox.returning(root_probe_result(uid="1000"))
    first = await sandbox_tools.resolve_root_access(sandbox)
    assert first.state == "unusable"
    assert sandbox._root_access is first
    assert await sandbox_tools.resolve_root_access(sandbox) is first
    assert len(sandbox.exec_calls) == 1

    # A recorded decision is returned as is, a failed probe included.
    failed = RootAccess("failed", "did not complete", TimeoutError())
    recorded = CannedSandbox.returning(root_probe_result())
    recorded._root_access = failed
    assert await sandbox_tools.resolve_root_access(recorded) is failed
    assert recorded.exec_calls == []


async def test_resolve_shares_the_decision_between_proxy_and_provider_object() -> None:
    """Whichever of the two views is probed first decides for both.

    `as_type()` returns the object behind the proxy, and a provider may hand out
    one object under several names.
    """
    inner = CannedSandbox.returning(root_probe_result())
    proxy = SandboxEnvironmentProxy(inner)
    access = await sandbox_tools.resolve_root_access(proxy)
    assert proxy._root_access is access
    assert inner._root_access is access
    assert await sandbox_tools.resolve_root_access(inner) is access
    # A second proxy over the same object adopts the decision without probing.
    assert (
        await sandbox_tools.resolve_root_access(SandboxEnvironmentProxy(inner))
        is access
    )
    assert len(inner.exec_calls) == 1

    # And a proxy over an object that already carries one adopts it.
    recorded = CannedSandbox.returning(root_probe_result())
    recorded._root_access = ROOT_UNUSABLE
    assert (
        await sandbox_tools.resolve_root_access(SandboxEnvironmentProxy(recorded))
        is ROOT_UNUSABLE
    )
    assert recorded.exec_calls == []


async def test_resolve_tolerates_a_provider_that_skips_the_base_init() -> None:
    """Several providers never call `SandboxEnvironment.__init__`."""

    class BareSandbox(CannedSandbox):
        def __init__(self) -> None:
            self.policy = lambda cmd, user: root_probe_result()
            self.exec_calls = []
            self.inputs = []
            self.envs = []
            self.concurrency = []
            self.timeouts = []
            self.written = []

    inner = BareSandbox()
    access = await sandbox_tools.resolve_root_access(SandboxEnvironmentProxy(inner))
    assert access.state == "usable"
    assert inner._root_access is access
    assert await sandbox_tools.resolve_root_access(inner) is access


def provider(**policies: ExecPolicy) -> type[SandboxEnvironment]:
    """A sandbox provider whose sample holds one canned sandbox per policy."""

    class Provider(CannedSandbox):
        @classmethod
        async def sample_init(
            cls,
            task_name: str,
            config: SandboxEnvironmentConfigType | None,
            metadata: dict[str, str],
        ) -> dict[str, SandboxEnvironment]:
            return {name: cls(policy) for name, policy in policies.items()}

    return Provider


async def test_sample_init_records_root_access_for_every_sandbox_last(
    _warn_once_messages: list[str],
) -> None:
    """Every sandbox is probed once at the end of sample init and only recorded.

    The probe runs after the sample files and the setup script, so it is the last
    thing that happens before the solver, and no outcome (a failing probe, no
    verdict) fails or warns at this point; the sandbox tools act on it later.
    """

    def usable(cmd: list[str], user: str | None) -> ExecResult[str]:
        return root_probe_result() if is_root_probe(cmd) else OK

    def no_verdict(cmd: list[str], user: str | None) -> ExecResult[str]:
        return NO_ROOT if is_root_probe(cmd) else OK

    def unavailable(cmd: list[str], user: str | None) -> ExecResult[str]:
        if is_root_probe(cmd):
            raise SandboxUnavailableError("container is not running")
        return OK

    transcript = Transcript()
    init_transcript(transcript)
    environments = await init_sandbox_environments_sample(
        provider(default=usable, other=no_verdict, broken=unavailable),
        "task",
        None,
        files={"a.txt": b"a"},
        setup=b"#!/bin/sh\necho hi\n",
        metadata={},
    )

    states = {
        name: env._root_access.state if env._root_access is not None else None
        for name, env in environments.items()
    }
    assert states == {"default": "usable", "other": "ambiguous", "broken": "failed"}
    assert _warn_once_messages == []

    # Files and the setup script (chmod, run, rm) precede the probe on the default
    # sandbox; the others see nothing but their probe.
    default = environments["default"].as_type(CannedSandbox)
    assert default.written[0] == "a.txt" and len(default.written) == 2
    assert len(default.exec_calls) == 4
    last_cmd, last_user = default.exec_calls[-1]
    assert last_user == "root" and is_root_probe(last_cmd)
    for name in ("other", "broken"):
        [(cmd, user)] = environments[name].as_type(CannedSandbox).exec_calls
        assert user == "root" and is_root_probe(cmd)

    # The decision sits on the recording proxy and on the provider object behind
    # it, which `as_type()` hands out. Each probe whose exec returned is one
    # transcript event, the audit record of the verdict (the proxy records nothing
    # for an exec that raised).
    for env in environments.values():
        assert isinstance(env, SandboxEnvironmentProxy)
        assert env.as_type(CannedSandbox)._root_access is env._root_access
    probe_events = [
        e
        for e in transcript.events
        if isinstance(e, SandboxEvent) and (e.options or {}).get("user") == "root"
    ]
    assert len(probe_events) == 2
