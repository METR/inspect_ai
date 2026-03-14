"""Unit tests for exec_remote Job user-switching logic."""

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest
from inspect_sandbox_tools._remote_tools._exec_remote._job import _make_preexec

_OOM_PATCH = "inspect_sandbox_tools._remote_tools._exec_remote._job._set_oom_score_adj"


class TestMakePreexec:
    """Tests for the _make_preexec function."""

    def test_no_user_only_sets_oom(self) -> None:
        """When username is None, preexec only sets OOM score."""
        preexec = _make_preexec(None)
        with patch(_OOM_PATCH) as mock_oom, patch("os.setuid") as mock_setuid:
            preexec()
            mock_oom.assert_called_once()
            mock_setuid.assert_not_called()

    @patch("os.setuid")
    @patch("os.setgid")
    @patch("os.initgroups")
    @patch("pwd.getpwnam")
    def test_user_switches_in_correct_order(
        self,
        mock_getpwnam: MagicMock,
        mock_initgroups: MagicMock,
        mock_setgid: MagicMock,
        mock_setuid: MagicMock,
    ) -> None:
        """Verifies initgroups -> setgid -> setuid order and correct args."""
        pw = MagicMock()
        pw.pw_uid = 1000
        pw.pw_gid = 1000
        mock_getpwnam.return_value = pw

        call_order: list[str] = []
        mock_initgroups.side_effect = lambda *a: call_order.append("initgroups")
        mock_setgid.side_effect = lambda *a: call_order.append("setgid")
        mock_setuid.side_effect = lambda *a: call_order.append("setuid")

        preexec = _make_preexec("testuser")
        with patch(_OOM_PATCH):
            preexec()

        mock_getpwnam.assert_called_once_with("testuser")
        mock_initgroups.assert_called_once_with("testuser", 1000)
        mock_setgid.assert_called_once_with(1000)
        mock_setuid.assert_called_once_with(1000)
        assert call_order == ["initgroups", "setgid", "setuid"]

    @patch("os._exit", side_effect=SystemExit(1))
    @patch("pwd.getpwnam", side_effect=KeyError("testuser"))
    def test_nonexistent_user_exits(
        self, mock_getpwnam: MagicMock, mock_exit: MagicMock
    ) -> None:
        """When the user doesn't exist in /etc/passwd, preexec calls os._exit(1)."""
        preexec = _make_preexec("testuser")
        with patch(_OOM_PATCH), pytest.raises(SystemExit):
            preexec()
        mock_exit.assert_called_once_with(1)

    @patch("os._exit", side_effect=SystemExit(1))
    @patch("os.initgroups", side_effect=PermissionError("Operation not permitted"))
    @patch("pwd.getpwnam")
    def test_permission_error_exits(
        self,
        mock_getpwnam: MagicMock,
        mock_initgroups: MagicMock,
        mock_exit: MagicMock,
    ) -> None:
        """When setuid/setgid fails due to missing capabilities, calls os._exit(1)."""
        pw = MagicMock()
        pw.pw_uid = 1000
        pw.pw_gid = 1000
        mock_getpwnam.return_value = pw

        preexec = _make_preexec("testuser")
        with patch(_OOM_PATCH), pytest.raises(SystemExit):
            preexec()
        mock_exit.assert_called_once_with(1)


class TestJobCreateUserValidation:
    """Test that Job.create() rejects user when can_switch_user is False."""

    @pytest.mark.asyncio
    async def test_user_without_can_switch_raises(self) -> None:
        from inspect_sandbox_tools._remote_tools._exec_remote._job import Job
        from inspect_sandbox_tools._util.common_types import ToolException

        with pytest.raises(ToolException, match="Cannot switch to user"):
            await Job.create("echo hello", user="nobody", can_switch_user=False)

    @pytest.mark.asyncio
    async def test_no_user_without_can_switch_works(self) -> None:
        from inspect_sandbox_tools._remote_tools._exec_remote._job import Job

        job = await Job.create("echo hello", user=None, can_switch_user=False)
        assert job.pid > 0
        await job.kill()

    @pytest.mark.asyncio
    async def test_current_user_without_can_switch_works(self) -> None:
        """Requesting the current user should succeed even without root."""
        import getpass

        from inspect_sandbox_tools._remote_tools._exec_remote._job import Job

        current_user = getpass.getuser()
        job = await Job.create("echo hello", user=current_user, can_switch_user=False)
        assert job.pid > 0
        await job.kill()


class TestExecRemoteUserIntegration:
    """Integration tests requiring root. Skipped when not root."""

    @pytest.mark.skipif(os.getuid() != 0, reason="Requires root")
    @pytest.mark.asyncio
    async def test_run_as_nobody(self) -> None:
        from inspect_sandbox_tools._remote_tools._exec_remote._job import Job

        job = await Job.create("id -un", user="nobody", can_switch_user=True)
        result = await job.poll()
        for _ in range(50):
            if result.state == "completed":
                break
            await asyncio.sleep(0.1)
            result = await job.poll()
        assert result.state == "completed"
        assert result.exit_code == 0
        assert result.stdout.strip() == "nobody"
