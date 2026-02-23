"""Reusable tool-to-CLI component.

Converts a list of ToolDef objects into a CLI script installed in a sandbox,
with an RPC bridge back to the host for actual tool execution.
"""

import json
from textwrap import dedent
from typing import Any, Callable, Sequence
from uuid import uuid4

from pydantic import JsonValue

from inspect_ai.tool._tool import Tool, ToolResult, ToolSource
from inspect_ai.tool._tool_def import ToolDef, tool_defs
from inspect_ai.tool._tool_params import ToolParam
from inspect_ai.util._sandbox.environment import SandboxEnvironment
from inspect_ai.util._sandbox.service import SandboxServiceMethod, sandbox_service


async def install_tool_cli(
    tools: Sequence[Tool | ToolDef | ToolSource],
    sandbox: SandboxEnvironment,
    *,
    command_name: str = "tools",
    service_name: str = "tool_cli",
    install_dir: str = "/opt/tool_cli",
    user: str | None = None,
) -> dict[str, SandboxServiceMethod]:
    """Generate a CLI script, install it into a sandbox, and return service methods.

    The returned methods dict should be passed to ``sandbox_service()`` by the
    caller, who controls the service lifecycle.

    Args:
        tools: Tools to expose as CLI commands.
        sandbox: Sandbox environment to install into.
        command_name: Shell alias for the CLI command.
        service_name: Name for the sandbox service (used for RPC).
        install_dir: Directory in the sandbox to install the CLI script.
        user: Sandbox user to install as.

    Returns:
        A dict of service methods to pass to ``sandbox_service()``.
    """
    resolved = await tool_defs(tools)
    script = generate_tool_cli_script(resolved, service_name=service_name)
    methods = tool_cli_service_methods(resolved)

    # install into sandbox
    await _install_script(
        sandbox,
        script,
        resolved,
        command_name=command_name,
        install_dir=install_dir,
        user=user,
    )

    return methods


async def run_tool_cli_service(
    tools: Sequence[Tool | ToolDef | ToolSource],
    sandbox: SandboxEnvironment,
    *,
    until: Callable[[], bool],
    command_name: str = "tools",
    service_name: str = "tool_cli",
    install_dir: str = "/opt/tool_cli",
    user: str | None = None,
    polling_interval: float | None = None,
) -> None:
    """Install the tool CLI and run the sandbox service until stopped.

    Convenience that combines ``install_tool_cli()`` + ``sandbox_service()``.

    Args:
        tools: Tools to expose as CLI commands.
        sandbox: Sandbox environment to install into.
        until: Function that returns True when the service should stop.
        command_name: Shell alias for the CLI command.
        service_name: Name for the sandbox service (used for RPC).
        install_dir: Directory in the sandbox to install the CLI script.
        user: Sandbox user to install as.
        polling_interval: Polling interval for RPC request checking.
    """
    methods = await install_tool_cli(
        tools,
        sandbox,
        command_name=command_name,
        service_name=service_name,
        install_dir=install_dir,
        user=user,
    )
    await sandbox_service(
        service_name,
        methods,
        until,
        sandbox,
        user=user,
        polling_interval=polling_interval,
    )


def generate_tool_cli_script(
    tool_defs: list[ToolDef],
    service_name: str = "tool_cli",
) -> str:
    """Generate a Python CLI script that calls tools via sandbox service RPC.

    Args:
        tool_defs: Tool definitions to generate CLI commands for.
        service_name: Name of the sandbox service for RPC calls.

    Returns:
        Python source code for the CLI script.
    """
    parts: list[str] = []

    # header
    parts.append(
        dedent(f"""\
        #!/usr/bin/env python3
        import argparse
        import json
        import sys

        sys.path.append("/var/tmp/sandbox-services/{service_name}")
        from {service_name} import call_{service_name}


        def _parse_json(value):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
    """)
    )

    # per-tool handler functions
    for td in tool_defs:
        parts.append(_generate_handler(td, service_name))

    # argparse setup
    parts.append(_generate_parser(tool_defs))

    # dispatch
    parts.append(_generate_dispatch(tool_defs))

    return "\n\n".join(parts) + "\n"


def tool_cli_service_methods(
    tool_defs: list[ToolDef],
) -> dict[str, SandboxServiceMethod]:
    """Create host-side RPC handler methods without installing anything.

    Args:
        tool_defs: Tool definitions to create handlers for.

    Returns:
        A dict mapping method names to async handler functions.
    """
    tools_by_name = {td.name: td for td in tool_defs}

    async def call_tool(tool_name: str, **arguments: Any) -> JsonValue:
        from inspect_ai.event._tool import ToolEvent
        from inspect_ai.log._transcript import transcript
        from inspect_ai.util._span import span

        td = tools_by_name.get(tool_name)
        if td is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool_id = uuid4().hex
        event = ToolEvent(
            id=tool_id,
            function=td.name,
            arguments=_sanitize_arguments(arguments),
        )

        async with span(name=td.name, type="tool"):
            transcript()._event(event)
            result: ToolResult = await td.tool(**arguments)

        event._set_result(
            result=result,
            truncated=None,
            error=None,
            waiting_time=0,
            agent=None,
            failed=None,
            message_id=None,
        )

        return _serialize_result(result)

    return {"call_tool": call_tool}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitize_arguments(arguments: dict[str, Any]) -> dict[str, JsonValue]:
    """Ensure arguments dict is JSON-serializable for ToolEvent."""
    sanitized: dict[str, JsonValue] = {}
    for k, v in arguments.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            sanitized[k] = v
        elif isinstance(v, (list, dict)):
            try:
                json.dumps(v)
                sanitized[k] = v
            except (TypeError, ValueError):
                sanitized[k] = str(v)
        else:
            sanitized[k] = str(v)
    return sanitized


def _serialize_result(result: ToolResult) -> JsonValue:
    """Convert a ToolResult to a JSON-compatible value for RPC response."""
    from inspect_ai._util.content import ContentText

    if isinstance(result, (str, int, float, bool)):
        return result
    if isinstance(result, ContentText):
        return result.text
    if isinstance(result, list):
        text_parts: list[str] = []
        for item in result:
            if isinstance(item, ContentText):
                text_parts.append(item.text)
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts)
    return str(result)


def _param_type_str(param: ToolParam) -> str | None:
    """Get the primary JSON Schema type as a string."""
    if param.type is None:
        return None
    if isinstance(param.type, list):
        for t in param.type:
            if t != "null":
                return t
        return None
    return param.type


def _generate_handler(td: ToolDef, service_name: str) -> str:
    """Generate a handler function for a single tool."""
    lines: list[str] = []
    lines.append(f"def handle_{_safe_name(td.name)}(args):")

    # build arguments dict from parsed args
    lines.append("    kwargs = {}")
    for pname, param in td.parameters.properties.items():
        type_str = _param_type_str(param)
        safe_pname = _safe_name(pname)
        if type_str in ("array", "object"):
            lines.append(f"    if args.{safe_pname} is not None:")
            lines.append(f"        kwargs[{pname!r}] = _parse_json(args.{safe_pname})")
        elif type_str == "boolean":
            lines.append(f"    if args.{safe_pname}:")
            lines.append(f"        kwargs[{pname!r}] = True")
        else:
            lines.append(f"    if args.{safe_pname} is not None:")
            lines.append(f"        kwargs[{pname!r}] = args.{safe_pname}")

    # RPC call and output
    lines.append(
        f"    result = call_{service_name}("
        f"'call_tool', tool_name={td.name!r}, **kwargs)"
    )
    lines.append("    if result is not None:")
    lines.append("        print(result)")

    return "\n".join(lines)


def _generate_parser(tool_defs_list: list[ToolDef]) -> str:
    """Generate the argparse setup code."""
    lines: list[str] = []
    lines.append('parser = argparse.ArgumentParser(description="Tool CLI")')
    lines.append('subparsers = parser.add_subparsers(dest="_tool_name")')

    for td in tool_defs_list:
        safe = _safe_name(td.name)
        desc = td.description.replace('"', '\\"')
        lines.append(
            f'{safe}_parser = subparsers.add_parser({td.name!r}, help="{desc}")'
        )
        for pname, param in td.parameters.properties.items():
            lines.append(_generate_arg(td, pname, param, safe))

    return "\n".join(lines)


def _generate_arg(td: ToolDef, pname: str, param: ToolParam, parser_var: str) -> str:
    """Generate an add_argument call for a single parameter."""
    type_str = _param_type_str(param)
    is_required = pname in td.parameters.required
    description = (param.description or "").replace('"', '\\"')

    # boolean -> store_true flag
    if type_str == "boolean":
        flag = f"--{pname.replace('_', '-')}"
        return (
            f'{parser_var}_parser.add_argument("{flag}", '
            f'action="store_true", default=False, help="{description}")'
        )

    # array/object -> always a --flag taking a JSON string
    if type_str in ("array", "object"):
        flag = f"--{pname.replace('_', '-')}"
        return (
            f'{parser_var}_parser.add_argument("{flag}", '
            f'type=str, default=None, help="{description}")'
        )

    # simple types: positional if required, flag if optional
    type_map = {"string": "str", "integer": "int", "number": "float"}
    py_type = type_map.get(type_str or "string", "str")

    if is_required:
        # positional arg
        extras = f"type={py_type}"
        if param.enum:
            choices = json.dumps(param.enum)
            extras += f", choices={choices}"
        return (
            f"{parser_var}_parser.add_argument({pname!r}, "
            f'{extras}, help="{description}")'
        )
    else:
        # optional flag
        flag = f"--{pname.replace('_', '-')}"
        extras = f"type={py_type}, default=None"
        if param.enum:
            choices = json.dumps(param.enum)
            extras += f", choices={choices}"
        return (
            f'{parser_var}_parser.add_argument("{flag}", '
            f'{extras}, help="{description}")'
        )


def _generate_dispatch(tool_defs_list: list[ToolDef]) -> str:
    """Generate the dispatch block."""
    lines: list[str] = []
    lines.append("args = parser.parse_args()")
    lines.append("command = args._tool_name")

    # build dispatch
    handlers: list[str] = []
    for td in tool_defs_list:
        safe = _safe_name(td.name)
        handlers.append(f'"{td.name}": handle_{safe}')

    lines.append("handlers = {" + ", ".join(handlers) + "}")
    lines.append("if command in handlers:")
    lines.append("    handlers[command](args)")
    lines.append("else:")
    lines.append("    parser.print_help()")

    return "\n".join(lines)


def _safe_name(name: str) -> str:
    """Convert a tool name to a valid Python identifier."""
    return name.replace("-", "_").replace(".", "_")


async def _install_script(
    sandbox: SandboxEnvironment,
    script: str,
    tool_defs_list: list[ToolDef],
    *,
    command_name: str,
    install_dir: str,
    user: str | None,
) -> None:
    """Install the CLI script into the sandbox."""
    # create install dir
    await _checked_exec(sandbox, ["mkdir", "-p", install_dir], user="root")
    if user and user != "root":
        await _checked_exec(sandbox, ["chown", user, install_dir], user="root")

    # Named distinctly from the service module (e.g. "tool_cli.py") to avoid
    # a circular import when the script's directory is on sys.path.
    script_path = f"{install_dir}/tool_cli_entry.py"
    await _checked_exec(sandbox, ["tee", "--", script_path], input=script, user=user)
    await _checked_exec(sandbox, ["chmod", "+x", script_path], user=user)

    # determine user's home directory for .bashrc
    if user:
        result = await sandbox.exec(
            ["bash", "-c", f"getent passwd {user} | cut -d: -f6"], user=user
        )
        home_dir = result.stdout.strip() if result.success else f"/home/{user}"
    else:
        result = await sandbox.exec(["bash", "-c", "echo $HOME"], user=user)
        home_dir = (
            result.stdout.strip()
            if result.success and result.stdout.strip()
            else "/root"
        )

    # build bash alias and tab completion
    tool_names = " ".join(td.name for td in tool_defs_list)
    bashrc_addition = dedent(f"""

        # Tool CLI alias and completion
        alias {command_name}='python3 {script_path}'

        _{command_name}_completion() {{
            local cur
            cur="${{COMP_WORDS[COMP_CWORD]}}"
            if [ "$COMP_CWORD" -eq 1 ]; then
                COMPREPLY=($(compgen -W "{tool_names}" -- ${{cur}}))
            fi
        }}
        complete -F _{command_name}_completion {command_name}
    """)

    await _checked_exec(
        sandbox,
        ["tee", "-a", f"{home_dir}/.bashrc"],
        input=bashrc_addition,
        user=user,
    )


async def _checked_exec(
    sandbox: SandboxEnvironment,
    cmd: list[str],
    input: str | None = None,
    user: str | None = None,
) -> str:
    """Execute a command in the sandbox, raising on failure."""
    result = await sandbox.exec(cmd, input=input, user=user)
    if not result.success:
        raise RuntimeError(f"Error executing command {' '.join(cmd)}: {result.stderr}")
    return result.stdout
