from inspect_ai.solver import Setting, Workspace, setting


def test_setting_defaults():
    s = Setting()
    assert s.workspaces == ()
    assert s.tools == ()
    assert s.on_turn is None


def test_workspace_defaults():
    ws = Workspace()
    assert ws.sandbox == "default"
    assert ws.description == ""
    assert ws.user is None


def test_workspace_with_all_fields():
    ws = Workspace(sandbox="main", description="Primary workspace", user="hacker")
    assert ws.sandbox == "main"
    assert ws.description == "Primary workspace"
    assert ws.user == "hacker"


def test_setting_with_workspaces():
    s = Setting(
        workspaces=(
            Workspace(sandbox="default", description="Workspace", user="user"),
            Workspace(sandbox="db", description="Database", user="postgres"),
        ),
    )
    assert len(s.workspaces) == 2
    assert s.workspaces[0].sandbox == "default"
    assert s.workspaces[1].sandbox == "db"


def test_setting_accessor_returns_none_when_no_state():
    assert setting() is None
