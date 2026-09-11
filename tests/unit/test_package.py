from controlflow.cli import app
from controlflow.serving.app import health


def test_cli_and_health_import() -> None:
    assert app.info.help
    assert health()["external_actions"] == "disabled"
