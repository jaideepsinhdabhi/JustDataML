import subprocess
import sys

import pytest

from conftest import write_config


def test_no_args_prints_help(workdir, run_cli, capsys):
    run_cli()
    assert "usage" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("args, message", [
    (["-T"], "--Config"),
    (["-O", "x"], "--Predict"),
    (["-C", "cfg.csv"], "--Train and/or --Predict"),
])
def test_argument_errors(workdir, run_cli, capsys, args, message):
    with pytest.raises(SystemExit) as exc:
        run_cli(*args)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_missing_prediction_file(workdir, run_cli, capsys):
    cfg = write_config(workdir / "cfg.csv")
    with pytest.raises(SystemExit):
        run_cli("-C", cfg, "-P", "nope.csv")
    assert "not found" in capsys.readouterr().err


def test_console_script_smoke(workdir):
    cfg = write_config(workdir / "cfg.csv")
    script = [sys.executable, "-m", "jdml.cli"]
    result = subprocess.run([*script, "-C", cfg, "-T", "-P", "test_clf.csv", "-O", "cli"],
                            cwd=workdir, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (workdir / "Output" / "cli.csv").exists()
    assert list((workdir / "logs").rglob("*.log"))  # CLI sets up logging
