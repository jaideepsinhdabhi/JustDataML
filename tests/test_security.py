import os

import pytest

from conftest import write_config
from jdml.Exception import CustomException
from jdml.Utils import load_object, safe_join, save_object


@pytest.mark.parametrize("name", ["../x.csv", "/etc/x.csv", "a/../../x", "..", "sub/../../x"])
def test_safe_join_rejects_escapes(workdir, name):
    with pytest.raises(ValueError):
        safe_join("Output", name)


def test_safe_join_allows_inside_paths(workdir):
    assert safe_join("Output", "ok.csv") == os.path.join("Output", "ok.csv")
    assert safe_join("Output", "sub/ok.csv") == os.path.join("Output", "sub", "ok.csv")


def test_safe_join_rejects_symlink_escape(workdir):
    (workdir / "Output").mkdir()
    os.symlink(workdir.parent, workdir / "Output" / "link")
    with pytest.raises(ValueError):
        safe_join("Output", "link/x.csv")


def test_cli_rejects_traversing_output_before_training(workdir, run_cli):
    cfg = write_config(workdir / "cfg.csv")
    with pytest.raises(SystemExit) as exc:
        run_cli("-C", cfg, "-T", "-P", "test_clf.csv", "-O", "../escaped")
    assert exc.value.code == 2
    assert not (workdir / "artifact").exists()  # nothing trained
    assert not (workdir.parent / "escaped.csv").exists()


def test_data_name_traversal_is_rejected(workdir):
    from jdml.Configuration import DataProcessor

    cfg = write_config(workdir / "cfg.csv", Data_Name="../cfg.csv")
    dp = DataProcessor(cfg)
    dp.load_config()
    with pytest.raises(CustomException, match="resolves outside"):
        dp.read_data()


def test_artifact_roundtrip(workdir):
    save_object("artifact/obj.pkl", {"a": [1, 2]})
    assert load_object("artifact/obj.pkl") == {"a": [1, 2]}


def test_tampered_artifact_is_refused(workdir):
    save_object("artifact/obj.pkl", {"a": 1})
    with open("artifact/obj.pkl", "ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(CustomException, match="Checksum mismatch"):
        load_object("artifact/obj.pkl")


def test_artifact_without_checksum_is_refused(workdir):
    save_object("artifact/obj.pkl", {"a": 1})
    os.remove("artifact/obj.pkl.sha256")
    with pytest.raises(CustomException, match="No checksum"):
        load_object("artifact/obj.pkl")


def test_replaced_artifact_is_refused(workdir):
    import joblib

    save_object("artifact/obj.pkl", {"a": 1})
    joblib.dump({"evil": True}, "artifact/obj.pkl")  # swapped without updating the checksum
    with pytest.raises(CustomException, match="Checksum mismatch"):
        load_object("artifact/obj.pkl")


def test_importing_jdml_has_no_side_effects(tmp_path):
    import subprocess
    import sys

    code = "import jdml.cli, jdml.Logger, os; print(sorted(os.listdir('.')))"
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
