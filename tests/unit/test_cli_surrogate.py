"""``suspkin surrogate sample``."""

from pathlib import Path

from typer.testing import CliRunner

from suspkin.cli import app
from suspkin.surrogate.dataset import load_dataset
from tests.unit.test_surrogate_sampling import REAR_VAR, STUDY

runner = CliRunner()


def _study(tmp_path: Path) -> str:
    p = tmp_path / "study.yaml"
    p.write_text(STUDY.replace(REAR_VAR, ""), encoding="utf-8")
    return str(p)


def test_sample_front_and_resume(tmp_path: Path) -> None:
    study, out = _study(tmp_path), tmp_path / "out"
    cmd = ["surrogate", "sample", study, "--root", str(out), "--axle", "front"]
    res = runner.invoke(app, cmd)
    assert res.exit_code == 0, res.output
    assert "feasible" in res.output and "bytes" in res.output
    assert load_dataset(out, "front").n == 12
    assert runner.invoke(app, cmd).exit_code == 2  # non-empty without --resume
    resumed = runner.invoke(app, [*cmd, "--resume"])
    assert resumed.exit_code == 0 and "0 new" in resumed.output


def test_both_skips_axle_without_variables(tmp_path: Path) -> None:
    out = tmp_path / "out"
    res = runner.invoke(app, ["surrogate", "sample", _study(tmp_path), "--root", str(out)])
    assert res.exit_code == 0, res.output
    assert "rear: skipped" in res.output and (out / "dataset" / "front").is_dir()
    assert not (out / "dataset" / "rear").exists()


def test_errors(tmp_path: Path) -> None:
    study, out = _study(tmp_path), str(tmp_path / "o")
    base = ["surrogate", "sample", study, "--root", out]
    assert runner.invoke(app, [*base, "--axle", "middle"]).exit_code == 2
    assert runner.invoke(app, [*base, "--axle", "rear"]).exit_code == 2
    assert runner.invoke(app, [*base, "--workers", "0"]).exit_code == 2
    missing = ["surrogate", "sample", str(tmp_path / "no.yaml"), "--root", out]
    assert runner.invoke(app, missing).exit_code == 2


# --------------------------------------------------------------------------- train / validate

import pytest  # noqa: E402

from tests.unit.test_surrogate_workflow import study_text  # noqa: E402


@pytest.fixture(scope="module")
def cli_trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Path]:
    base = tmp_path_factory.mktemp("clitrain")
    study = base / "study.yaml"
    study.write_text(study_text(), encoding="utf-8")
    root = base / "root"
    res = runner.invoke(app, ["surrogate", "sample", str(study), "--root", str(root)])
    assert res.exit_code == 0, res.output
    res = runner.invoke(
        app,
        ["surrogate", "train", str(study), "--root", str(root), "--device", "cpu", "--trials", "1"],
    )
    assert res.exit_code == 0, res.output
    assert "device: cpu" in res.output and "trials complete" in res.output
    assert "best val score" in res.output and "model" in res.output
    assert (root / "models" / "front" / "model" / "model.pt").is_file()
    assert (root / "models" / "front" / "trials.csv").is_file()
    return str(study), root


def test_train_validate_cli(cli_trained: tuple[str, Path]) -> None:
    study, root = cli_trained
    res = runner.invoke(
        app, ["surrogate", "validate", study, "--root", str(root), "--axle", "front"]
    )
    assert res.exit_code in (0, 1), res.output  # a 4-epoch net is over tolerance: 1 expected
    assert "channel" in res.output and "camber" in res.output and "ratio" in res.output
    assert (root / "validation" / "front" / "validation.csv").is_file()
    assert (root / "validation" / "front" / "figures" / "parity.png").is_file()
    if res.exit_code == 1:
        assert "over tolerance" in res.output


def test_validate_mismatched_study_exit_2(cli_trained: tuple[str, Path], tmp_path: Path) -> None:
    _, root = cli_trained
    other = tmp_path / "other.yaml"
    other.write_text(study_text().replace("points: 7", "points: 9"), encoding="utf-8")
    res = runner.invoke(app, ["surrogate", "validate", str(other), "--root", str(root)])
    assert res.exit_code == 2 and "does not match the study" in res.output
    res = runner.invoke(app, ["surrogate", "train", str(other), "--root", str(root)])
    assert res.exit_code == 2


def test_cuda_unavailable_exit_2(
    cli_trained: tuple[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    study, root = cli_trained
    for cmd in ("train", "validate"):
        res = runner.invoke(app, ["surrogate", cmd, study, "--root", str(root), "--device", "cuda"])
        assert res.exit_code == 2 and "no CUDA device" in res.output


def test_train_validate_argument_errors(cli_trained: tuple[str, Path], tmp_path: Path) -> None:
    study, root = cli_trained
    base = ["surrogate", "train", study, "--root", str(root)]
    assert runner.invoke(app, [*base, "--axle", "middle"]).exit_code == 2
    assert runner.invoke(app, [*base, "--device", "tpu"]).exit_code == 2
    assert runner.invoke(app, [*base, "--timeout", "3 mm"]).exit_code == 2
    assert runner.invoke(app, [*base, "--axle", "rear"]).exit_code == 2  # no rear dataset
    empty = ["surrogate", "validate", study, "--root", str(tmp_path / "none")]
    assert runner.invoke(app, empty).exit_code == 2
    missing = ["surrogate", "validate", str(tmp_path / "no.yaml"), "--root", str(root)]
    assert runner.invoke(app, missing).exit_code == 2


def test_sample_out_alias(tmp_path: Path) -> None:
    out = tmp_path / "o"
    cmd = ["surrogate", "sample", _study(tmp_path), "--out", str(out), "--axle", "front"]
    assert runner.invoke(app, cmd).exit_code == 0
    assert load_dataset(out, "front").n == 12


# --------------------------------------------------------------------------- optimize --surrogate

from tests.unit.test_surrogate_active import pristine, work  # noqa: E402,F401


def test_optimize_surrogate_cli(work: tuple[Path, Path, Path]) -> None:  # noqa: F811
    study, root, run = work
    cmd = [
        "optimize", str(study), "--surrogate", str(root), "--out", str(run),
        "--device", "cpu", "--workers", "1", "--no-report",
    ]  # fmt: skip
    res = runner.invoke(app, cmd)
    assert res.exit_code in (0, 1), res.output
    assert "active learning:" in res.output and "iter 0" in res.output
    assert "f_total: baseline" in res.output and "stopped on max_iterations" in res.output
    assert (run / "al.csv").is_file() and (run / "figures" / "al_progress.png").is_file()
    assert (root / "models" / "front" / "al_0" / "model.pt").is_file()
    again = runner.invoke(app, [*cmd, "--resume", "--evaluations", "500"])
    assert again.exit_code in (0, 1), again.output
    assert "0 new iterations" in again.output


def test_optimize_surrogate_cli_errors(tmp_path: Path) -> None:
    study = Path(_study(tmp_path))
    bad = runner.invoke(app, ["optimize", str(study), "--surrogate", str(tmp_path / "none")])
    assert bad.exit_code == 2 and "error:" in bad.output
    dev = runner.invoke(app, ["optimize", str(study), "--device", "cpu"])
    assert dev.exit_code == 2 and "--surrogate" in dev.output
    odd = runner.invoke(
        app, ["optimize", str(study), "--surrogate", str(tmp_path), "--device", "tpu"]
    )
    assert odd.exit_code == 2
