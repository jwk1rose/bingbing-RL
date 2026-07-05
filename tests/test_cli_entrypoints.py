from __future__ import annotations

import os
import subprocess
import sys


def _run_help(module: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not pythonpath else f"src{os.pathsep}{pythonpath}"
    return subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


def test_package_cli_help_entrypoints_are_available():
    modules = (
        "masked_team_league.cli.run_round",
        "masked_team_league.cli.run_selfplay",
        "masked_team_league.cli.train",
        "masked_team_league.cli.report",
        "masked_team_league.cli.calibrate",
        "masked_team_league.cli.ablate",
    )
    for module in modules:
        result = _run_help(module)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
