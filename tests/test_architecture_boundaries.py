from pathlib import Path


def test_domain_layer_does_not_import_runtime_layers():
    root = Path("src/masked_team_league/domain")
    forbidden = ("oracles", "league", "training", "reporting", "real_platform")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert f"masked_team_league.{name}" not in text
            assert f"..{name}" not in text
