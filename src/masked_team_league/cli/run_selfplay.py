"""运行多轮自博弈编排的包内 CLI 入口。"""

from __future__ import annotations

from masked_team_league.cli.commands.run_selfplay_orchestrator import build_parser, main


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
