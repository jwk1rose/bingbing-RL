"""运行单轮联赛的包内 CLI 入口。"""

from __future__ import annotations

from masked_team_league.cli.commands.run_league_round import build_parser, main


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
