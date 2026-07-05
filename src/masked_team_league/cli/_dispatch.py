"""CLI 子命令分发工具。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from importlib import import_module
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandSpec:
    """描述一个迁入 `cli.commands` 的具体命令。"""

    module: str
    summary: str


def dispatch_group(
    *,
    description: str,
    commands: Mapping[str, CommandSpec],
    argv: Sequence[str] | None = None,
) -> int:
    """按第一段子命令把参数转交给旧命令主体。

    旧命令已经各自拥有完整 argparse 解析器。这里仅做分组路由，所以不会
    重复定义业务参数，也避免同一参数在多个地方漂移。
    """

    args = list(sys.argv[1:] if argv is None else argv)
    epilog = "\n".join(f"  {name:<28} {spec.summary}" for name, spec in sorted(commands.items()))
    parser = argparse.ArgumentParser(
        description=description,
        epilog=f"available commands:\n{epilog}" if epilog else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", choices=sorted(commands))

    if not args:
        parser.print_help()
        return 0

    namespace = parser.parse_args(args[:1])
    if namespace.command is None:
        parser.print_help()
        return 0

    spec = commands[namespace.command]
    module = import_module(f"masked_team_league.cli.commands.{spec.module}")

    original_argv = sys.argv
    sys.argv = [f"{original_argv[0]} {namespace.command}", *args[1:]]
    try:
        return int(module.main())
    finally:
        sys.argv = original_argv
