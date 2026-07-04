from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from masked_team_league.backend import OracleBackendClient, OracleBackendSimulator, is_oracle_backend_ready
from masked_team_league.backend_codec import build_plan_battle_requests, result_to_attack_win_rate
from masked_team_league.constraints import ConstraintEngine
from masked_team_league.generation import LegalPlanGenerator
from masked_team_league.models import DefensePlan, MatchFormat, Team
from masked_team_league.resources import (
    load_decoded_runtime_rules,
    load_hero_resource_bundle,
    load_peak_arena_camp_hero_ids,
    load_unique_legend_equip_ids,
)


def _write_heroes(path: Path, count: int = 20) -> None:
    heroes = []
    for hero_id in range(1, count + 1):
        heroes.append(
            {
                "id": hero_id,
                "name": f"H{hero_id}",
                "displayName": f"英雄{hero_id}",
                "script": f"h{hero_id}",
                "level": 100,
                "stars": 5,
                "rank": 23,
                "equipIds": [6000 + hero_id, 6100 + hero_id, 6200 + hero_id, 6300 + hero_id, 6400 + hero_id, 1000 + hero_id],
                "stats": {"GS": 10000 + hero_id * 100, "HP": 1000 + hero_id},
                "positionType": "front" if hero_id <= 5 else "mid" if hero_id <= 12 else "back",
            }
        )
    path.write_text(json.dumps({"heroes": heroes}, ensure_ascii=False), encoding="utf-8")


def test_load_hero_resource_bundle_produces_position_aware_loadouts(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)

    bundle = load_hero_resource_bundle(heroes_path, unique_equip_star=4, unique_legend_equip_ids={1001})

    assert len(bundle.loadouts) == 20
    first = bundle.loadouts[0]
    assert first.hero_id == 1
    assert first.unique_equip_id == 1001
    assert first.unique_equip_star == 4
    assert first.final_power == 10100
    assert first.standing_bucket == "front"
    assert first.standing_rank < bundle.by_hero_id[12].standing_rank < bundle.by_hero_id[20].standing_rank
    proto = bundle.to_hero_proto(first, instance_id=3)
    assert proto["_tid"] == 1
    assert proto["_legend_equip"]["_equip"]["_type_id"] == 1001
    assert proto["_legend_equip"]["_equip"]["_star"] == 4
    assert len(proto["_items"]) == 6


def test_load_hero_resource_bundle_does_not_treat_normal_equips_as_unique(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)

    bundle = load_hero_resource_bundle(heroes_path)

    assert bundle.by_hero_id[1].unique_equip_id is None
    assert bundle.by_hero_id[1].normal_equip_ids[-1] == 1001


def test_build_plan_battle_requests_uses_backend_proto_shape(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path, count=40)
    bundle = load_hero_resource_bundle(heroes_path, unique_equip_star=5, unique_legend_equip_ids={1000 + value for value in range(1, 41)})
    fmt = MatchFormat(3)
    generator = LegalPlanGenerator(bundle.loadouts, seed=10)
    attack = generator.generate_attack_plan(fmt)
    defense = generator.generate_defense_plan(fmt)

    requests = build_plan_battle_requests(
        attack,
        defense,
        bundle,
        request_prefix="unit",
        base_seed=20260704,
        season_buff_ids=[101, 102],
        camp_group=3,
    )

    assert len(requests) == 3
    assert requests[0]["request_id"] == "unit-r1"
    assert requests[0]["round"] == 1
    assert requests[0]["battleIdx"] == 1
    assert requests[0]["seed"] == 20260704
    assert requests[0]["seasonBuffIds"] == [101, 102]
    assert requests[0]["peakArenaCampGroup"] == 3
    assert len(requests[0]["self_heroes_proto"]) == 5
    assert len(requests[0]["oppo_heroes_proto"]) == 5
    assert len(requests[0]["self_teams_proto"]) == 3
    assert len(requests[0]["oppo_teams_proto"]) == 3
    attack_heroes = [hero["_tid"] for team in requests[0]["self_teams_proto"] for hero in team]
    defense_heroes = [hero["_tid"] for team in requests[0]["oppo_teams_proto"] for hero in team]
    assert len(attack_heroes) == len(set(attack_heroes))
    assert len(defense_heroes) == len(set(defense_heroes))


class _FakeBackendHandler(BaseHTTPRequestHandler):
    jobs: dict[str, dict[str, object]] = {}
    submitted_requests: list[dict[str, object]] = []

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json({"status": "ok"})
            return
        if self.path == "/api/status":
            self._json({"status": "ok", "workers": 2, "runtime_state": "ready"})
            return
        if self.path.startswith("/jobs/") and self.path.endswith("/results"):
            job_id = self.path.split("/")[2]
            rows = self.jobs[job_id]["results"]
            data = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path.startswith("/jobs/"):
            job_id = self.path.split("/")[2]
            self._json(self.jobs[job_id])
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or "0")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path == "/jobs":
            self.submitted_requests = list(payload["requests"])
            job_id = "job_test"
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "completed",
                "total": len(self.submitted_requests),
                "completed": len(self.submitted_requests),
                "errors": 0,
                "results": [
                    {"request_id": request["request_id"], "status": "completed", "battle_result": 0}
                    for request in self.submitted_requests
                ],
            }
            self._json(self.jobs[job_id], status=202)
            return
        self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, payload: dict[str, object], *, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_oracle_backend_client_submits_waits_and_reads_results() -> None:
    _FakeBackendHandler.jobs = {}
    _FakeBackendHandler.submitted_requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeBackendHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        client = OracleBackendClient(url, poll_seconds=0.01, timeout_seconds=5)
        assert client.health()["status"] == "ok"
        status = client.submit_and_wait(
            [{"request_id": "req-1", "self_heroes_proto": [], "oppo_heroes_proto": []}],
            metadata={"kind": "unit"},
        )
        assert status["status"] == "completed"
        assert client.read_results(str(status["job_id"]))[0]["request_id"] == "req-1"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_oracle_backend_simulator_returns_match_win_rate(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path, count=40)
    bundle = load_hero_resource_bundle(heroes_path)
    fmt = MatchFormat(3)
    generator = LegalPlanGenerator(bundle.loadouts, seed=22)
    attack = generator.generate_attack_plan(fmt)
    defense = generator.generate_defense_plan(fmt)

    class FakeClient:
        def submit_and_wait(self, requests, *, metadata=None):
            self.requests = requests
            return {"job_id": "job-1", "status": "completed", "completed": len(requests), "errors": 0}

        def read_results(self, job_id):
            return [
                {"request_id": "oracle-r1", "status": "completed", "battle_result": 0},
                {"request_id": "oracle-r2", "status": "completed", "battle_result": 1},
                {"request_id": "oracle-r3", "status": "completed", "battle_result": 0},
            ]

    simulator = OracleBackendSimulator(FakeClient(), bundle)
    score = simulator.run_plan(attack, defense, request_prefix="oracle", base_seed=1)

    assert score.attack_match_win_rate > 0.999
    assert score.round_win_rates == (1.0, 0.0, 1.0)
    assert len(score.requests) == 3
    assert result_to_attack_win_rate({"battle_result": 0}) == 1.0
    assert result_to_attack_win_rate({"battle_result": 1}) == 0.0


def test_oracle_backend_ready_requires_running_worker_pool() -> None:
    assert is_oracle_backend_ready({"runtime_state": "ready"}) is True
    assert is_oracle_backend_ready({"runtime_ready": True}) is True
    assert is_oracle_backend_ready({"persistent_pool_ready": True}) is True
    assert is_oracle_backend_ready({"runtime_state": "stopped"}) is False
    assert is_oracle_backend_ready({"status": "ok"}) is False


def test_load_unique_legend_equip_ids_from_lua(tmp_path: Path) -> None:
    path = tmp_path / "LegendEquip.lua"
    path.write_text(
        'return {{[1] = {1,"A","i","s","p",true,4},[2] = {2,"B","i","s","p",false,4},[3] = {3,"C","i","s","p",true,4}}}',
        encoding="utf-8",
    )

    assert load_unique_legend_equip_ids(path) == (1, 3)


def test_load_peak_arena_camp_hero_ids_from_decoded_lua(tmp_path: Path) -> None:
    (tmp_path / "PeakArenaCampGroup.lua").write_text(
        "return {{[3] = {3,{[1] = 301,[2] = 302}}},{},1}",
        encoding="utf-8",
    )
    (tmp_path / "PeakArenaCampList.lua").write_text(
        'return {{[301] = {301,{[1] = 10,[2] = 20},"A"},[302] = {302,{[1] = 20,[2] = 30},"B"}}},{},1}',
        encoding="utf-8",
    )

    assert load_peak_arena_camp_hero_ids(tmp_path, camp_group=3) == (10, 20, 30)


def test_decoded_runtime_rules_add_battle_only_proto_fields(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path, count=3)
    (tmp_path / "LegendEquip.lua").write_text(
        'return {{[1] = {1,"唯一","i","s","p",true,4},[5] = {5,"普通","i","s","p",false,4}}}',
        encoding="utf-8",
    )
    (tmp_path / "ShardToHero.lua").write_text(
        'return {{[10011] = {10011,1,"英雄1魂匣"}}}',
        encoding="utf-8",
    )
    (tmp_path / "Astrolabe.lua").write_text(
        'return {{[1] = {1,"星盘","i","s","p",1,900,0}}}',
        encoding="utf-8",
    )
    (tmp_path / "AstrolabeRandomAttr.lua").write_text(
        'return {{[1] = {900,101,"A",{},10,true,10},[2] = {900,102,"B",{},20,true,20},'
        '[3] = {900,103,"C",{},30,true,30},[4] = {900,104,"D",{},40,true,40},'
        '[5] = {900,105,"E",{},50,true,50}}}',
        encoding="utf-8",
    )

    rules = load_decoded_runtime_rules(tmp_path)
    bundle = load_hero_resource_bundle(heroes_path, runtime_rules=rules)
    proto = bundle.to_hero_proto(bundle.by_hero_id[1])

    assert proto["_legend_equip"]["_equip"]["_type_id"] == 5
    assert proto["_shard"] == {"_id": 10011, "_level": 25}
    assert proto["_astrolabe"]["_is_unlock"] is True
    assert len(proto["_astrolabe"]["_stars"]) == 5
