from __future__ import annotations

import ipaddress
import json
import time
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from .backend_codec import PlanBattleScore, build_plan_battle_requests, score_plan_battle_results
from ..domain import AttackPlan, DefensePlan
from .resources import HeroResourceBundle


class OracleBackendClient:
    def __init__(
        self,
        base_url: str,
        *,
        poll_seconds: float = 2.0,
        timeout_seconds: float = 86_400.0,
        request_timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.poll_seconds = float(poll_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self._opener = _build_backend_opener(self.base_url)

    def health(self) -> dict[str, Any]:
        return self._get_json("/health")

    def status(self) -> dict[str, Any]:
        return self._get_json("/api/status")

    def submit_job(self, requests: list[dict[str, Any]], *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._post_json("/jobs", {"requests": requests, "metadata": metadata or {}})

    def job_status(self, job_id: str) -> dict[str, Any]:
        return self._get_json(f"/jobs/{job_id}")

    def wait_for_job(self, job_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            status = self.job_status(job_id)
            if status.get("status") in {"completed", "failed"}:
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(f"oracle backend job timed out: {job_id}")
            time.sleep(self.poll_seconds)

    def submit_and_wait(
        self,
        requests: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        submitted = self.submit_job(requests, metadata=metadata)
        job_id = str(submitted.get("job_id") or "")
        if not job_id:
            raise RuntimeError(f"oracle backend did not return job_id: {submitted}")
        status = self.wait_for_job(job_id)
        if status.get("status") != "completed":
            raise RuntimeError(f"oracle backend job failed: {status}")
        return status

    def read_results(self, job_id: str) -> list[dict[str, Any]]:
        data = self._get_bytes(f"/jobs/{job_id}/results")
        rows: list[dict[str, Any]] = []
        for raw in data.decode("utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("oracle backend results must be JSON objects")
            rows.append(record)
        return rows

    def _get_json(self, path: str) -> dict[str, Any]:
        data = self._get_bytes(path)
        payload = json.loads(data.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"GET {path} did not return a JSON object")
        return payload

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        req = urlrequest.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with self._opener.open(req, timeout=self.request_timeout_seconds) as response:
                data = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"POST {path} failed: HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"POST {path} failed: {exc}") from exc
        payload = json.loads(data.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"POST {path} did not return a JSON object")
        return payload

    def _get_bytes(self, path: str) -> bytes:
        try:
            with self._opener.open(self.base_url + path, timeout=self.request_timeout_seconds) as response:
                return response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GET {path} failed: HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"GET {path} failed: {exc}") from exc


def _build_backend_opener(base_url: str) -> urlrequest.OpenerDirector:
    """为本地 oracle 后端绕过系统代理。"""

    if _is_loopback_url(base_url):
        return urlrequest.build_opener(urlrequest.ProxyHandler({}))
    return urlrequest.build_opener()


def _is_loopback_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_oracle_backend_ready(status: dict[str, Any]) -> bool:
    return bool(
        status.get("runtime_state") == "ready"
        or status.get("runtime_ready")
        or status.get("persistent_pool_ready")
    )


class OracleBackendSimulator:
    def __init__(
        self,
        client: Any,
        resources: HeroResourceBundle,
        *,
        season_buff_ids: int | list[int] | None = None,
        camp_group: int | None = None,
    ) -> None:
        self.client = client
        self.resources = resources
        self.season_buff_ids = season_buff_ids
        self.camp_group = camp_group

    def run_plan(
        self,
        attack: AttackPlan,
        defense: DefensePlan,
        *,
        request_prefix: str,
        base_seed: int,
        metadata: dict[str, Any] | None = None,
    ) -> PlanBattleScore:
        requests = build_plan_battle_requests(
            attack,
            defense,
            self.resources,
            request_prefix=request_prefix,
            base_seed=base_seed,
            season_buff_ids=self.season_buff_ids,
            camp_group=self.camp_group,
        )
        status = self.client.submit_and_wait(requests, metadata=metadata or {"kind": "masked-team-league"})
        results = self.client.read_results(str(status["job_id"]))
        return score_plan_battle_results(attack, requests, results)
