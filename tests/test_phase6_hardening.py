"""Phase 6 hardening: manifests, security, restart, fault and full local SIL flow."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gds.http_app import LocalSilMission, create_app
from gds.release_manifest import build_release_manifest
from gds.run_manifest import (
    AtomicRunManifestStore,
    ReplayAvailability,
    RunManifestError,
    RunState,
    SimulationRunManifest,
)
from gds.topology import TopologyError, TopologyProfile
from link_sim.fault_model import FaultModel, FaultProfile
from link_sim.replay_manager import ArtifactStatus, ReplayManager, ReplaySegment
from protocol.slo import SloProfile


ROOT = Path(__file__).resolve().parents[1]


def _manifest() -> SimulationRunManifest:
    return SimulationRunManifest.open(
        simulation_run_id=1,
        release_id="release-p6",
        spacecraft_instance_id=1,
        scoped_scene_ref={"scene_ref": {"catalog_epoch": 1, "scene_id": 1, "scene_revision": 1}},
        source_snapshot={"source_sha256": "a" * 64},
        config_revision="config-1",
        model_revision="model-1",
        deployment_profile_revision="cpu-1",
        fault_profile_revision="lossless-1",
        seed=42,
        clock={"base": "tai", "epoch": "2000-01-01T00:00:00Z", "resolution_ns": 1},
        opened_at="2026-07-20T00:00:00Z",
    )


def test_run_manifest_atomic_lifecycle_and_evict_does_not_claim_bytes(tmp_path: Path):
    store = AtomicRunManifestStore(tmp_path / "run.json")
    opened = _manifest()
    store.write(opened)
    assert store.read().state == RunState.OPEN
    recovered = store.recover_open()
    assert recovered is not None and recovered.state == RunState.INCOMPLETE_CRASH
    with pytest.raises(RunManifestError):
        recovered.finalize(commands=[], replay_sha256="a" * 64, replay_size_bytes=1, replay_revision="r1")

    final = _manifest().finalize(
        commands=[{"request_key": {"ground_instance_id": "0000000000000001", "request_id": 1}, "opcode": 1}],
        replay_sha256="b" * 64,
        replay_size_bytes=17,
        replay_revision="replay-v1",
    )
    assert final.state == RunState.FINAL
    assert final.replay.state == ReplayAvailability.PRESENT
    evicted = final.set_replay_state(ReplayAvailability.EVICTED)
    assert evicted.replay.sha256 is None and evicted.replay.size_bytes is None


def test_slo_profile_matches_deployable_reference_artifact():
    profile = SloProfile.from_file(ROOT / "protocol" / "slo_profile.yaml")
    artifact_path = ROOT / "artifacts" / "benchmarks" / "local-cpu-pytorch-v2.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    profile.validate_benchmark(artifact, hashlib.sha256(artifact_path.read_bytes()).hexdigest())
    assert profile.oldest_ack_age_ms == 1000
    assert profile.health_max_latency_ms == 2000


def test_release_material_is_reproducible_for_fixed_epoch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1720000000")
    first, first_sha = build_release_manifest(ROOT, require_clean=False)
    second, second_sha = build_release_manifest(ROOT, require_clean=False)
    assert first == second
    assert first_sha == second_sha
    assert first["sbom_sha256"]


def test_topology_profiles_and_negative_limits():
    host = TopologyProfile.from_file(ROOT / "protocol" / "runtime_profile.yaml")
    compose = TopologyProfile.from_file(ROOT / "deploy" / "compose_runtime_profile.yaml")
    host.validate_startup("127.0.0.1")
    compose.validate_startup("0.0.0.0")
    with pytest.raises(TopologyError, match="Host"):
        host.validate_request(host="10.0.0.7", origin=None, peer="10.0.0.7", body_bytes=1, header_bytes=1)
    with pytest.raises(TopologyError, match="body"):
        host.validate_request(host="127.0.0.1", origin=None, peer="127.0.0.1", body_bytes=host.limits.request_body_bytes + 1, header_bytes=1)
    with pytest.raises(TopologyError, match="headers"):
        host.validate_request(host="127.0.0.1", origin=None, peer="127.0.0.1", body_bytes=1, header_bytes=host.limits.header_bytes + 1)


def test_fault_profile_decision_is_replayable_and_seed_scoped():
    profile = FaultProfile(
        frame_loss_rate_ppm=100_000,
        frame_duplicate_rate_ppm=50_000,
        base_latency_ns=110,
        jitter_abs_ns=100,
        bitrate_bps=1_000_000,
    )
    first = tuple(FaultModel(7, 9).apply_faults(profile, 0, frame_id, 1024 * 8, 0, 0) for frame_id in range(64))
    second = tuple(FaultModel(7, 9).apply_faults(profile, 0, frame_id, 1024 * 8, 0, 0) for frame_id in range(64))
    different = tuple(FaultModel(8, 9).apply_faults(profile, 0, frame_id, 1024 * 8, 0, 0) for frame_id in range(64))
    assert first == second
    assert any(left != right for left, right in zip(first, different))


def test_replay_manager_final_and_evicted_states(tmp_path: Path):
    manager = ReplayManager(tmp_path, global_cap_bytes=128, pin_quota_bytes=64, max_artifact_bytes=64)
    assert manager.reserve_artifact(1, 1)
    data = b"replay"
    segment_path = tmp_path / "0000000000000001" / "00000000.seg"
    segment_path.parent.mkdir()
    segment_path.write_bytes(data)
    manager.finalize_artifact(1, ArtifactStatus.FINAL, [ReplaySegment(0, len(data), hashlib.sha256(data).hexdigest(), segment_path)], 2)
    assert manager.get_artifact(1).artifact_sha256
    assert manager.evict_artifact(1)
    artifact = manager.get_artifact(1)
    assert artifact.replay_state.value == "EVICTED"


@pytest.fixture(scope="module")
def local_mission(tmp_path_factory: pytest.TempPathFactory):
    state = tmp_path_factory.mktemp("p6-http")
    mission = LocalSilMission(ROOT, state_directory=state)
    yield mission
    mission.close()


def _client(mission: LocalSilMission) -> TestClient:
    app = create_app(ROOT, service=mission)
    return TestClient(
        app,
        base_url="http://127.0.0.1:8000",
        client=("127.0.0.1", 41000),
    )


def test_http_security_rejects_foreign_peer_origin_and_oversize(local_mission: LocalSilMission):
    client = _client(local_mission)
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["satellite"]["state"] == "READY"
    assert health.json()["slo"]["oldest_ack_age_ms"] == 1000
    assert health.json()["scheduler"]["queue_depths"] == {"ACK": 0, "CONTROL": 0, "FILE": 0}
    assert client.get("/api/state", headers={"host": "10.0.0.8"}).status_code == 403
    assert client.post(
        "/api/commands",
        headers={"Origin": "http://10.0.0.8", "Idempotency-Key": "bad-origin-001"},
        json={"target_spacecraft_instance_id": "0000000000000001", "opcode": 65541, "payload": {}, "delivery_mode": "immediate"},
    ).status_code == 403
    oversized = "x" * (local_mission.topology.limits.request_body_bytes + 1)
    response = client.post(
        "/api/commands",
        headers={"Origin": "http://127.0.0.1:8000", "Idempotency-Key": "too-large-001", "Content-Type": "application/json"},
        content=oversized,
    )
    assert response.status_code == 413


def test_http_websocket_snapshot_and_full_scene_roi_product_flow(local_mission: LocalSilMission):
    client = _client(local_mission)
    state_response = client.get("/api/state")
    assert state_response.status_code == 200
    state = state_response.json()["state"]
    instance = next(iter(state["spacecraft"]))
    catalog = client.get(f"/api/spacecraft/{instance}/scenes").json()
    scene = catalog["scenes"][0]
    config = state["configs"][instance]
    with client.websocket_connect("/ws/telemetry", headers={"host": "127.0.0.1", "origin": "http://127.0.0.1:8000"}) as socket:
        first = socket.receive_json()
        assert first["type"] == "snapshot"
    response = client.post(
        "/api/commands",
        headers={"Origin": "http://127.0.0.1:8000", "Idempotency-Key": "p6-e2e-roi-001"},
        json={
            "target_spacecraft_instance_id": instance,
            "opcode": 0x00010005,
            "payload": {
                "scene_ref": scene["scene_ref"],
                "roi": {"x": 0, "y": 0, "width": 256, "height": 256},
                "expected_config_epoch": config["config_epoch"],
                "expected_config_revision": config["config_revision"],
                "model_threshold_bp": config["model_threshold_bp"],
                "coverage_limit_bp": config["coverage_limit_bp"],
            },
            "delivery_mode": "immediate",
        },
    )
    assert response.status_code == 202, response.text
    accepted = response.json()
    deadline = time.monotonic() + 90
    product = None
    while time.monotonic() < deadline:
        state_poll = client.get("/api/state")
        assert state_poll.status_code == 200, state_poll.text
        current = state_poll.json()["state"]
        product = next((value for value in current["products"].values() if value.get("state") == "PUBLISHED"), None)
        if product is not None:
            break
        time.sleep(1.0)
    if product is None:
        snapshot = local_mission.snapshot()
        raise AssertionError(
            "product was not published: "
            + json.dumps(
                {
                    "commands": snapshot["commands"],
                    "worker": local_mission.satellite.payload.worker_client.health() if local_mission.satellite.payload.worker_client else None,
                },
                sort_keys=True,
            )
        )
    command = client.get(f"/api/commands/{accepted['request_key']['ground_instance_id']}/{accepted['request_key']['request_id']}").json()
    assert command["request_key"] == accepted["request_key"]
    assert command["command_state"] == "ACKED"
    assert command["product_state"] == "PUBLISHED"
    assert product["verified"] is True
    download = client.get(f"/api/products/{instance}/{product['product_ref']['origin_boot_id']}/{product['product_ref']['product_id']}/download")
    assert download.status_code == 200 and len(download.content) == product["bundle_size"]


def test_blackout_next_contact_and_immediate_rejection(local_mission: LocalSilMission):
    client = _client(local_mission)
    instance = next(iter(client.get("/api/state").json()["state"]["spacecraft"]))
    local_mission.set_contact("BLACKOUT")
    body = {"target_spacecraft_instance_id": instance, "opcode": 0x00010002, "payload": {}, "delivery_mode": "immediate"}
    immediate = client.post("/api/commands", headers={"Origin": "http://127.0.0.1:8000", "Idempotency-Key": "p6-blackout-immediate"}, json=body)
    assert immediate.status_code == 409 and immediate.json()["error"] == "NO_CONTACT"
    held = client.post("/api/commands", headers={"Origin": "http://127.0.0.1:8000", "Idempotency-Key": "p6-blackout-next"}, json={**body, "delivery_mode": "next_contact"})
    assert held.status_code == 202 and held.json()["outbox_state"] == "HELD_NO_CONTACT"
    local_mission.set_contact("CONTACT_OPEN")
