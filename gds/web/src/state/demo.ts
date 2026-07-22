import type {
  AppState,
  CatalogState,
  CommandLifecycle,
  ConfigSnapshot,
  EventRecord,
  JobLifecycle,
  ProductLifecycle,
  Scene,
  SpacecraftStatus,
  TelemetrySample,
  U64,
} from "../types";

export const DEMO_INSTANCE: U64 = "0000000000000001";
const DEMO_GROUND_INSTANCE: U64 = "4f9a2c71d6e80b13";

const demoScene: Scene = {
  scene_ref: { catalog_epoch: 1, scene_id: 1, scene_revision: 7 },
  source_sha256: "c3cdc5b3206f0592c653fffca3b9226466bcd9d88e2976ec9e21976fccd88bd8b3",
  sidecar_sha256: "7d8e2b2e8e6c4a4c30f2a7763a1b8f1c4f8f1b695f7d8e6d10d9e3a1f3c2b4d5",
  shape: [10980, 10980, 3],
  capability: "VERIFIED",
  domain: { acquisition: "2026-07-19T03:11:41Z", region: "T48PYS" },
  metadata: {
    sensor: "RGB optical",
    resolution_m: 10,
    display_profile: "rgb_uint16_srgb-v1",
  },
  active_preview_product_ref: {
    spacecraft_instance_id: DEMO_INSTANCE,
    origin_boot_id: 42,
    product_id: 314,
  },
};

const demoConfig: ConfigSnapshot = {
  config_epoch: 1,
  config_revision: 12,
  model_threshold_bp: 4200,
  coverage_limit_bp: 6500,
};

const demoSpacecraft: SpacecraftStatus = {
  instance_id: DEMO_INSTANCE,
  state: "READY",
  spacecraft_id: 68,
  boot_id: 42,
  link_session_id: "0000000000000039",
  link_generation: "0000000000000001",
  queue_depth: 1,
  queue_capacity: 4,
  contact: "CONNECTED",
  last_telemetry_at: Date.now() - 1_800,
  tm_stale_after_seconds: 5,
  model_release_id: "mobilenetv3-small-demo-20260719",
  model_assurance: "demo_non_validated",
};

const demoCommand: CommandLifecycle = {
  request_key: { ground_instance_id: DEMO_GROUND_INSTANCE, request_id: 41 },
  target_spacecraft_instance_id: DEMO_INSTANCE,
  opcode: 0x00010005,
  command_state: "ACKED",
  outbox_state: "ACKED",
  delivery_mode: "immediate",
  effective_expires_at: "2026-07-20T10:30:00Z",
  accepted_at: "2026-07-20T10:12:10Z",
  updated_at: "2026-07-20T10:12:12Z",
  job_key: { ground_instance_id: DEMO_GROUND_INSTANCE, request_id: 41 },
  product_ref: { spacecraft_instance_id: DEMO_INSTANCE, origin_boot_id: 42, product_id: 314 },
};

const demoJob: JobLifecycle = {
  job_key: { ground_instance_id: DEMO_GROUND_INSTANCE, request_id: 41 },
  spacecraft_instance_id: DEMO_INSTANCE,
  state: "SUCCEEDED",
  progress_bp: 10000,
  stage: "PRODUCT_READY",
  roi: { x: 4096, y: 3072, width: 2048, height: 2048 },
  science_decision: "CLEAR",
  product_ref: { spacecraft_instance_id: DEMO_INSTANCE, origin_boot_id: 42, product_id: 314 },
  model_release_id: demoSpacecraft.model_release_id,
  model_assurance: demoSpacecraft.model_assurance,
  updated_at: "2026-07-20T10:12:26Z",
};

const demoProduct: ProductLifecycle = {
  product_ref: { spacecraft_instance_id: DEMO_INSTANCE, origin_boot_id: 42, product_id: 314 },
  state: "PUBLISHED",
  verified: true,
  bundle_sha256: "5b52e2e43d62bc2f7fd44d70d523f3cc9c00e4f28d5dc5c3a3556f1d2c19a2f1",
  bundle_size: 486_120,
  transfer_id: 17,
  transfer_state: "VERIFIED",
  progress_bp: 10000,
  expected_bytes: 486_120,
  received_bytes: 486_120,
  gap_count: 0,
  checksum_status: "SHA256_MATCH",
  artifacts: [
    { name: "quicklook.webp", size: 128_440, sha256: "b".repeat(64), verified: true, content_type: "image/webp" },
    { name: "cloud_mask.tif", size: 196_608, sha256: "c".repeat(64), verified: true, content_type: "image/tiff" },
    { name: "analysis.json", size: 6_142, sha256: "d".repeat(64), verified: true, content_type: "application/json" },
  ],
};

const demoTelemetry: TelemetrySample[] = [
  {
    source_spacecraft_instance_id: DEMO_INSTANCE,
    source_boot_id: 42,
    simulation_run_id: "0000000000001024",
    link_session_id: "0000000000000039",
    link_frame_id: "0000000000000911",
    apid: 1,
    channel_id: 1001,
    decoded_value: 1,
    satellite_time_us: 1_784_524_328_000_000,
    received_at_us: Date.now() * 1000 - 1_800_000,
  },
  {
    source_spacecraft_instance_id: DEMO_INSTANCE,
    source_boot_id: 42,
    simulation_run_id: "0000000000001024",
    link_session_id: "0000000000000039",
    link_frame_id: "0000000000000912",
    apid: 1,
    channel_id: 1007,
    decoded_value: 0.37,
    satellite_time_us: 1_784_524_328_100_000,
    received_at_us: Date.now() * 1000 - 1_500_000,
  },
];

const demoEvents: EventRecord[] = [
  {
    event_id: "0000000000000038",
    event_name: "PRODUCT_VERIFIED",
    severity: "INFO",
    message: { product_id: 314, checksum: "SHA256_MATCH" },
    server_time: "2026-07-20T10:12:26Z",
    source_spacecraft_instance_id: DEMO_INSTANCE,
    source_boot_id: 42,
    request_key: demoCommand.request_key,
  },
  {
    event_id: "0000000000000037",
    event_name: "ANALYSIS_PROGRESS",
    severity: "INFO",
    message: { stage: "INFERENCE", progress_bp: 7200 },
    server_time: "2026-07-20T10:12:23Z",
    source_spacecraft_instance_id: DEMO_INSTANCE,
    source_boot_id: 42,
    request_key: demoCommand.request_key,
  },
  {
    event_id: "0000000000000036",
    event_name: "COMMAND_ACKED",
    severity: "INFO",
    message: { opcode: "ROI_REQUEST" },
    server_time: "2026-07-20T10:12:12Z",
    source_spacecraft_instance_id: DEMO_INSTANCE,
    source_boot_id: 42,
    request_key: demoCommand.request_key,
  },
];

export function createDemoState(): AppState {
  const catalog: CatalogState = {
    status: {
      spacecraft_instance_id: DEMO_INSTANCE,
      catalog_epoch: 1,
      catalog_revision: 7,
      snapshot_sha256: "a".repeat(64),
      synced: true,
      stale: false,
      scene_count: 1,
      source_boot_id: 42,
      link_session_id: demoSpacecraft.link_session_id,
    },
    sceneKeys: ["0000000000000001:1:1:7"],
    nextCursor: null,
  };
  return {
    runtime: {
      browser_gds: "DEMO",
      gds_satellite: "CONNECTED",
      as_of_event_id: "0000000000000038",
      link_session_id: demoSpacecraft.link_session_id,
      last_updated_at: new Date(Date.now() - 1_800).toISOString(),
      fault_profile: {
        profile_id: "demo_lossless",
        seed: "000000000000002a",
        loss_bp: 0,
        duplicate_bp: 0,
        corruption_bp: 0,
        blackout: false,
      },
    },
    spacecraft: { [DEMO_INSTANCE]: demoSpacecraft },
    catalogs: { [DEMO_INSTANCE]: catalog },
    scenes: { ["0000000000000001:1:1:7"]: demoScene },
    commands: { [`${DEMO_GROUND_INSTANCE}:41`]: demoCommand },
    jobs: { [`${DEMO_GROUND_INSTANCE}:41`]: demoJob },
    products: { [`${DEMO_INSTANCE}:42:314`]: demoProduct },
    telemetry: demoTelemetry,
    events: demoEvents,
    configs: { [DEMO_INSTANCE]: demoConfig },
    last_event_id: "0000000000000038",
    snapshot_received_at: Date.now(),
  };
}
