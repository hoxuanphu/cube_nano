export type U64 = string;

export type ConnectionState = "CONNECTED" | "DISCONNECTED" | "RECONNECTING" | "DEMO";
export type ContactState = "CONNECTED" | "NO_CONTACT" | "BLACKOUT";
export type SatelliteState = "READY" | "DEGRADED" | "BUSY" | "OFFLINE";
export type SceneCapability = "VERIFIED" | "UNSUPPORTED" | "INVALID";

export interface RequestKey {
  ground_instance_id: U64;
  request_id: number;
}

export interface SceneRef {
  catalog_epoch: number;
  scene_id: number;
  scene_revision: number;
}

export interface ProductRef {
  spacecraft_instance_id: U64;
  origin_boot_id: number;
  product_id: number;
}

export interface ROI {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ConfigSnapshot {
  config_epoch: number;
  config_revision: number;
  model_threshold_bp: number;
  coverage_limit_bp: number;
}

export interface Scene {
  scene_ref: SceneRef;
  source_sha256: string;
  sidecar_sha256: string;
  shape: [number, number, number];
  capability: SceneCapability;
  domain: Record<string, unknown>;
  metadata: Record<string, unknown>;
  active_preview_product_ref?: ProductRef | null;
}

export interface CatalogStatus {
  spacecraft_instance_id: U64;
  catalog_epoch: number | null;
  catalog_revision: number | null;
  snapshot_sha256: string | null;
  synced: boolean;
  stale: boolean;
  scene_count: number;
  source_boot_id: number | null;
  link_session_id: U64 | null;
}

export interface CatalogState {
  status: CatalogStatus;
  sceneKeys: string[];
  nextCursor: number | null;
}

export interface SpacecraftStatus {
  instance_id: U64;
  state: SatelliteState;
  spacecraft_id: number;
  boot_id: number | null;
  link_session_id: U64 | null;
  link_generation: U64 | null;
  queue_depth: number;
  queue_capacity: number;
  contact: ContactState;
  last_telemetry_at: number | null;
  tm_stale_after_seconds: number;
  model_release_id: string;
  model_assurance: string;
}

export interface CommandLifecycle {
  request_key: RequestKey;
  target_spacecraft_instance_id: U64;
  opcode: number;
  command_state: string;
  outbox_state: string;
  delivery_mode: "immediate" | "next_contact";
  effective_expires_at: string;
  accepted_at: string;
  updated_at?: string;
  last_error?: string | null;
  job_key?: RequestKey | null;
  product_ref?: ProductRef | null;
  transfer_id?: number | null;
  progress_bp?: number;
  stage?: string;
}

export interface JobLifecycle {
  job_key: RequestKey;
  spacecraft_instance_id: U64;
  state: string;
  progress_bp: number;
  stage: string;
  roi?: ROI;
  science_decision?: string | null;
  error_code?: string | null;
  product_ref?: ProductRef | null;
  model_release_id?: string;
  model_assurance?: string;
  updated_at?: string;
}

export interface ProductArtifact {
  name: string;
  size: number;
  sha256: string;
  verified: boolean;
  content_type?: string;
}

export interface ProductLifecycle {
  product_ref: ProductRef;
  state: "RECEIVING" | "VERIFIED" | "PUBLISHED" | "EVICTED" | string;
  verified: boolean;
  bundle_sha256?: string;
  bundle_size?: number;
  artifacts?: ProductArtifact[];
  transfer_id?: number | null;
  transfer_state?: string;
  progress_bp?: number;
  expected_bytes?: number;
  received_bytes?: number;
  gap_count?: number;
  checksum_status?: string;
  evicted_reason?: string;
  tombstone_until?: string;
}

export interface TelemetrySample {
  source_spacecraft_instance_id: U64;
  source_boot_id: number;
  simulation_run_id: U64;
  link_session_id: U64;
  link_frame_id: U64;
  apid: number;
  channel_id: number;
  decoded_value: unknown;
  satellite_time_us?: number | null;
  received_at_us: number;
}

export interface EventRecord {
  event_id: U64;
  event_name: string;
  severity: "INFO" | "WARNING" | "ERROR" | "CRITICAL" | string;
  message: unknown;
  server_time: string;
  source_spacecraft_instance_id?: U64;
  source_boot_id?: number;
  request_key?: RequestKey;
}

export interface RuntimeStatus {
  browser_gds: ConnectionState;
  gds_satellite: ContactState;
  as_of_event_id: U64;
  link_session_id: U64 | null;
  last_updated_at: string | null;
  fault_profile: {
    profile_id: string;
    seed: U64;
    loss_bp: number;
    duplicate_bp: number;
    corruption_bp: number;
    blackout: boolean;
  };
}

export type SnapshotState = Partial<Omit<AppState, "runtime">> & {
  runtime?: Partial<RuntimeStatus>;
  [key: string]: unknown;
};

export interface AppState {
  runtime: RuntimeStatus;
  spacecraft: Record<U64, SpacecraftStatus>;
  catalogs: Record<U64, CatalogState>;
  scenes: Record<string, Scene>;
  commands: Record<string, CommandLifecycle>;
  jobs: Record<string, JobLifecycle>;
  products: Record<string, ProductLifecycle>;
  telemetry: TelemetrySample[];
  events: EventRecord[];
  configs: Record<U64, ConfigSnapshot>;
  last_event_id: U64;
  snapshot_received_at: number;
}

export interface SnapshotEnvelope {
  state: SnapshotState;
  as_of_event_id?: U64;
  last_event_id?: U64;
}

export interface EventEnvelope {
  type?: string;
  event?: EventRecord;
  snapshot?: SnapshotEnvelope;
  events?: EventRecord[];
  error?: string;
  message?: string;
}

export type CommandBody = {
  target_spacecraft_instance_id: U64;
  opcode: number;
  payload: Record<string, unknown>;
  delivery_mode: "immediate" | "next_contact";
  expires_at?: string;
};

export const OPCODES = {
  CLOUD_SET_CONFIG: 0x00010001,
  SCENE_REQUEST_CATALOG: 0x00010002,
  SCENE_REQUEST_PREVIEW: 0x00010003,
  SCENE_ANALYZE: 0x00010004,
  ROI_REQUEST: 0x00010005,
  JOB_GET_STATUS: 0x00010006,
  JOB_CANCEL: 0x00010007,
  PRODUCT_REQUEST_DOWNLINK: 0x00010008,
  PRODUCT_CANCEL_DOWNLINK: 0x00010009,
} as const;

export function sceneKey(instance: U64, ref: SceneRef): string {
  return `${instance}:${ref.catalog_epoch}:${ref.scene_id}:${ref.scene_revision}`;
}

export function requestKeyOf(value: RequestKey): string {
  return `${value.ground_instance_id}:${value.request_id}`;
}

export function productKey(value: ProductRef): string {
  return `${value.spacecraft_instance_id}:${value.origin_boot_id}:${value.product_id}`;
}

export function isU64(value: unknown): value is U64 {
  return typeof value === "string" && /^[0-9a-f]{16}$/.test(value);
}

export function asU64(value: unknown, label: string): U64 {
  if (!isU64(value)) {
    throw new Error(`${label} must be a 16-digit lowercase hexadecimal U64`);
  }
  return value;
}

export function bpToPercent(value: number): number {
  return Math.round(value) / 100;
}

export function percentToBp(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(10000, Math.round(value * 100)));
}
