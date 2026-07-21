import type {
  AppState,
  CatalogState,
  CommandLifecycle,
  EventEnvelope,
  EventRecord,
  JobLifecycle,
  ProductLifecycle,
  Scene,
  SnapshotEnvelope,
  SpacecraftStatus,
  TelemetrySample,
  U64,
} from "../types";
import { asU64, productKey, requestKeyOf, sceneKey } from "../types";
import { createDemoState } from "./demo";

type Listener = (state: AppState) => void;

const MAX_EVENTS = 250;
const MAX_TELEMETRY = 500;

function compareCursor(left: U64, right: U64): number {
  const leftValue = BigInt(`0x${asU64(left, "left cursor")}`);
  const rightValue = BigInt(`0x${asU64(right, "right cursor")}`);
  return leftValue === rightValue ? 0 : leftValue < rightValue ? -1 : 1;
}

function objectRecord<T>(value: unknown): Record<string, T> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, T>) : {};
}

function normalizeSpacecraft(value: unknown, key: U64): SpacecraftStatus {
  const item = objectRecord<unknown>(value);
  return {
    instance_id: asU64(item.instance_id ?? key, "spacecraft_instance_id"),
    state: (item.state as SpacecraftStatus["state"]) ?? "OFFLINE",
    spacecraft_id: Number(item.spacecraft_id ?? 68),
    boot_id: item.boot_id == null ? null : Number(item.boot_id),
    link_session_id: item.link_session_id == null ? null : asU64(item.link_session_id, "link_session_id"),
    link_generation: item.link_generation == null ? null : asU64(item.link_generation, "link_generation"),
    queue_depth: Number(item.queue_depth ?? 0),
    queue_capacity: Number(item.queue_capacity ?? 0),
    contact: (item.contact as SpacecraftStatus["contact"]) ?? "NO_CONTACT",
    last_telemetry_at: item.last_telemetry_at == null ? null : Number(item.last_telemetry_at),
    tm_stale_after_seconds: Number(item.tm_stale_after_seconds ?? 5),
    model_release_id: String(item.model_release_id ?? "unknown"),
    model_assurance: String(item.model_assurance ?? "unknown"),
  };
}

function normalizeSnapshotState(snapshot: SnapshotEnvelope): Partial<AppState> {
  const raw = snapshot.state ?? {};
  const partial: Partial<AppState> = {};
  if (Object.prototype.hasOwnProperty.call(raw, "runtime")) partial.runtime = raw.runtime as AppState["runtime"];
  if (Object.prototype.hasOwnProperty.call(raw, "spacecraft")) {
    const spacecraftRaw = objectRecord<unknown>(raw.spacecraft);
    const spacecraft: Record<U64, SpacecraftStatus> = {};
    for (const [key, value] of Object.entries(spacecraftRaw)) {
      if (/^[0-9a-f]{16}$/.test(key)) spacecraft[key] = normalizeSpacecraft(value, key);
    }
    partial.spacecraft = spacecraft;
  }
  if (Object.prototype.hasOwnProperty.call(raw, "catalogs")) partial.catalogs = objectRecord<CatalogState>(raw.catalogs);
  if (Object.prototype.hasOwnProperty.call(raw, "scenes")) partial.scenes = objectRecord<Scene>(raw.scenes);
  if (Object.prototype.hasOwnProperty.call(raw, "commands")) partial.commands = objectRecord<CommandLifecycle>(raw.commands);
  if (Object.prototype.hasOwnProperty.call(raw, "jobs")) partial.jobs = objectRecord<JobLifecycle>(raw.jobs);
  if (Object.prototype.hasOwnProperty.call(raw, "products")) partial.products = objectRecord<ProductLifecycle>(raw.products);
  if (Object.prototype.hasOwnProperty.call(raw, "telemetry")) partial.telemetry = Array.isArray(raw.telemetry) ? raw.telemetry as TelemetrySample[] : [];
  if (Object.prototype.hasOwnProperty.call(raw, "events")) partial.events = Array.isArray(raw.events) ? raw.events as EventRecord[] : [];
  if (Object.prototype.hasOwnProperty.call(raw, "configs")) partial.configs = objectRecord(raw.configs) as Record<U64, import("../types").ConfigSnapshot>;
  const cursor = snapshot.last_event_id ?? snapshot.as_of_event_id ?? raw.last_event_id;
  if (cursor !== undefined) partial.last_event_id = asU64(cursor, "snapshot last_event_id");
  return partial;
}

export function normalizeSnapshot(snapshot: SnapshotEnvelope, base = createDemoState()): AppState {
  const normalized = normalizeSnapshotState(snapshot);
  const state: AppState = {
    ...base,
    ...normalized,
    runtime: { ...base.runtime, ...(normalized.runtime ?? {}) },
    spacecraft: normalized.spacecraft ?? base.spacecraft,
    catalogs: normalized.catalogs ?? base.catalogs,
    scenes: normalized.scenes ?? base.scenes,
    commands: normalized.commands ?? base.commands,
    jobs: normalized.jobs ?? base.jobs,
    products: normalized.products ?? base.products,
    telemetry: normalized.telemetry ?? base.telemetry,
    events: normalized.events ?? base.events,
    configs: normalized.configs ?? base.configs,
    last_event_id: normalized.last_event_id ?? base.last_event_id,
    snapshot_received_at: Date.now(),
  };
  state.runtime = {
    ...state.runtime,
    browser_gds: state.runtime.browser_gds === "DEMO" ? "CONNECTED" : state.runtime.browser_gds,
    as_of_event_id: state.runtime.as_of_event_id ?? state.last_event_id,
  };
  return state;
}

function mergeMessage<T extends object>(current: T | undefined, message: unknown): T {
  return {
    ...(current ?? {}),
    ...(message && typeof message === "object" ? message : {}),
  } as T;
}

export class NormalizedStore {
  private state: AppState;
  private readonly listeners = new Set<Listener>();

  constructor(initial = createDemoState()) {
    this.state = initial;
  }

  getState(): AppState {
    return this.state;
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  replaceSnapshot(snapshot: SnapshotEnvelope): void {
    this.state = normalizeSnapshot(snapshot, this.state);
    this.emit();
  }

  setRuntime(patch: Partial<AppState["runtime"]>): void {
    this.state = { ...this.state, runtime: { ...this.state.runtime, ...patch } };
    this.emit();
  }

  upsertCatalog(instance: U64, catalog: CatalogState, scenes: Scene[]): void {
    const nextScenes = { ...this.state.scenes };
    for (const scene of scenes) nextScenes[sceneKey(instance, scene.scene_ref)] = scene;
    this.state = {
      ...this.state,
      catalogs: { ...this.state.catalogs, [instance]: catalog },
      scenes: nextScenes,
    };
    this.emit();
  }

  applyEnvelope(envelope: EventEnvelope): void {
    if (envelope.snapshot) this.replaceSnapshot(envelope.snapshot);
    for (const event of envelope.events ?? []) this.applyEvent(event);
    if (envelope.event) this.applyEvent(envelope.event);
  }

  applyEvent(event: EventRecord): void {
    const state = this.state;
    const eventId = asU64(event.event_id, "event_id");
    if (state.events.some((item) => item.event_id === eventId)) return;
    const eventInstance = event.source_spacecraft_instance_id;
    const currentBoot = eventInstance ? state.spacecraft[eventInstance]?.boot_id : null;
    if (eventInstance && event.source_boot_id != null && currentBoot != null && event.source_boot_id < currentBoot) {
      return;
    }
    const advancesCursor = compareCursor(eventId, state.last_event_id) > 0;
    const next: AppState = {
      ...state,
      events: [{ ...event, event_id: eventId }, ...state.events.filter((item) => item.event_id !== eventId)].slice(0, MAX_EVENTS),
      last_event_id: advancesCursor ? eventId : state.last_event_id,
      runtime: {
        ...state.runtime,
        as_of_event_id: advancesCursor ? eventId : state.runtime.as_of_event_id,
        last_updated_at: advancesCursor ? event.server_time : state.runtime.last_updated_at,
      },
    };
    const message = event.message && typeof event.message === "object" ? event.message as Record<string, unknown> : {};
    const requestKey = event.request_key ? requestKeyOf(event.request_key) : undefined;
    const instance = event.source_spacecraft_instance_id;
    if (instance && event.source_boot_id != null && currentBoot != null && event.source_boot_id > currentBoot) {
      next.events = next.events.filter((item) => item.event_id === eventId || item.source_spacecraft_instance_id !== instance).slice(0, MAX_EVENTS);
      next.telemetry = next.telemetry.filter((item) => item.source_spacecraft_instance_id !== instance);
      next.jobs = Object.fromEntries(Object.entries(next.jobs).filter(([, job]) => job.spacecraft_instance_id !== instance));
    }
    if (instance && next.spacecraft[instance]) {
      next.spacecraft = {
        ...next.spacecraft,
        [instance]: {
          ...next.spacecraft[instance],
          boot_id: event.source_boot_id ?? next.spacecraft[instance].boot_id,
          last_telemetry_at: event.event_name.includes("TELEMETRY") ? Date.parse(event.server_time) : next.spacecraft[instance].last_telemetry_at,
          state: event.event_name.includes("DEGRADED") ? "DEGRADED" : next.spacecraft[instance].state,
        },
      };
    }
    if (event.event_name.includes("TELEMETRY") && instance) {
      const sample = message as unknown as TelemetrySample;
      if (sample.channel_id != null) next.telemetry = [sample, ...state.telemetry].slice(0, MAX_TELEMETRY);
    }
    if (requestKey && next.commands[requestKey]) {
      next.commands = {
        ...next.commands,
        [requestKey]: mergeMessage(next.commands[requestKey], message),
      };
    }
    if (requestKey && (event.event_name.includes("JOB") || event.event_name.includes("ANALYSIS"))) {
      const current = next.jobs[requestKey];
      next.jobs = { ...next.jobs, [requestKey]: mergeMessage(current, { job_key: event.request_key, spacecraft_instance_id: instance, ...message }) };
    }
    if (message.product_ref && typeof message.product_ref === "object") {
      const ref = message.product_ref as ProductLifecycle["product_ref"];
      const key = productKey(ref);
      next.products = { ...next.products, [key]: mergeMessage(next.products[key], message) };
    }
    this.state = next;
    this.emit();
  }

  addCommand(command: CommandLifecycle): void {
    const key = requestKeyOf(command.request_key);
    this.state = { ...this.state, commands: { ...this.state.commands, [key]: command } };
    this.emit();
  }

  private emit(): void {
    for (const listener of this.listeners) listener(this.state);
  }
}
