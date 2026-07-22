import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  AlertTriangle,
  Archive,
  ArrowDownToLine,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Clock3,
  Cloud,
  Command,
  Database,
  Download,
  Eye,
  FileDown,
  Link2Off,
  LoaderCircle,
  MapPinned,
  Menu,
  Move,
  PackageCheck,
  Radio,
  RadioTower,
  RefreshCw,
  RotateCcw,
  Satellite,
  ScanLine,
  Search,
  Send,
  Settings2,
  ShieldAlert,
  SlidersHorizontal,
  SquareStack,
  TimerReset,
  UploadCloud,
  Wifi,
  X,
} from "lucide-react";
import { GDSApiClient, ApiError, makeIdempotencyKey } from "./api/client";
import { RealtimeClient, type RealtimeStatus } from "./api/realtime";
import { QuicklookViewer } from "./components/QuicklookViewer";
import { NormalizedStore } from "./state/store";
import { DEMO_INSTANCE } from "./state/demo";
import type {
  AppState,
  CatalogState,
  CommandBody,
  CommandLifecycle,
  ConfigSnapshot,
  EventRecord,
  JobLifecycle,
  ProductLifecycle,
  ProductRef,
  ROI,
  Scene,
  SpacecraftStatus,
  U64,
} from "./types";
import { bpToPercent, OPCODES, percentToBp, productKey, requestKeyOf, sceneKey } from "./types";
import { clampRoi, validateRoi, type SceneBounds } from "./utils/roi";

const api = new GDSApiClient();
const store = new NormalizedStore();

function short(value: string | null | undefined, size = 8): string {
  if (!value) return "--";
  return value.length > size * 2 ? `${value.slice(0, size)}...${value.slice(-size)}` : value;
}

function formatAge(timestamp: number | null | undefined): string {
  if (!timestamp) return "--";
  const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 1) return "now";
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  return `${Math.floor(seconds / 3600)}h`;
}

function formatTime(value: string | number | null | undefined): string {
  if (value == null) return "--";
  const date = typeof value === "number" ? new Date(value) : new Date(value);
  return Number.isNaN(date.valueOf()) ? "--" : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

function classForStatus(value: string): string {
  const normalized = value.toUpperCase();
  if (["CONNECTED", "READY", "ACKED", "PUBLISHED", "VERIFIED", "SUCCEEDED", "SHA256_MATCH"].includes(normalized)) return "status-good";
  if (["DEMO", "BUSY", "RECEIVING", "RUNNING", "QUEUED", "RECONNECTING", "STALE TM"].includes(normalized)) return "status-warn";
  if (["DEGRADED", "BLACKOUT", "NO_CONTACT", "HELD_NO_CONTACT", "EVICTED", "FAILED", "DELIVERY_FAILED"].includes(normalized)) return "status-bad";
  return "status-neutral";
}

function StatusPill({ value, label }: { value: string | null | undefined; label?: string }) {
  const safeValue = value ?? "PENDING";
  return <span className={`status-pill ${classForStatus(safeValue)}`}><span className="status-dot" />{label ?? safeValue.replaceAll("_", " ")}</span>;
}

function IconButton({ label, children, onClick, disabled = false, className = "" }: { label: string; children: ReactNode; onClick?: () => void; disabled?: boolean; className?: string }) {
  return <button type="button" className={`icon-button ${className}`} aria-label={label} title={label} onClick={onClick} disabled={disabled}>{children}</button>;
}

function SectionTitle({ icon, eyebrow, title, action }: { icon: React.ReactNode; eyebrow: string; title: string; action?: React.ReactNode }) {
  return <div className="section-title"><div className="section-title-copy"><span className="section-icon">{icon}</span><div><span className="eyebrow">{eyebrow}</span><h2>{title}</h2></div></div>{action}</div>;
}

function EmptyState({ icon, title, detail }: { icon: React.ReactNode; title: string; detail: string }) {
  return <div className="empty-state"><span className="empty-icon">{icon}</span><strong>{title}</strong><span>{detail}</span></div>;
}

function capabilityText(scene: Scene): string {
  if (scene.capability === "VERIFIED") return "verified source";
  if (scene.capability === "UNSUPPORTED") return "unsupported input";
  return "invalid source";
}

function commandName(opcode: number): string {
  const names: Record<number, string> = {
    [OPCODES.CLOUD_SET_CONFIG]: "CLOUD_SET_CONFIG",
    [OPCODES.SCENE_REQUEST_CATALOG]: "SCENE_REQUEST_CATALOG",
    [OPCODES.SCENE_REQUEST_PREVIEW]: "SCENE_REQUEST_PREVIEW",
    [OPCODES.SCENE_ANALYZE]: "SCENE_ANALYZE",
    [OPCODES.ROI_REQUEST]: "ROI_REQUEST",
    [OPCODES.PRODUCT_REQUEST_DOWNLINK]: "PRODUCT_REQUEST_DOWNLINK",
  };
  return names[opcode] ?? `OPCODE 0x${opcode.toString(16).padStart(8, "0")}`;
}

function commandKeyFromAccepted(value: { ground_instance_id: U64; request_id: number }): string {
  return `${value.ground_instance_id}:${value.request_id}`;
}

function defaultRoi(scene: Scene): ROI {
  const bounds: SceneBounds = { width: scene.shape[1], height: scene.shape[0], minPatchSize: 256 };
  return clampRoi({ x: Math.floor((bounds.width - 2048) / 2), y: Math.floor((bounds.height - 2048) / 2), width: 2048, height: 2048 }, bounds);
}

function useMissionState(): AppState {
  const [state, setState] = useState(store.getState());
  const realtimeRef = useRef<RealtimeClient | null>(null);
  useEffect(() => {
    const unsubscribe = store.subscribe(setState);
    let alive = true;
    const sync = async () => {
      try {
        const snapshot = await api.getState();
        if (!alive) return;
        store.replaceSnapshot(snapshot);
        store.setRuntime({ browser_gds: "CONNECTED" });
        const instances = Object.keys(store.getState().spacecraft) as U64[];
        await Promise.all(instances.map(async (instance) => {
          const catalog = await api.getCatalog(instance);
          const catalogState: CatalogState = {
            status: catalog.catalog,
            sceneKeys: catalog.scenes.map((scene) => sceneKey(instance, scene.scene_ref)),
            nextCursor: catalog.next_cursor,
          };
          store.upsertCatalog(instance, catalogState, catalog.scenes);
        }));
        const realtime = new RealtimeClient(api, {
          onStatus: (status: RealtimeStatus) => {
            const browser = status === "CONNECTED" ? "CONNECTED" : status === "CLOSED" ? "DISCONNECTED" : "RECONNECTING";
            store.setRuntime({ browser_gds: browser });
          },
          onSnapshot: (next) => store.replaceSnapshot(next),
          onEvent: (event) => store.applyEvent(event),
          onResync: async () => {
            const refreshed = await api.getState();
            store.replaceSnapshot(refreshed);
            return refreshed.last_event_id ?? refreshed.as_of_event_id;
          },
          onError: (error) => console.warn("GDS realtime:", error),
        });
        realtimeRef.current = realtime;
        realtime.start(store.getState().last_event_id);
      } catch (error) {
        if (!alive) return;
        store.setRuntime({ browser_gds: "DEMO" });
        console.info("GDS API unavailable; showing bounded demo snapshot.", error);
      }
    };
    void sync();
    const poll = window.setInterval(() => {
      void api.getState().then((snapshot) => {
        if (alive) store.replaceSnapshot(snapshot);
      }).catch(() => {
        if (alive) store.setRuntime({ browser_gds: "RECONNECTING" });
      });
    }, 1000);
    return () => {
      alive = false;
      window.clearInterval(poll);
      realtimeRef.current?.stop();
      realtimeRef.current = null;
      unsubscribe();
    };
  }, []);
  return state;
}

function StatusStrip({ state, instance }: { state: AppState; instance: SpacecraftStatus | undefined }) {
  const telemetryAge = instance?.last_telemetry_at == null ? null : Date.now() - instance.last_telemetry_at;
  const stale = telemetryAge != null && telemetryAge > (instance?.tm_stale_after_seconds ?? 5) * 1000;
  const contact = state.runtime.gds_satellite;
  return <div className="status-strip" aria-label="Connection and spacecraft status">
    <div className="status-group"><span className="status-label">BROWSER / GDS</span><StatusPill value={state.runtime.browser_gds} /></div>
    <div className="status-group"><span className="status-label">GDS / SATELLITE</span><StatusPill value={contact} /></div>
    <div className="status-group"><span className="status-label">SPACECRAFT</span><StatusPill value={instance?.state ?? "OFFLINE"} /></div>
    <div className="status-group"><span className="status-label">TM AGE</span><StatusPill value={stale ? "STALE TM" : "CONNECTED"} label={instance?.last_telemetry_at ? formatAge(instance.last_telemetry_at) : "no sample"} /></div>
    <div className="status-group status-group-right"><span className="status-label">QUEUE</span><strong className="mono">{instance?.queue_depth ?? 0}/{instance?.queue_capacity ?? 0}</strong><span className="status-label">LINK</span><span className="mono">{short(state.runtime.link_session_id)}</span></div>
  </div>;
}

function SceneCatalog({ state, instanceId, selectedKey, onSelect }: { state: AppState; instanceId: U64; selectedKey: string | null; onSelect: (key: string) => void }) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<"ALL" | "VERIFIED" | "UNSUPPORTED" | "INVALID">("ALL");
  const catalog = state.catalogs[instanceId];
  const scenes = (catalog?.sceneKeys ?? []).map((key) => state.scenes[key]).filter(Boolean);
  const filtered = scenes.filter((scene) => {
    const searchable = `${scene.scene_ref.scene_id} ${scene.scene_ref.scene_revision} ${scene.metadata.sensor ?? ""} ${scene.domain.region ?? ""}`.toLowerCase();
    return searchable.includes(query.toLowerCase()) && (filter === "ALL" || scene.capability === filter);
  });
  return <aside className="panel catalog-panel" aria-label="Scene catalog">
    <SectionTitle icon={<Archive size={18} />} eyebrow="ONBOARD REPLICA" title="Scene catalog" action={<span className="count-badge">{filtered.length}</span>} />
    <div className="catalog-meta"><span>epoch {catalog?.status.catalog_epoch ?? "--"} / rev {catalog?.status.catalog_revision ?? "--"}</span>{catalog?.status.stale && <StatusPill value="STALE TM" label="STALE" />}</div>
    <label className="search-field"><Search size={16} /><span className="sr-only">Search scenes</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search scene, region, sensor" /></label>
    <div className="filter-row" role="group" aria-label="Scene capability filter">
      {(["ALL", "VERIFIED", "UNSUPPORTED", "INVALID"] as const).map((value) => <button key={value} type="button" className={filter === value ? "filter-button active" : "filter-button"} onClick={() => setFilter(value)}>{value === "ALL" ? "All" : value[0] + value.slice(1).toLowerCase()}</button>)}
    </div>
    <div className="scene-list">
      {filtered.length === 0 ? <EmptyState icon={<Database size={20} />} title="No matching scenes" detail="Catalog replica has no scene for this filter." /> : filtered.map((scene) => {
        const key = sceneKey(instanceId, scene.scene_ref);
        const disabled = scene.capability !== "VERIFIED";
        return <button key={key} type="button" className={`scene-row ${selectedKey === key ? "selected" : ""}`} onClick={() => onSelect(key)}>
          <span className={`scene-thumb ${disabled ? "scene-thumb-muted" : ""}`}><MapPinned size={18} /></span>
          <span className="scene-row-copy"><strong>SCN-{String(scene.scene_ref.scene_id).padStart(4, "0")}</strong><span>rev {scene.scene_ref.scene_revision} / {String(scene.metadata.sensor ?? "RGB optical")}</span><span>{capabilityText(scene)}</span></span>
          <span className={`capability-mark ${disabled ? "bad" : "good"}`} aria-label={scene.capability}><span className="status-dot" /></span>
        </button>;
      })}
    </div>
    <div className="catalog-footer"><span className="live-dot" /> verified replica <span className="mono">{short(catalog?.status.snapshot_sha256, 6)}</span></div>
  </aside>;
}

function NumericRoiEditor({ roi, bounds, onChange }: { roi: ROI | null; bounds: SceneBounds | null; onChange: (roi: ROI) => void }) {
  const validation = roi && bounds ? validateRoi(roi, bounds) : { valid: false, errors: {} };
  const update = (field: keyof ROI, value: string) => {
    if (!roi) return;
    const next = { ...roi, [field]: Number(value) };
    onChange(next);
  };
  return <div className="roi-editor">
    <div className="field-grid four-up">
      {(["x", "y", "width", "height"] as const).map((field) => <label key={field} className="numeric-field"><span>{field}</span><input type="number" min={0} step={1} value={roi?.[field] ?? ""} onChange={(event) => update(field, event.target.value)} aria-invalid={Boolean(validation.errors[field])} /><small>px</small></label>)}
    </div>
    {!validation.valid && <div className="inline-error" role="alert"><AlertTriangle size={14} />{validation.errors.scene ?? validation.errors.width ?? validation.errors.height ?? validation.errors.x ?? validation.errors.y ?? "ROI is incomplete"}</div>}
    {roi && <div className="roi-summary"><span>{(roi.width * roi.height).toLocaleString()} px2</span><span className="mono">[{roi.x}, {roi.y}) -&gt; [{roi.x + roi.width}, {roi.y + roi.height})</span></div>}
  </div>;
}

function ThresholdControl({ label, valueBp, onChange }: { label: string; valueBp: number; onChange: (value: number) => void }) {
  return <div className="threshold-control"><div className="threshold-label"><label htmlFor={label}>{label}</label><span className="mono">{bpToPercent(valueBp).toFixed(2)}%</span></div><div className="threshold-input"><input id={label} type="range" min={0} max={100} step={0.01} value={bpToPercent(valueBp)} onChange={(event) => onChange(percentToBp(Number(event.target.value)))} /><input aria-label={`${label} numeric percentage`} type="number" min={0} max={100} step={0.01} value={bpToPercent(valueBp)} onChange={(event) => onChange(percentToBp(Number(event.target.value)))} /><span>%</span></div></div>;
}

function CommandModal({ instanceId, scene, roi, config, idempotencyKey, contact, faultProfile, catalogStale, tmStale, deliveryMode, setDeliveryMode, onClose, onConfirm, pending, error }: { instanceId: U64; scene: Scene; roi: ROI; config: ConfigSnapshot; idempotencyKey: string; contact: string; faultProfile: AppState["runtime"]["fault_profile"]; catalogStale: boolean; tmStale: boolean; deliveryMode: "immediate" | "next_contact"; setDeliveryMode: (mode: "immediate" | "next_contact") => void; onClose: () => void; onConfirm: () => void; pending: boolean; error: string | null }) {
  const nextContactExpiry = new Date(Date.now() + 60 * 60 * 1000).toISOString();
  const warning = deliveryMode === "next_contact" ? "The command will remain persisted in the GDS outbox until contact or expiry." : "Immediate delivery is authoritative at backend admission; stale TM does not guarantee readiness.";
  return <div className="modal-backdrop" role="presentation"><div className="command-modal" role="dialog" aria-modal="true" aria-labelledby="command-dialog-title">
    <div className="modal-header"><div><span className="eyebrow">COMMAND ADMISSION</span><h2 id="command-dialog-title">Confirm ROI analysis</h2></div><IconButton label="Close confirmation" onClick={onClose}><X size={18} /></IconButton></div>
    <div className="modal-warning"><ShieldAlert size={17} /><span>{warning}</span></div>
    <div className="preview-warning-grid"><div><span className="detail-label">CONTACT</span><StatusPill value={contact} /></div><div><span className="detail-label">FAULT PROFILE</span><strong className="mono">{faultProfile.profile_id} / seed {short(faultProfile.seed, 6)}</strong></div><div><span className="detail-label">CATALOG</span><StatusPill value={catalogStale ? "STALE TM" : "CONNECTED"} label={catalogStale ? "stale replica" : "current replica"} /></div><div><span className="detail-label">TM ASSURANCE</span><StatusPill value={tmStale ? "STALE TM" : "CONNECTED"} label={tmStale ? "cached" : "fresh"} /></div></div>
    <div className="confirmation-grid"><div><span className="detail-label">TARGET INSTANCE</span><strong className="mono">{instanceId}</strong></div><div><span className="detail-label">SCENE REF</span><strong className="mono">{scene.scene_ref.catalog_epoch}/{scene.scene_ref.scene_id}/{scene.scene_ref.scene_revision}</strong></div><div><span className="detail-label">ROI PIXELS</span><strong className="mono">{roi.x}, {roi.y}, {roi.width} x {roi.height}</strong></div><div><span className="detail-label">CONFIG</span><strong className="mono">epoch {config.config_epoch} / rev {config.config_revision}</strong></div><div><span className="detail-label">MODEL / COVERAGE</span><strong>{bpToPercent(config.model_threshold_bp).toFixed(2)}% / {bpToPercent(config.coverage_limit_bp).toFixed(2)}%</strong></div><div><span className="detail-label">EST. DOWNLINK</span><strong>~ 8.4 s / 526 frames</strong></div></div>
    <div className="field-grid two-up modal-fields"><label className="select-field"><span>Delivery mode</span><select value={deliveryMode} onChange={(event) => setDeliveryMode(event.target.value as "immediate" | "next_contact")}><option value="immediate">Immediate</option><option value="next_contact">Next contact</option></select></label><div className="detail-block"><span className="detail-label">EXPIRY</span><strong>{deliveryMode === "next_contact" ? formatTime(nextContactExpiry) : "server default / 5 min"}</strong></div></div>
    <div className="idempotency-block"><span className="detail-label">HTTP IDEMPOTENCY-KEY</span><code>{idempotencyKey}</code><span>RequestKey is allocated only after the server returns 202 Accepted.</span></div>
    {error && <div className="inline-error" role="alert"><AlertTriangle size={14} />{error}</div>}
    <div className="modal-actions"><button type="button" className="button secondary" onClick={onClose}><X size={16} />Cancel</button><button type="button" className="button primary" onClick={onConfirm} disabled={pending}>{pending ? <LoaderCircle size={16} className="spin" /> : <Send size={16} />} {pending ? "Submitting" : "Transmit analysis"}</button></div>
  </div></div>;
}

function LifecyclePanel({ state }: { state: AppState }) {
  const commands = Object.values(state.commands).sort((a, b) => (b.updated_at ?? b.accepted_at).localeCompare(a.updated_at ?? a.accepted_at)).slice(0, 5);
  return <section className="panel lifecycle-panel"><SectionTitle icon={<Command size={18} />} eyebrow="DURABLE LEDGER" title="Command lifecycle" action={<span className="panel-caption">request -&gt; job -&gt; product -&gt; transfer</span>} />
    <div className="lifecycle-list">{commands.length === 0 ? <EmptyState icon={<Command size={20} />} title="No admitted commands" detail="Accepted commands will appear here with their server-issued RequestKey." /> : commands.map((command) => {
      const key = requestKeyOf(command.request_key);
      const job = state.jobs[key];
      const productRef = job?.product_ref ?? command.product_ref;
      const product = productRef ? state.products[productKey(productRef)] : undefined;
      return <div className="lifecycle-row" key={key}><div className="lifecycle-main"><span className="lifecycle-icon"><CheckCircle2 size={16} /></span><div><strong>{commandName(command.opcode)}</strong><span className="mono">RK {short(command.request_key.ground_instance_id)} / {command.request_key.request_id}</span></div></div><div className="lifecycle-stage"><span className="stage-label">COMMAND</span><StatusPill value={command.command_state} /><ChevronRight size={14} /><span className="stage-label">OUTBOX</span><StatusPill value={command.outbox_state} /><ChevronRight size={14} /><span className="stage-label">SCIENCE</span><StatusPill value={job?.state ?? "PENDING"} /><ChevronRight size={14} /><span className="stage-label">PRODUCT</span><StatusPill value={product?.state ?? "PENDING"} /><ChevronRight size={14} /><span className="stage-label">TRANSFER</span><StatusPill value={product?.transfer_state ?? (command.transfer_id ? "QUEUED" : "PENDING")} /></div><div className="lifecycle-time"><span>{command.delivery_mode}</span><time>{formatTime(command.updated_at ?? command.accepted_at)}</time></div></div>;
    })}</div>
  </section>;
}

function TelemetryPanel({ state, instanceId }: { state: AppState; instanceId: U64 }) {
  const events = state.events.filter((event) => !event.source_spacecraft_instance_id || event.source_spacecraft_instance_id === instanceId).slice(0, 5);
  return <section className="panel telemetry-panel"><SectionTitle icon={<Activity size={18} />} eyebrow="APID 1 / APID 2" title="Telemetry & events" action={<span className="panel-caption">cursor {short(state.last_event_id)}</span>} />
    <div className="telemetry-content"><div className="telemetry-metrics"><div><span>TM CHANNELS</span><strong>{new Set(state.telemetry.filter((item) => item.source_spacecraft_instance_id === instanceId).map((item) => item.channel_id)).size}</strong></div><div><span>EVENTS</span><strong>{events.length}</strong></div><div><span>LAST SAMPLE</span><strong>{formatAge(state.spacecraft[instanceId]?.last_telemetry_at)}</strong></div></div><div className="event-list">{events.length === 0 ? <EmptyState icon={<RadioTower size={18} />} title="Waiting for TM" detail="Events are replayed from the server cursor." /> : events.map((event) => <div className="event-row" key={event.event_id}><span className={`event-severity ${classForStatus(event.severity)}`} aria-label={event.severity}><span className="status-dot" /></span><div><strong>{event.event_name.replaceAll("_", " ")}</strong><span>{typeof event.message === "string" ? event.message : JSON.stringify(event.message)}</span></div><time>{formatTime(event.server_time)}</time></div>)}</div></div>
  </section>;
}

function ProductPanel({ state, onDownload }: { state: AppState; onDownload: (product: ProductLifecycle) => void }) {
  const products = Object.values(state.products).slice(0, 4);
  return <section className="panel product-panel"><SectionTitle icon={<PackageCheck size={18} />} eyebrow="APID 3 / VERIFIED STORE" title="Products & transfers" action={<span className="panel-caption">ground authority</span>} />
    {products.length === 0 ? <EmptyState icon={<SquareStack size={20} />} title="No product downlink" detail="Verified bundles will become available after FilePacket completion." /> : <div className="product-list">{products.map((product) => <div className="product-row" key={productKey(product.product_ref)}><div className="product-mark"><PackageCheck size={17} /></div><div className="product-copy"><strong>Product {product.product_ref.product_id}</strong><span className="mono">{short(product.product_ref.spacecraft_instance_id)} / boot {product.product_ref.origin_boot_id}</span><span>{product.bundle_size?.toLocaleString() ?? "--"} bytes / {product.artifacts?.length ?? 0} verified artifacts</span></div><div className="product-progress"><StatusPill value={product.state} /><div className="progress-track"><span style={{ width: `${Math.min(100, (product.progress_bp ?? 0) / 100)}%` }} /></div><small>{product.gap_count ?? 0} gaps / {product.checksum_status ?? "pending"}</small></div><div className="product-actions">{product.state === "PUBLISHED" && product.verified && <IconButton label="Download verified product" onClick={() => onDownload(product)}><Download size={17} /></IconButton>}<IconButton label="Open product details"><ChevronRight size={17} /></IconButton></div></div>)}</div>}
  </section>;
}

export default function App() {
  const state = useMissionState();
  const instanceId = (Object.keys(state.spacecraft)[0] as U64 | undefined) ?? DEMO_INSTANCE;
  const instance = state.spacecraft[instanceId];
  const [selectedKey, setSelectedKey] = useState<string | null>(state.catalogs[instanceId]?.sceneKeys[0] ?? null);
  const selectedScene = selectedKey ? state.scenes[selectedKey] ?? null : null;
  const [viewerMode, setViewerMode] = useState<"pan" | "select">("pan");
  const [showMask, setShowMask] = useState(false);
  const [roi, setRoi] = useState<ROI | null>(selectedScene ? defaultRoi(selectedScene) : null);
  const [draftConfig, setDraftConfig] = useState<ConfigSnapshot>(state.configs[instanceId] ?? { config_epoch: 1, config_revision: 0, model_threshold_bp: 4200, coverage_limit_bp: 6500 });
  const [configSaving, setConfigSaving] = useState(false);
  const [configNotice, setConfigNotice] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [deliveryMode, setDeliveryMode] = useState<"immediate" | "next_contact">("immediate");
  const [analysisIdempotencyKey, setAnalysisIdempotencyKey] = useState("");
  const [commandPending, setCommandPending] = useState(false);
  const [commandError, setCommandError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const previousScene = useRef<string | null>(null);
  const [clock, setClock] = useState(Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => setClock(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const nextKey = state.catalogs[instanceId]?.sceneKeys[0] ?? null;
    if (!selectedKey || !state.scenes[selectedKey] || !selectedKey.startsWith(`${instanceId}:`)) setSelectedKey(nextKey);
  }, [instanceId, selectedKey, state.catalogs, state.scenes]);

  useEffect(() => {
    if (!selectedScene || previousScene.current === selectedKey) return;
    previousScene.current = selectedKey;
    setRoi(defaultRoi(selectedScene));
    setDraftConfig(state.configs[instanceId] ?? draftConfig);
    setConfigNotice(null);
  }, [instanceId, selectedKey, selectedScene, state.configs]);

  const bounds = selectedScene ? { width: selectedScene.shape[1], height: selectedScene.shape[0], minPatchSize: 256 } : null;
  const roiValidation = roi && bounds ? validateRoi(roi, bounds) : { valid: false, errors: {} };
  const selectedProduct = selectedScene?.active_preview_product_ref ? state.products[productKey(selectedScene.active_preview_product_ref)] : undefined;
  const config = state.configs[instanceId] ?? draftConfig;
  const configDirty = draftConfig.model_threshold_bp !== config.model_threshold_bp || draftConfig.coverage_limit_bp !== config.coverage_limit_bp;
  const staleTm = instance?.last_telemetry_at != null && clock - instance.last_telemetry_at > (instance.tm_stale_after_seconds ?? 5) * 1000;

  const submitCommand = async (body: CommandBody, idempotencyKey: string, label: string) => {
    const accepted = await api.postCommand(body, idempotencyKey);
    const command: CommandLifecycle = {
      request_key: accepted.request_key,
      target_spacecraft_instance_id: accepted.target_spacecraft_instance_id,
      opcode: body.opcode,
      command_state: accepted.command_state,
      outbox_state: accepted.outbox_state,
      delivery_mode: body.delivery_mode,
      effective_expires_at: accepted.effective_expires_at,
      accepted_at: accepted.accepted_at,
      updated_at: accepted.accepted_at,
    };
    store.addCommand(command);
    setNotice(`${label} admitted / RequestKey ${short(accepted.request_key.ground_instance_id)}/${accepted.request_key.request_id}`);
    return accepted;
  };

  const saveConfig = async () => {
    if (configSaving) return;
    setConfigSaving(true);
    setConfigNotice(null);
    try {
      await submitCommand({
        target_spacecraft_instance_id: instanceId,
        opcode: OPCODES.CLOUD_SET_CONFIG,
        payload: {
          expected_config_epoch: config.config_epoch,
          expected_config_revision: config.config_revision,
          model_threshold_bp: draftConfig.model_threshold_bp,
          coverage_limit_bp: draftConfig.coverage_limit_bp,
        },
        delivery_mode: "immediate",
      }, makeIdempotencyKey(), "Configuration update");
      setConfigNotice("CLOUD_SET_CONFIG admitted; waiting for satellite ACK to publish the next revision.");
    } catch (error) {
      setConfigNotice(error instanceof ApiError ? `${error.code}: ${error.message}` : String(error));
    } finally {
      setConfigSaving(false);
    }
  };

  const confirmAnalysis = async () => {
    if (!selectedScene || !roi || !roiValidation.valid || commandPending) return;
    setCommandPending(true);
    setCommandError(null);
    try {
      await submitCommand({
        target_spacecraft_instance_id: instanceId,
        opcode: OPCODES.ROI_REQUEST,
        payload: {
          scene_ref: selectedScene.scene_ref,
          roi,
          expected_config_epoch: config.config_epoch,
          expected_config_revision: config.config_revision,
          model_threshold_bp: config.model_threshold_bp,
          coverage_limit_bp: config.coverage_limit_bp,
        },
        delivery_mode: deliveryMode,
        ...(deliveryMode === "next_contact" ? { expires_at: new Date(Date.now() + 60 * 60 * 1000).toISOString() } : {}),
      }, analysisIdempotencyKey, "ROI analysis");
      setModalOpen(false);
    } catch (error) {
      setCommandError(error instanceof ApiError ? `${error.code}: ${error.message}` : String(error));
    } finally {
      setCommandPending(false);
    }
  };

  const requestPreview = async () => {
    if (!selectedScene) return;
    try {
      await submitCommand({ target_spacecraft_instance_id: instanceId, opcode: OPCODES.SCENE_REQUEST_PREVIEW, payload: { scene_ref: selectedScene.scene_ref }, delivery_mode: "immediate" }, makeIdempotencyKey(), "Preview request");
    } catch (error) {
      setNotice(error instanceof ApiError ? `${error.code}: ${error.message}` : String(error));
    }
  };

  const downloadProduct = (product: ProductLifecycle) => {
    const anchor = document.createElement("a");
    anchor.href = api.productDownloadUrl(product.product_ref);
    // The response supplies the canonical filename through Content-Disposition.
    anchor.download = "";
    anchor.rel = "noopener";
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setNotice(`Started download of verified product ${product.product_ref.product_id}.`);
  };

  return <div className="app-shell">
    <header className="topbar"><div className="brand-block"><span className="brand-mark"><Satellite size={21} /></span><div><span className="eyebrow">GDS // FLIGHT OPERATIONS</span><h1>Mission control</h1></div></div><div className="topbar-context"><span className="context-label">SCID</span><strong className="mono">{instance?.spacecraft_id ?? 68}</strong><span className="context-divider" /><span className="context-label">INSTANCE</span><strong className="mono">{short(instanceId, 7)}</strong><span className="context-divider" /><span className="context-label">UTC</span><strong className="mono">{new Date(clock).toISOString().slice(11, 19)}</strong></div><div className="topbar-actions"><IconButton label="Refresh mission state" onClick={() => window.location.reload()}><RefreshCw size={17} /></IconButton><button type="button" className="operator-button"><span className="operator-avatar">LO</span><span>local-operator</span><ChevronRight size={14} /></button></div></header>
    <main id="main-content" className="workspace">
      <StatusStrip state={state} instance={instance} />
      {state.runtime.browser_gds === "DEMO" && <div className="demo-banner"><CircleHelp size={16} /><span>Demo snapshot is active because the GDS API is unavailable. Commands remain routed through the API client and are not executed locally.</span><span className="mono">VITE_API_BASE_URL</span></div>}
      {notice && <div className="toast" role="status"><CheckCircle2 size={16} /><span>{notice}</span><IconButton label="Dismiss notification" onClick={() => setNotice(null)}><X size={15} /></IconButton></div>}
      <div className="command-workspace">
        <SceneCatalog state={state} instanceId={instanceId} selectedKey={selectedKey} onSelect={setSelectedKey} />
        <section className="panel viewer-panel"><SectionTitle icon={<MapPinned size={18} />} eyebrow="GDS-DOWNLINKED QUICKLOOK" title={selectedScene ? `Scene ${String(selectedScene.scene_ref.scene_id).padStart(4, "0")}` : "Quicklook viewer"} action={<div className="viewer-tools"><div className="segmented-control" role="group" aria-label="Viewer mode"><button type="button" className={viewerMode === "pan" ? "selected" : ""} onClick={() => setViewerMode("pan")}><Move size={15} />Pan</button><button type="button" className={viewerMode === "select" ? "selected" : ""} onClick={() => setViewerMode("select")}><ScanLine size={15} />Select ROI</button></div><IconButton label="Reset ROI to centered 2048 pixel window" onClick={() => selectedScene && setRoi(defaultRoi(selectedScene))} disabled={!selectedScene}><RotateCcw size={16} /></IconButton></div>} />
          {selectedScene?.capability === "VERIFIED" ? <QuicklookViewer api={api} scene={selectedScene} productRef={selectedScene.active_preview_product_ref ?? null} tilesEnabled={state.runtime.browser_gds !== "DEMO"} showMask={showMask} demoMask={state.runtime.browser_gds === "DEMO"} roi={roi} mode={viewerMode} onModeChange={setViewerMode} onRoiChange={(next) => setRoi(clampRoi(next, bounds!))} onResetRoi={() => selectedScene && setRoi(defaultRoi(selectedScene))} /> : <EmptyState icon={<Eye size={22} />} title="Quicklook unavailable" detail="Only verified scenes can expose a GDS preview tile." />}
          <div className="viewer-footer"><div><span className="detail-label">SCENE EXTENT</span><strong className="mono">{selectedScene ? `${selectedScene.shape[1].toLocaleString()} x ${selectedScene.shape[0].toLocaleString()} px` : "--"}</strong></div><div><span className="detail-label">DISPLAY PROFILE</span><strong>{String(selectedScene?.metadata.display_profile ?? "not available")}</strong></div><div><span className="detail-label">PREVIEW</span>{selectedScene?.active_preview_product_ref ? <StatusPill value={selectedProduct?.state ?? "VERIFIED"} label="available" /> : <button type="button" className="text-button" onClick={requestPreview} disabled={!selectedScene || selectedScene.capability !== "VERIFIED"}><UploadCloud size={14} />Request preview</button>}</div><label className="mask-toggle"><input type="checkbox" checked={showMask} onChange={(event) => setShowMask(event.target.checked)} disabled={state.runtime.browser_gds !== "DEMO"} /><span className="toggle-box"><Check size={12} /></span><span>cloud mask {state.runtime.browser_gds === "DEMO" ? "demo" : "artifact"}</span></label></div>
        </section>
        <aside className="panel inspector-panel" aria-label="Analysis inspector"><SectionTitle icon={<SlidersHorizontal size={18} />} eyebrow="COMMAND BUILDER" title="Analysis inspector" action={<StatusPill value={staleTm ? "STALE TM" : "READY"} />} />
          <div className="inspector-section"><div className="inspector-heading"><span>ROI window</span><span className="mono">min 256 px</span></div><NumericRoiEditor roi={roi} bounds={bounds} onChange={setRoi} /></div>
          <div className="inspector-section"><div className="inspector-heading"><span>Cloud decision thresholds</span><span className="config-id mono">e{config.config_epoch} / r{config.config_revision}</span></div><ThresholdControl label="Model threshold" valueBp={draftConfig.model_threshold_bp} onChange={(value) => setDraftConfig((current) => ({ ...current, model_threshold_bp: value }))} /><ThresholdControl label="Coverage limit" valueBp={draftConfig.coverage_limit_bp} onChange={(value) => setDraftConfig((current) => ({ ...current, coverage_limit_bp: value }))} /><div className="config-actions"><span className="config-state">{configDirty ? <><span className="pending-dot" /> unsaved atomic config</> : <><Check size={14} /> committed snapshot</>}</span><button type="button" className="button secondary compact" onClick={saveConfig} disabled={!configDirty || configSaving}>{configSaving ? <LoaderCircle size={14} className="spin" /> : <Settings2 size={14} />} Commit both</button></div>{configNotice && <div className={`config-notice ${configNotice.includes(":") ? "error" : ""}`}>{configNotice}</div>}</div>
          <div className="inspector-section command-section"><div className="inspector-heading"><span>Mission action</span><span className="mono">ROI_REQUEST</span></div><div className="assurance-row"><span><Cloud size={15} /> model</span><strong>{instance?.model_release_id ?? "unknown"}</strong></div><div className="assurance-row"><span><ShieldAlert size={15} /> assurance</span><StatusPill value={instance?.model_assurance ?? "unknown"} /></div><div className="warning-list">{state.catalogs[instanceId]?.status.stale && <div><AlertTriangle size={14} />Catalog replica is stale.</div>}{staleTm && <div><TimerReset size={14} />Telemetry age {formatAge(instance?.last_telemetry_at)}; backend remains authority.</div>}{instance?.contact === "BLACKOUT" && <div><RadioTower size={14} />Scheduled blackout active.</div>}</div><button type="button" className="button primary full-width" disabled={!selectedScene || selectedScene.capability !== "VERIFIED" || !roiValidation.valid} onClick={() => { setCommandError(null); setAnalysisIdempotencyKey(makeIdempotencyKey()); setModalOpen(true); }}><Send size={16} />Preview & transmit</button><small className="form-hint">The browser submits one durable command; it never allocates a RequestKey.</small></div>
        </aside>
      </div>
      <div className="lower-workspace"><LifecyclePanel state={state} /><TelemetryPanel state={state} instanceId={instanceId} /><ProductPanel state={state} onDownload={downloadProduct} /></div>
      <footer className="workspace-footer"><span><Wifi size={14} /> local_sil / host loopback</span><span><Link2Off size={14} /> no external network exposure</span><span><Database size={14} /> SQLite cursor {short(state.last_event_id)}</span><span className="footer-right">F Prime v4.1.0 / APID 0 / 1 / 2 / 3 / TM 1024 B</span></footer>
    </main>
    {modalOpen && selectedScene && roi && <CommandModal instanceId={instanceId} scene={selectedScene} roi={roi} config={config} idempotencyKey={analysisIdempotencyKey} contact={state.runtime.gds_satellite} faultProfile={state.runtime.fault_profile} catalogStale={Boolean(state.catalogs[instanceId]?.status.stale)} tmStale={staleTm} deliveryMode={deliveryMode} setDeliveryMode={setDeliveryMode} onClose={() => setModalOpen(false)} onConfirm={confirmAnalysis} pending={commandPending} error={commandError} />}
  </div>;
}
