import type {
  CatalogStatus,
  CommandBody,
  EventEnvelope,
  ProductRef,
  Scene,
  SnapshotEnvelope,
  U64,
} from "../types";
import { productKey } from "../types";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly body: Record<string, unknown>;

  constructor(status: number, body: Record<string, unknown>) {
    super(String(body.message ?? body.error ?? `Request failed (${status})`));
    this.name = "ApiError";
    this.status = status;
    this.code = String(body.error ?? "HTTP_ERROR");
    this.body = body;
  }
}

export interface CatalogResponse {
  catalog: CatalogStatus;
  scenes: Scene[];
  next_cursor: number | null;
}

export interface AcceptedCommand {
  status: "accepted";
  request_key: { ground_instance_id: U64; request_id: number };
  target_spacecraft_instance_id: U64;
  effective_expires_at: string;
  command_state: string;
  outbox_state: string;
  mission_digest: string;
  http_idempotency_digest: string;
  accepted_at: string;
  replayed: boolean;
}

export class GDSApiClient {
  readonly baseUrl: string;

  constructor(baseUrl = import.meta.env.VITE_API_BASE_URL || "") {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  private url(path: string): string {
    return `${this.baseUrl}${path}`;
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(this.url(path), {
      credentials: "same-origin",
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...(init?.headers ?? {}),
      },
    });
    const text = await response.text();
    let body: Record<string, unknown> = {};
    if (text) {
      try {
        body = JSON.parse(text) as Record<string, unknown>;
      } catch {
        body = { message: text };
      }
    }
    if (!response.ok) throw new ApiError(response.status, body);
    return body as T;
  }

  async getState(): Promise<SnapshotEnvelope> {
    return this.request<SnapshotEnvelope>("/api/state");
  }

  async getCatalog(instance: U64, afterSceneId?: number): Promise<CatalogResponse> {
    const query = new URLSearchParams({ limit: "100" });
    if (afterSceneId != null) query.set("after_scene_id", String(afterSceneId));
    const response = await this.request<{ catalog: CatalogStatus; scenes: Scene[]; next_cursor: number | null }>(
      `/api/spacecraft/${instance}/scenes?${query.toString()}`,
    );
    return response;
  }

  async getScene(instance: U64, scene: Scene["scene_ref"]): Promise<Scene> {
    const response = await this.request<{ scene: Scene }>(
      `/api/spacecraft/${instance}/scenes/${scene.catalog_epoch}/${scene.scene_id}/${scene.scene_revision}`,
    );
    return response.scene;
  }

  async postCommand(body: CommandBody, idempotencyKey: string): Promise<AcceptedCommand> {
    return this.request<AcceptedCommand>("/api/commands", {
      method: "POST",
      headers: { "Idempotency-Key": idempotencyKey },
      body: JSON.stringify(body),
    });
  }

  async getCommand(key: { ground_instance_id: U64; request_id: number }): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>(`/api/commands/${key.ground_instance_id}/${key.request_id}`);
  }

  async getProduct(ref: ProductRef): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>(
      `/api/products/${ref.spacecraft_instance_id}/${ref.origin_boot_id}/${ref.product_id}`,
    );
  }

  productDownloadUrl(ref: ProductRef): string {
    return this.url(`/api/products/${ref.spacecraft_instance_id}/${ref.origin_boot_id}/${ref.product_id}/download`);
  }

  tileUrl(ref: ProductRef, z: number, x: number, y: number): string {
    return this.url(`/api/products/${ref.spacecraft_instance_id}/${ref.origin_boot_id}/${ref.product_id}/tiles/${z}/${x}/${y}`);
  }

  websocketUrl(lastEventId?: U64): string {
    const configured = import.meta.env.VITE_WS_BASE_URL as string | undefined;
    const base = configured || this.baseUrl || window.location.origin;
    const parsed = new URL(base);
    parsed.protocol = parsed.protocol === "https:" ? "wss:" : "ws:";
    parsed.pathname = "/ws/telemetry";
    parsed.search = lastEventId ? new URLSearchParams({ last_event_id: lastEventId }).toString() : "";
    return parsed.toString();
  }

  async openRealtime(lastEventId?: U64): Promise<EventEnvelope> {
    return this.request<EventEnvelope>(`/ws/telemetry${lastEventId ? `?last_event_id=${lastEventId}` : ""}`);
  }
}

export function makeIdempotencyKey(): string {
  if (crypto.randomUUID) return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function productPath(ref: ProductRef): string {
  return productKey(ref);
}
