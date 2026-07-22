import type { EventEnvelope, EventRecord, SnapshotEnvelope, U64 } from "../types";
import type { GDSApiClient } from "./client";

export type RealtimeStatus = "IDLE" | "CONNECTING" | "CONNECTED" | "RECONNECTING" | "RESYNC_REQUIRED" | "CLOSED";

const U64_CURSOR = /^[0-9a-f]{16}$/;

/** Parse the wire U64 form without ever routing it through JavaScript Number. */
export function parseU64Cursor(value: U64, label = "event cursor"): bigint {
  if (typeof value !== "string" || !U64_CURSOR.test(value)) {
    throw new Error(`${label} must be a 16-digit lowercase hexadecimal U64`);
  }
  // Bare strings containing a-f are not valid BigInt decimal literals.
  return BigInt(`0x${value}`);
}

export function compareU64Cursor(left: U64, right: U64): number {
  const leftValue = parseU64Cursor(left, "left cursor");
  const rightValue = parseU64Cursor(right, "right cursor");
  return leftValue === rightValue ? 0 : leftValue < rightValue ? -1 : 1;
}

function encodedEventBytes(event: EventRecord): number {
  return new TextEncoder().encode(JSON.stringify(event)).byteLength;
}

export class BoundedEventBuffer {
  private readonly maxEvents: number;
  private readonly maxBytes: number;
  private events: Array<{ event: EventRecord; bytes: number }> = [];
  private bytes = 0;

  constructor(maxEvents = 1000, maxBytes = 4 * 1024 * 1024) {
    this.maxEvents = maxEvents;
    this.maxBytes = maxBytes;
  }

  push(event: EventRecord): boolean {
    const bytes = encodedEventBytes(event);
    if (this.events.length >= this.maxEvents || this.bytes + bytes > this.maxBytes) return false;
    this.events.push({ event, bytes });
    this.bytes += bytes;
    return true;
  }

  get size(): number {
    return this.events.length;
  }

  get byteLength(): number {
    return this.bytes;
  }

  clear(): void {
    this.events = [];
    this.bytes = 0;
  }

  consume(eventId: U64): void {
    const index = this.events.findIndex(({ event }) => event.event_id === eventId);
    if (index < 0) return;
    const [entry] = this.events.splice(index, 1);
    this.bytes -= entry.bytes;
  }
}

interface RealtimeHandlers {
  onStatus?: (status: RealtimeStatus) => void;
  onSnapshot?: (snapshot: SnapshotEnvelope) => void;
  onEvent?: (event: EventRecord) => void;
  onResync?: () => U64 | undefined | Promise<U64 | undefined>;
  onError?: (error: Event | Error) => void;
}

/** Cursor-based WebSocket client with bounded replay memory and exponential reconnect. */
export class RealtimeClient {
  private socket: WebSocket | null = null;
  private stopped = true;
  private attempt = 0;
  private reconnectTimer: number | undefined;
  private lastEventId: U64 | undefined;
  private readonly buffer = new BoundedEventBuffer();
  private readonly recentEventIds = new Set<string>();
  private readonly recentEventOrder: string[] = [];
  private resyncInFlight = false;

  constructor(private readonly api: GDSApiClient, private readonly handlers: RealtimeHandlers) {}

  start(lastEventId?: U64): void {
    this.stopped = false;
    this.lastEventId = lastEventId === undefined ? undefined : this.checkedCursor(lastEventId);
    this.connect();
  }

  get pendingEventCount(): number {
    return this.buffer.size;
  }

  get pendingEventBytes(): number {
    return this.buffer.byteLength;
  }

  get cursor(): U64 | undefined {
    return this.lastEventId;
  }

  stop(): void {
    this.stopped = true;
    if (this.reconnectTimer !== undefined) window.clearTimeout(this.reconnectTimer);
    this.reconnectTimer = undefined;
    this.socket?.close(1000, "client stopped");
    this.socket = null;
    this.handlers.onStatus?.("CLOSED");
  }

  private connect(): void {
    if (this.stopped) return;
    this.handlers.onStatus?.(this.attempt === 0 ? "CONNECTING" : "RECONNECTING");
    try {
      this.socket = new WebSocket(this.api.websocketUrl(this.lastEventId));
    } catch (error) {
      this.handlers.onError?.(error instanceof Error ? error : new Error(String(error)));
      this.scheduleReconnect();
      return;
    }
    this.socket.onopen = () => {
      this.attempt = 0;
      this.handlers.onStatus?.("CONNECTED");
    };
    this.socket.onmessage = (message) => this.receive(message.data);
    this.socket.onerror = (error) => this.handlers.onError?.(error);
    this.socket.onclose = () => {
      this.socket = null;
      if (!this.stopped) this.scheduleReconnect();
    };
  }

  /** Accept a decoded browser WebSocket message. Public for deterministic tests. */
  receive(raw: unknown): void {
    let envelope: EventEnvelope;
    try {
      envelope = typeof raw === "string" ? JSON.parse(raw) as EventEnvelope : raw as EventEnvelope;
    } catch {
      this.handlers.onError?.(new Error("Realtime message is not valid JSON"));
      return;
    }
    if (envelope.error === "RESYNC_REQUIRED") {
      this.handlers.onStatus?.("RESYNC_REQUIRED");
      void this.resync();
      return;
    }
    if (envelope.snapshot) {
      try {
        const cursor = envelope.snapshot.last_event_id ?? envelope.snapshot.as_of_event_id;
        if (cursor !== undefined) this.lastEventId = this.checkedCursor(cursor);
        this.handlers.onSnapshot?.(envelope.snapshot);
      } catch (error) {
        this.handlers.onError?.(error instanceof Error ? error : new Error(String(error)));
        this.handlers.onStatus?.("RESYNC_REQUIRED");
        void this.resync();
        return;
      }
    }
    for (const event of envelope.events ?? []) this.dispatch(event);
    if (envelope.event) this.dispatch(envelope.event);
  }

  private dispatch(event: EventRecord): void {
    let eventId: U64;
    try {
      eventId = this.checkedCursor(event.event_id);
    } catch (error) {
      this.handlers.onError?.(error instanceof Error ? error : new Error(String(error)));
      this.handlers.onStatus?.("RESYNC_REQUIRED");
      void this.resync();
      return;
    }
    if (this.recentEventIds.has(eventId)) return;
    this.recentEventIds.add(eventId);
    this.recentEventOrder.push(eventId);
    while (this.recentEventOrder.length > 2048) {
      const oldest = this.recentEventOrder.shift();
      if (oldest !== undefined) this.recentEventIds.delete(oldest);
    }
    if (!this.buffer.push(event)) {
      this.handlers.onError?.(new Error("Realtime client buffer exceeded 1000 events or 4 MiB; resync required"));
      this.handlers.onStatus?.("RESYNC_REQUIRED");
      void this.resync();
      return;
    }
    if (this.lastEventId === undefined || compareU64Cursor(eventId, this.lastEventId) > 0) {
      this.lastEventId = eventId;
    }
    try {
      this.handlers.onEvent?.(event);
    } finally {
      this.buffer.consume(event.event_id);
    }
  }

  private async resync(): Promise<void> {
    if (this.resyncInFlight) return;
    this.resyncInFlight = true;
    try {
      const refreshedCursor = await this.handlers.onResync?.();
      this.buffer.clear();
      // Never reconnect with a stale cursor. When the snapshot exposes one,
      // use that exact durable boundary instead of replaying from the start.
      this.lastEventId = refreshedCursor === undefined ? undefined : this.checkedCursor(refreshedCursor);
      this.closeForResync("resynced");
    } catch (error) {
      this.handlers.onError?.(error instanceof Error ? error : new Error(String(error)));
      this.lastEventId = undefined;
      this.closeForResync("RESYNC_REQUIRED");
    } finally {
      this.resyncInFlight = false;
    }
  }

  private checkedCursor(value: U64): U64 {
    parseU64Cursor(value);
    return value;
  }

  private closeForResync(reason: string): void {
    if (this.socket !== null) {
      this.socket.close(4009, reason);
    } else {
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.stopped || this.reconnectTimer !== undefined) return;
    const delay = Math.min(30_000, 500 * 2 ** Math.min(this.attempt, 6));
    this.attempt += 1;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = undefined;
      this.connect();
    }, delay);
  }
}
