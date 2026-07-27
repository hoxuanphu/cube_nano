import { describe, expect, it } from "vitest";
import { BoundedEventBuffer, compareU64Cursor, parseU64Cursor, RealtimeClient } from "./realtime";
import type { GDSApiClient } from "./client";

function event(id: string) {
  return {
    event_id: id,
    event_name: "HEALTH",
    severity: "INFO",
    message: null,
    server_time: "2026-07-20T10:00:00Z",
  } as const;
}

describe("bounded realtime client buffer", () => {
  it("rejects before the event count can grow past the limit", () => {
    const buffer = new BoundedEventBuffer(2, 1024);
    expect(buffer.push(event("0000000000000001"))).toBe(true);
    expect(buffer.push(event("0000000000000002"))).toBe(true);
    expect(buffer.push(event("0000000000000003"))).toBe(false);
  });

  it("enforces the byte bound independently of event count", () => {
    const buffer = new BoundedEventBuffer(100, 80);
    expect(buffer.push({ ...event("0000000000000001"), message: "x".repeat(100) })).toBe(false);
  });

  it("parses lowercase hexadecimal U64 cursors using an explicit hexadecimal prefix", () => {
    expect(parseU64Cursor("00000000000000af")).toBe(0xafn);
    expect(compareU64Cursor("00000000000000af", "0000000000000100")).toBeLessThan(0);
    expect(() => parseU64Cursor("00000000000000AF")).toThrow(/lowercase/);
    expect(() => parseU64Cursor("af")).toThrow(/16-digit/);
  });

  it("deduplicates a replay/live boundary event and consumes it after dispatch", () => {
    const dispatched: string[] = [];
    const client = new RealtimeClient(
      { websocketUrl: () => "ws://example.invalid/ws/telemetry" } as unknown as GDSApiClient,
      { onEvent: (item) => dispatched.push(item.event_id) },
    );
    const boundary = event("00000000000000af");
    client.receive({ events: [boundary] });
    client.receive({ event: boundary });
    client.receive({ event: event("0000000000000100") });

    expect(dispatched).toEqual(["00000000000000af", "0000000000000100"]);
    expect(client.cursor).toBe("0000000000000100");
    expect(client.pendingEventCount).toBe(0);
    expect(client.pendingEventBytes).toBe(0);
  });

  it("uses the fresh snapshot cursor when recovering from RESYNC_REQUIRED", async () => {
    const client = new RealtimeClient(
      { websocketUrl: () => "ws://example.invalid/ws/telemetry" } as unknown as GDSApiClient,
      { onResync: async () => "00000000000000af" },
    );

    client.receive({ type: "error", error: "RESYNC_REQUIRED" });
    await new Promise<void>((resolve) => setTimeout(resolve, 0));

    expect(client.cursor).toBe("00000000000000af");
  });
});
