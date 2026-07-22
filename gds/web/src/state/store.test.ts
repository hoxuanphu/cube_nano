import { describe, expect, it } from "vitest";
import { NormalizedStore } from "./store";
import { createDemoState } from "./demo";

describe("normalized mission store", () => {
  it("keeps server entities instance-scoped and applies cursor events once", () => {
    const store = new NormalizedStore(createDemoState());
    const event = {
      event_id: "0000000000000040",
      event_name: "ANALYSIS_PROGRESS",
      severity: "INFO",
      message: { progress_bp: 7800, stage: "INFERENCE" },
      server_time: "2026-07-20T10:12:24Z",
      source_spacecraft_instance_id: "0000000000000001",
      source_boot_id: 42,
      request_key: { ground_instance_id: "4f9a2c71d6e80b13", request_id: 41 },
    } as const;
    store.applyEvent(event);
    store.applyEvent(event);
    const state = store.getState();
    expect(state.last_event_id).toBe("0000000000000040");
    expect(state.events.filter((item) => item.event_id === event.event_id)).toHaveLength(1);
    expect(state.commands["4f9a2c71d6e80b13:41"].progress_bp).toBe(7800);
    expect(state.spacecraft["0000000000000001"].instance_id).toBe("0000000000000001");
  });

  it("preserves local normalized entities when a partial snapshot omits them", () => {
    const store = new NormalizedStore(createDemoState());
    store.replaceSnapshot({ state: { runtime: { browser_gds: "CONNECTED" } } });
    expect(store.getState().scenes["0000000000000001:1:1:7"]).toBeDefined();
    expect(store.getState().runtime.browser_gds).toBe("CONNECTED");
  });

  it("starts a new instance epoch on a newer satellite boot and ignores late old boot events", () => {
    const store = new NormalizedStore(createDemoState());
    store.applyEvent({
      event_id: "0000000000000041",
      event_name: "BOOT_READY",
      severity: "INFO",
      message: { state: "READY" },
      server_time: "2026-07-20T10:13:00Z",
      source_spacecraft_instance_id: "0000000000000001",
      source_boot_id: 43,
    });
    expect(store.getState().spacecraft["0000000000000001"].boot_id).toBe(43);
    expect(store.getState().events).toHaveLength(1);
    store.applyEvent({
      event_id: "0000000000000042",
      event_name: "OLD_BOOT_EVENT",
      severity: "WARNING",
      message: null,
      server_time: "2026-07-20T10:13:01Z",
      source_spacecraft_instance_id: "0000000000000001",
      source_boot_id: 42,
    });
    expect(store.getState().last_event_id).toBe("0000000000000041");
  });

  it("keeps the greatest hexadecimal U64 cursor when replay and live events arrive out of order", () => {
    const store = new NormalizedStore(createDemoState());
    store.applyEvent({
      event_id: "00000000000000af",
      event_name: "REPLAY_BOUNDARY",
      severity: "INFO",
      message: null,
      server_time: "2026-07-20T10:14:00Z",
    });
    store.applyEvent({
      event_id: "000000000000009f",
      event_name: "LATE_REPLAY",
      severity: "INFO",
      message: null,
      server_time: "2026-07-20T10:13:59Z",
    });

    expect(store.getState().last_event_id).toBe("00000000000000af");
    expect(store.getState().runtime.as_of_event_id).toBe("00000000000000af");
  });
});
