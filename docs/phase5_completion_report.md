# Phase 5 completion report

Date: 2026-07-20
Status: IMPLEMENTATION COMPLETE, 14/14 P5 tasks
Scope: React/TypeScript GDS operator webapp core; local SIL API contract; no
production authentication, TLS or remote exposure.

## Delivered

| Task | Implementation | Evidence |
|---|---|---|
| P5-01 | Vite React shell, instance-scoped normalized store, local editing state | `gds/web/src/App.tsx`, `gds/web/src/state/store.ts` |
| P5-02 | Browser/GDS, contact, spacecraft, TM age and queue status strip | `gds/web/src/App.tsx` |
| P5-03 | Catalog search, capability filter, epoch/revision/stale/product context | `gds/web/src/App.tsx`, `gds/web/src/api/client.ts` |
| P5-04 | OpenLayers pixel projection, GDS-only XYZ tiles, bounded LRU tile cache and categorical mask toggle | `gds/web/src/components/QuicklookViewer.tsx`, `gds/web/src/utils/tileCache.ts` |
| P5-05..06 | Pan/select segmented control, rectangle draw/modify/translate, integer ROI editor and shared floor/ceil/clamp rules | `gds/web/src/utils/roi.ts`, `gds/web/src/components/QuicklookViewer.tsx` |
| P5-07 | Model and coverage thresholds committed by one `CLOUD_SET_CONFIG` command | `gds/web/src/App.tsx` |
| P5-08 | Full SceneRef/ROI/config/fault/contact/expiry preview; stable HTTP Idempotency-Key; RequestKey shown only after response | `gds/web/src/App.tsx` |
| P5-09..11 | Command/outbox/science/product/transfer lifecycle, telemetry/event timeline, verified product download and transfer progress | `gds/web/src/App.tsx`, `gds/web/src/state/store.ts` |
| P5-12 | Blackout/no-contact/stale/degraded warnings and persisted next-contact mode | `gds/web/src/App.tsx` |
| P5-13 | Cursor reconnect, snapshot resync, exponential backoff and 1000-event/4 MiB client bound | `gds/web/src/api/realtime.ts` |
| P5-14 | Responsive desktop/tablet/mobile layout, skip link, labels, focus states, reduced motion and icon tooltips | `gds/web/src/styles.css`, `gds/web/index.html` |

## Contract decisions

- The frontend uses the planned catalog routes under `/api/spacecraft/{instance}/scenes` and the full ProductRef tile/download routes. It does not use the older shorthand route.
- U64 values remain opaque 16-character lowercase hexadecimal strings in TypeScript. No U64 is parsed through JavaScript `Number`.
- ROI editing is local until admission. Drag uses floor for the start corner and ceil for the end corner; numeric values are validated against scene bounds and minimum patch size.
- Configuration thresholds are sent together in one `CLOUD_SET_CONFIG` payload. A stale telemetry `READY` state produces a warning but does not become a browser-side authority.
- The demo snapshot is read-only presentation data. It is explicitly labelled and never invokes inference or invents a server-issued RequestKey.

## Verification

```text
gds/web: npm test -- --run       9 passed
gds/web: npm run build           PASS
python -m pytest -q              203 passed, 19 subtests passed
python -m compileall -q gds flight link_sim scripts   PASS
GET http://127.0.0.1:4173/      200
```

The bundled production JavaScript is approximately 577 kB before gzip because
OpenLayers is included in the initial viewer chunk. This is a performance
follow-up for P6; functionality and bounded tile requests are already covered.

The in-app browser plugin could not be initialized in this environment because
its browser asset bootstrap returned a missing-path error. A local Chromium
smoke nevertheless verified desktop rendering, mobile rendering without
horizontal overflow, Select ROI mode, invalid numeric ROI rejection, the
confirmation dialog and next-contact expiry display. The full backend-backed
Playwright workflow, fault injection and reconnect E2E remain part of the P6
hardening gate.
