# GDS Mission Control Web

React/TypeScript operations UI for the CCSDS satellite SIL. The browser only
consumes the framework-neutral routes exposed by `gds.api.GDSApi`; it does not
import `sat_ai`, read satellite scene files, or allocate mission `RequestKey`s.

## Run

```powershell
cd gds/web
npm install
npm run dev
```

The Vite server binds to `127.0.0.1` and chooses port `4173` (or the next free
port). Set `VITE_API_BASE_URL` when the GDS HTTP adapter is served elsewhere.
`VITE_WS_BASE_URL` is optional; when omitted, the WebSocket uses the current
origin and `/ws/telemetry`.

When the API is unavailable, the app renders a clearly labelled, bounded demo
snapshot so the operator workflow can be inspected without pretending that a
command was admitted. Command buttons still call the API client and never run
inference locally.

## Checks

```powershell
npm test -- --run
npm run build
```

The frontend tests cover pixel ROI rounding/clamping, normalized state event
deduplication and the 1000-event/4 MiB realtime buffer contract.
