# WebSocket Reconnection and Session Recovery

## 1. Disconnection Protocol

The moment a WebSocket connection drop or heartbeat timeout (> 15s) is detected:
1. Mark market data state as `DISCONNECTED`.
2. Block all new signal generation and order placement.
3. Preserve last known valid market tick timestamp.
4. Prevent in-flight order executions.

## 2. Reconnection Workflow

```
DISCONNECT DETECTED
       │
       ▼
HALT NEW TRADING DECISIONS
       │
       ▼
BACKOFF RETRY RECONNECT
       │
       ▼
RE-AUTHENTICATE & RESUBSCRIBE
       │
       ▼
VALIDATE INCOMING STREAM
       │
       ▼
DETECT TIME GAP (t_now - t_last)
       │
       ├─── Gap <= 65s ───► RESUME LIVE TRADING
       │
       └─── Gap > 65s ────► TRIGGER GAP RECOVERY
```

Merely re-establishing the socket connection is NOT sufficient proof that the market state is ready for live trading. Market data gap analysis is mandatory.
