# Expansion Security Controls

## 1. Multi-Tenant Authorization

- Request contexts carry authenticated identity and tenant scope.
- `MultiUserManager` enforces ownership boundaries before any state mutation or telemetry query.

## 2. Live Mode Interlocks

- Live trading requires explicit two-factor confirmation:
  1. `ExecutionMode.LIVE` specified in runtime configuration.
  2. `ALLOW_LIVE_REAL_MONEY_TRADING=true` environment flag actively exported.
- Violation triggers immediate process termination via `ModeGuard`.

## 3. Secret Protection

- Multi-user API credentials are encrypted at rest.
- Redaction filters prevent private keys or tokens from appearing in telemetry or log streams.
