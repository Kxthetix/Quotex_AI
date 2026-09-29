# Security and Secret Management

## 1. Zero-Hardcoded Secrets Policy

- No plaintext API keys, secret tokens, or database passwords are ever committed to Git or hardcoded in source files.
- Secrets are loaded exclusively from environment variables or secure vault injection.

## 2. Startup Secret Validation

During application startup:
1. `SecretManager.validate_required_secrets()` checks for mandatory variables:
   - `TRADING_DATABASE_URL`
   - `TRADING_REDIS_URL`
   - `MARKET_DATA_API_KEY`
2. If running under `TRADING_ENV=production` and any secret is missing:
   - Startup is immediately BLOCKED.
   - A `SecretValidationError` is raised.

## 3. Secret Masking and Sanitization

1. **Token Masking**: `mask_secret("very_secret_api_key_12345")` produces `ve****45`.
2. **URI Password Sanitization**:
   - Input: `postgresql://trader_user:super_secret_pw@db.prod.internal:5432/trading_db`
   - Output: `postgresql://trader_user:***@db.prod.internal:5432/trading_db`
3. **Structured Log & Audit Redaction**:
   - JSON loggers automatically scan and replace sensitive keywords (`api_key`, `secret`, `password`, `token`, `bearer`, `auth`) with `***REDACTED***`.
