# Disaster Recovery and Restore Verification

## 1. 5-Point Backup Verification Gates

A backup archive is never assumed recoverable merely because the creation command succeeded. `BackupVerifier` executes five verification gates:

1. **Existence Check**: Archive file exists on disk and is non-empty.
2. **Manifest Consistency**: Paired `.manifest.json` exists and parses cleanly.
3. **SHA-256 Checksum Integrity**: Recomputed archive checksum matches manifest byte-for-byte.
4. **Archive Readability**: `zipfile.testzip()` executes cleanly with zero CRC corruption.
5. **Inventory Match**: Every file declared in the manifest exists within the archive.

## 2. Restore Testing Protocol

Periodic automated restore tests execute the following cycle:
```
BACKUP ARCHIVE
      │
      ▼
VERIFY CHECKSUM & MANIFEST
      │
      ▼
UNPACK INTO ISOLATED SANDBOX
      │
      ▼
VERIFY EXTRACTED FILE SHA-256 HASHES
      │
      ▼
SIMULATE SERVICE STATE RESTORATION
```

If any step fails, the backup is marked INVALID, and a CRITICAL alert is dispatched.

## 3. Disaster Recovery Runbooks for Infrastructure Loss

| Scenario | Detection | Recovery Action | Validation |
| :--- | :--- | :--- | :--- |
| **A. Database Loss** | Connection refused; table missing. | Restore latest verified DB backup; replay audit trail; flush in-memory buffer. | Query row count against manifest; run test SELECT. |
| **B. Redis Loss** | Connection timeout; pub/sub dead. | Restart Redis daemon; restore cache snapshot or run direct feature computation. | Verify PING responds with PONG. |
| **C. Machine Failure** | Host unreachable; hardware fault. | Provision replacement host; unpack full backup package; verify secret environment. | Execute readiness check before opening live socket. |
| **D. Model Loss** | Missing `.pkl` / `.bin` weights. | Retrieve model weights from model artifact backup; verify hash against ModelRegistry. | Pass test tensor through model inference. |
