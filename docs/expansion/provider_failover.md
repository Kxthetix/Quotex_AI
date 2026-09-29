# Provider Priority and Controlled Failover

## 1. Provider Priority Tiers

Data sources are configured in priority tiers:
1. `PRIMARY` (e.g. Primary direct exchange feed)
2. `SECONDARY` (e.g. Secondary institutional market data aggregator)
3. `TERTIARY` (e.g. REST fallback or synthetic simulated stream)

Priority is managed strictly via configuration rather than hardcoded logic.

## 2. Pre-Failover Validation Gates

Before switching from a degraded Primary to a Secondary provider, the `ProviderFailoverManager` validates four criteria:
1. **Connection Health**: Secondary provider must be actively connected and responsive.
2. **Symbol Mapping**: All active instruments must have valid ticker mappings on the candidate feed.
3. **Data Freshness**: Candidate provider must have recent ticks without latency spikes.
4. **Schema Compatibility**: Incoming payload structure matches canonical expectations.

If validation fails, failover to that candidate is rejected, and the system attempts tertiary providers or enters safe degraded mode.

## 3. Cross-Provider Quote Consistency

The `CrossProviderConsistencyChecker` continuously samples contemporaneous quotes across active providers. If price divergence exceeds the threshold (e.g. 0.5%), a divergence event is recorded and logged to operational telemetry.
