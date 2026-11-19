# Company Pilot Deployment Runbook

This runbook is a phase-1 guide for the partner's OT/IT, maintenance, security, and ML teams. The repository is a research prototype: it does not currently configure TLS/mTLS, secure aggregation, differential privacy, or federated normalization. Do not connect it to production control networks or use its RUL output as an automatic shutdown command. Close the go/no-go items below in a segregated pilot environment first.

## 1. Entry criteria and ownership

Name one owner each from OT, IT/networking, maintenance/reliability, ML, and security/privacy. Record the pilot assets, approved data fields, maintenance decision being supported, rollback owner, and success measures before deployment. A useful phase-1 scope is two or more representative assets, advisory-only alerts, and a fixed observation window that includes planned inspections or known maintenance events.

## 2. Hardware and network prerequisites

- **Edge gateways:** one 64-bit Linux gateway per participating site or data boundary, with Python 3.11, reliable storage for local telemetry and logs, time synchronization, and enough CPU/RAM to pass the edge profile below. Keep the gateway outside the safety-control path. Use service accounts, disk encryption, secure boot where available, patching, and a tested replacement/rollback image.
- **Coordinator:** a managed host reachable by every gateway. The current Flower server binds to TCP `8081` by default. Permit gateway-initiated outbound TCP only to the coordinator's approved address and gRPC port; do not allow general internet egress or inbound sessions to gateways. Confirm DNS, proxy/NAT behavior, connection timeouts, bandwidth, and clock synchronization with IT.
- **TLS/mTLS:** the checked-in `server.py` and `client.py` start Flower with TLS disabled. Before any plant-network pilot, provision a private CA, a coordinator server certificate with the correct SANs, and a unique client certificate/key per gateway. Store private keys in the site secret store or TPM, define rotation and revocation, restrict file permissions, and verify certificate expiry monitoring. TLS/mTLS support must be added and tested at both Flower endpoints, or an approved mTLS proxy must terminate and re-establish the protected connection. A packet capture showing encrypted traffic and rejected unknown/expired clients is a go-live requirement.
- **Data boundary:** keep raw sensor history on the gateway. Allow only the reviewed model-update and aggregate diagnostic fields across the boundary. Model updates can still leak information; mTLS protects transport but is not a formal privacy guarantee.

## 3. Phase-1 implementation checklist

### A. Map plant telemetry to model inputs

1. Inventory each tag's source, asset, engineering unit, sample interval, quality code, retention, and owner. Support an isolated read-only `OPC-UA` subscription, `MQTT` topic, or staged CSV export; never write back to PLCs/DCS from this pilot.
2. Produce a version-controlled mapping to the exact `USEFUL_SENSORS` order:

   `setting1`, `setting2`, `setting3`, `s2`, `s3`, `s4`, `s7`, `s8`, `s9`, `s11`, `s12`, `s13`, `s14`, `s15`, `s17`, `s20`, `s21`.

   C-MAPSS names are placeholders, not a universal plant schema. The reliability engineer must document the physical meaning and unit conversion for every mapped channel. Do not silently substitute a missing tag.
3. Define resampling, timestamp alignment, late/out-of-order handling, bad-quality filtering, gap limits, and the behavior for stale or constant signals. Validate a 30-sample window end to end and compare the gateway output with a reviewed offline CSV fixture.
4. Reject inference when required channels, units, ordering, or data quality are invalid. Log metadata and reason codes without copying raw telemetry into central logs.

### B. Establish federated normalization

The checked-in `global_scale.json` was computed centrally and is not federated. Treat replacement normalization as a required pilot integration, not an existing capability.

1. Fit normalization only on an agreed training period, excluding held-out evaluation periods and known sensor faults. Freeze a version for the pilot.
2. Have each gateway compute local per-channel count, minimum/maximum, and preferably robust quantiles. Aggregate only approved statistics through the protected channel; apply minimum-cohort and disclosure rules from the privacy review. If cross-site statistics are not approved, use per-site normalization and evaluate the resulting site shift explicitly.
3. Publish one signed normalization manifest containing channel order, units, statistic definitions, fitting interval, participating sites, clipping policy, version, and hash. Install the identical approved version on coordinator and gateways where a global transform is used.
4. Test constant channels, outliers, unit changes, and values outside the fitting range. Monitor drift without automatically refitting; require change control and a rollback copy for every new manifest.

### C. Calibrate loss and maintenance alerts

1. Start with the repository's `AsymmetricRULLoss` baseline, `alpha_late=2.5` and `alpha_early=0.5`. Here a late prediction overestimates remaining life and should cost more, so require `alpha_late > alpha_early > 0`.
2. With time-ordered, asset-separated validation data, sweep a small predeclared grid of `(alpha_late, alpha_early)`. Compare RMSE, NASA score, late-prediction rate, alert lead time, missed actionable failures, and nuisance alerts. Do not tune on the final acceptance period.
3. Translate maintenance SLAs into thresholds. If planning needs 14 days and execution needs 3 days, the warning threshold must provide at least 17 days plus an agreed uncertainty/data-latency margin. Define warning, critical, hysteresis, persistence, acknowledgement, and missing-data behavior with maintenance—not from model metrics alone.
4. Run alerts in shadow/advisory mode. A human owns work-order creation and shutdown decisions. Approve thresholds only after reviewing performance by asset type, operating regime, site, and failure mode; document unsupported regimes.

### D. Profile each target gateway

Run the profile on every target hardware/software image with the pilot archive held locally:

```bash
python3 experiments/profile_edge.py \
  --data /secure/local/CMAPSSData.zip \
  --dataset FD002 \
  --partition regime \
  --client 0 \
  --rounds 5 \
  --epochs 2 \
  --output /tmp/fedpdm-edge-profile.json
```

Capture CPU model, cores/limits, RAM, storage, OS/kernel, Python/PyTorch versions, thermal state, and whether other gateway workloads were active. Review round time, peak RSS, thermal throttling, storage growth, restart behavior, and tensor payload. The reported communication payload excludes gRPC/TLS framing, and `mTLS` in the current profile is `false`; measure encrypted wire traffic separately. Set resource limits and an abort threshold before the shadow run.

### E. Security and privacy sign-off

The security/privacy owner must sign the following evidence before plant connection:

- architecture and data-flow diagram; asset inventory; threat model; vendor/dependency and vulnerability review;
- firewall allow-list showing outbound gateway-to-coordinator access only; mTLS certificate issuance, storage, rotation, revocation, and negative-client tests;
- packet capture confirming encryption; authentication/authorization and audit-log review; secrets absent from source, images, and logs;
- retention/deletion rules for raw telemetry, model checkpoints, updates, metrics, and backups; incident response and gateway/coordinator rollback drill;
- review of model-update leakage risk and an explicit decision on secure aggregation or differential privacy;
- preflight evidence from `python3 scripts/preflight_check.py --with-cmapss`, including zero repository symlinks, protected artifact hashes, test results, dashboard conservation, and diagnostics network-audit invariants.

The committed diagnostics result reports zero socket calls and zero raw sensor bytes for the instrumented diagnostics path only. It is not proof that the whole system sends no data and must not replace network testing.

## 4. Pilot execution and acceptance

1. **Lab:** replay approved data, test malformed/missing inputs, certificate failure, coordinator outage, disk pressure, restart, rollback, and alert routing.
2. **Shadow plant run:** ingest read-only telemetry and suppress automated maintenance actions. Review daily connectivity/data-quality health and weekly model/alert behavior with OT and maintenance.
3. **Acceptance:** require the agreed alert lead-time and nuisance-alert targets, no unresolved critical security findings, stable gateway resource use, successful certificate rotation and rollback, and signed OT/IT/ML/security evidence.
4. **Stop conditions:** isolate the gateway on unexpected outbound traffic, certificate/authentication bypass, incorrect units or channel order, persistent data-quality failure, resource interference with OT workloads, or unsafe/misleading alerts. Preserve logs, revert the deployment image/configuration, and return to the last approved advisory process.

At phase-1 close, document what was measured, what remains unvalidated, dataset and configuration versions, deviations, incidents, and the explicit decision to stop, extend shadow mode, or design a controlled production integration.
