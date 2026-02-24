# EV Charging Station Anomaly Detection


## 1. Problem Understanding

The objective of this exercise is to detect anomalous behavior in EV charging station event logs using an unsupervised approach.

Each row in the dataset represents an **event** during a charging session - not a full session. This means anomalies may exist at multiple levels simultaneously:

- **Event level** - e.g., an abnormal voltage spike in a single reading
- **Session level** - e.g., an unusually high error rate across a session
- **Station level** - e.g., one charger behaving differently from its peers
- **Temporal level** - e.g., failures concentrated at certain hours

Given the synthetic nature of the dataset and the presence of injected faults, the following assumptions were made:

- The majority of data represents normal operating behavior
- Anomalies are rare (estimated 2–5% of events)
- No reliable ground-truth labels exist for training
- Some fault patterns are explicitly knowable from domain rules (e.g., non-zero error codes, physically impossible power values)

This frames the problem as **unsupervised anomaly detection**, supplemented by a deterministic rule layer for known fault signatures. The goal is not only to flag anomalies accurately, but to build a system that is interpretable, scalable, robust to noise, and operationally practical for NOC use.

---

## 2. Key Insights from EDA

### 2.1 Feature Distributions

The numeric signals exhibit largely normal distributions under healthy operating conditions, but with meaningful outlier tails:

- **Voltage** is centered around ~228 V, but includes extreme readings below 90 V and above 390 V — well outside any reasonable operating range
- **Current** clusters around ~30 A but includes near-zero values inconsistent with active charging
- **Power** is mostly positive, but contains **negative values** which are physically impossible during a charging session
- **Temperature** has a wide range, with rare spikes above 80°C suggesting thermal excursion events
- **Power discrepancy** (|reported power − V×I/1000|) shows a heavy right tail, indicating frequent small measurement noise and occasional large metering inconsistencies

These patterns confirm the presence of injected faults and suggest that derived features will be more diagnostic than raw signals alone.

### 2.2 Correlation Analysis

Strong correlations were observed between electrically related features:

- Voltage × Current ≈ Power (as expected by physics)
- Calculated power is highly correlated with reported power
- Energy rate tracks closely with power output

Notably, `error_code` shows **low correlation with raw electrical signals**, suggesting that firmware errors do not always correspond to obvious signal spikes. This means a purely threshold-based approach on raw signals would miss a significant portion of real faults — multivariate modeling is necessary.

### 2.3 Temporal Patterns

- Power usage varies slightly by hour and day of week, consistent with usage-driven demand patterns
- Error rates show mild temporal variation — not strongly time-locked, which implies faults are predominantly equipment-driven rather than load-driven
- Some stations exhibit consistently higher fault rates regardless of time, pointing to hardware-level issues

This indicates that station-level normalization is essential; global thresholds applied without context would produce excessive false positives on high-utilization stations and miss anomalies on low-utilization ones.

### 2.4 Error Code Analysis

- Approximately **0.56% of events** carry a non-zero firmware error code
- Only **four unique error codes** appear in the dataset
- Distribution across stations is uneven, with some stations accounting for a disproportionate share

Since error codes represent explicitly known fault signatures, they warrant zero-tolerance detection — any non-zero code is anomalous by definition. Their rarity confirms that they should serve as a hard rule layer rather than a model feature.

### 2.5 Power vs Temperature

Scatter analysis revealed a generally stable power-temperature relationship across the dataset, with identifiable outlier clusters. Events with power discrepancy greater than 2 kW represent roughly **2.24% of all events** — a meaningful anomaly signal that directly motivated the inclusion of `power_discrepancy` as both a feature and a hard rule trigger.

---

## 3. Feature Engineering

Feature engineering was designed to capture anomalies at multiple levels of abstraction, moving from raw signals to contextual and relational metrics.

### 3.1 Basic Derived Features

| Feature | Description |
|---|---|
| `power_calc_kw` | V × I / 1000 - the physically expected power |
| `power_discrepancy` | \|reported - calculated power\| - catches metering faults |
| `has_error` | Binary flag for non-zero error codes |
| `energy_rate` | kWh / duration_sec - detects meter or sensor faults |
| `hour`, `dayofweek`, `is_weekend` | Temporal context features |

These features capture measurement inconsistencies, known fault signatures, and time-of-day context that raw signals do not carry directly.

### 3.2 Rolling Features (Per Station, Window = 5 Events)

For `voltage`, `current`, `power_kw`, and `temperature_c`, the following were computed per-station:

- Rolling mean and rolling standard deviation
- Delta from rolling mean (deviation from recent local behavior)

These capture **sudden local deviations** that global statistics would miss. For example, a reading of 250 V may be globally within range but anomalous for a station normally operating at 220 V. Rolling features provide the short-term context needed to identify such cases.

### 3.3 Session-Level Aggregates

Aggregates computed per `session_id` and joined back to each event:

- Session error rate, event count, total energy delivered
- Average power and maximum temperature per session

This enables detection of **degraded sessions** - cases where no single event is extreme, but the session as a whole exhibits persistent instability or unusually high fault density.

### 3.4 Station Baseline Z-Scores

Per-station mean and standard deviation were computed across the full dataset. Z-scores were derived as:

```
z = (value - station_mean) / station_std
```

This normalizes for inter-station variability caused by hardware differences, installation conditions, and environmental factors. A z-score threshold of ±3.5 was used as a hard rule trigger, and z-scores for all four key signals were also included as model features.

---

## 4. Modeling Approach

A **hybrid anomaly detection system** was implemented, combining a statistical model with a deterministic rule layer.

### 4.1 Isolation Forest

Isolation Forest was selected as the primary model for several reasons:

- The dataset is unlabeled — supervised methods are not applicable
- Anomalies are expected to be rare and structurally different from normal data
- The feature space (~30 features) is moderately high-dimensional, and Isolation Forest handles this efficiently via random subspace splitting
- It produces a **continuous anomaly score**, enabling threshold tuning and priority ranking without binary commitment
- Training and inference are fast, making it viable for production-scale deployment

Contamination was set to **3%**, derived from observed fault prevalence:

- Power discrepancy events > 2 kW: ~2.24%
- Firmware error events: ~0.56%
- Combined expected anomaly rate: ~3%

This value balances detection sensitivity against false positive volume.

### 4.2 Domain-Informed Rule Layer

To guarantee detection of all known fault patterns regardless of model confidence, a deterministic rule layer was added. An event is flagged as anomalous if **any** of the following conditions hold:

1. Non-zero firmware error code
2. Negative power value (physically impossible)
3. Station z-score > ±3.5 for voltage, current, or temperature
4. Power discrepancy > 2 kW

**Final label: `Anomaly = Model Prediction OR Rule Trigger`**

This design ensures zero false negatives for explicitly known faults while retaining the model's ability to surface unknown or subtle anomalies.

### 4.3 Tradeoffs

**Strengths:**
- Captures subtle multivariate anomalies that rules alone would miss
- Guarantees detection of all known fault signatures
- Interpretable layered design, easy to explain to NOC engineers
- Scalable: fits and scores 200k+ events in seconds

**Limitations:**
- Static contamination threshold - does not adapt to drift over time
- Station baselines are computed on the full dataset; in production, these should be computed on a historical training window only
- No per-station adaptive contamination - high-fault stations may see elevated false positives

---

## 5. Evaluation Methodology

In the absence of complete ground-truth labels, evaluation relied on multiple complementary methods.

### 5.1 Score Distribution Analysis

Anomaly score distributions were compared between flagged and non-flagged events, confirming meaningful separation - flagged events cluster at lower (more anomalous) scores while normal events concentrate at higher scores.

### 5.2 Station-Level Anomaly Rates

Anomaly rates were aggregated per station. Stations with elevated rates correspond predictably to those with high error code frequency - a strong sanity check that the model is capturing real patterns, not noise.

### 5.3 Temporal Anomaly Trends

Anomaly rates were examined by hour of day. The absence of implausible temporal spikes confirms no modeling artifact was introduced from time-of-day features.

### 5.4 Contamination Sensitivity Analysis

The contamination parameter was swept from 1% to 10%. Flagged event counts increased gradually and consistently - no cliff edges or instabilities - validating that the chosen 3% value sits in a stable, well-behaved regime.

### 5.5 Feature Importance (Correlation with Anomaly Score)

Pearson correlation was computed between each feature and the raw anomaly score (lower = more anomalous). The strongest drivers were `power_discrepancy`, rolling standard deviations, and station z-scores - confirming that the engineered features are the primary signal sources and that the model is reacting to instability and inconsistency as intended.

### 5.6 Model vs Rule Breakdown

Anomalies were categorized into three groups: model-only, rule-only, and both. Roughly **65% of flagged events matched known fault patterns** via the rule layer; the remaining ~35% represent subtle statistical deviations detected solely by the Isolation Forest. This demonstrates genuine complementarity between the two layers rather than redundancy.

### 5.7 Detection Rate on Known Fault Patterns

The Isolation Forest alone captured approximately **95% of power discrepancy events** and only **~15% of firmware error events** - the latter being rare and not obviously reflected in continuous signal space. The rule layer closes this gap, achieving **100% recall** on all known fault types.

---

## 6. Results and Interpretation

Approximately **4.29% of events** were flagged as anomalous across the dataset. The majority were explained by known fault patterns; the remainder represent subtle statistical deviations.

**Examples of flagged events:**
- Zero current with non-zero reported power
- Voltage readings deviating more than 3.5 standard deviations from the station's baseline
- Negative power values
- Sessions with elevated rolling volatility across multiple signals simultaneously

**False positives** most likely arise from sudden but operationally valid spikes - e.g., a charger restarting or a brief overcurrent that self-corrected. These could be reduced by incorporating a session-phase feature to distinguish startup transients from sustained anomalies.

**False negatives** most likely occur for slow drift - gradual degradation that stays within rolling and z-score thresholds but represents a real decline in charger health. Addressing this would require longer-horizon time series features or a dedicated drift detection layer.

**In production, the system would:**
- Prioritize stations with persistent elevated anomaly rates for proactive maintenance
- Use anomaly scores to rank events for NOC triage rather than treating all flags equally
- Batch alerts by session or station to reduce alert fatigue

---

## 7. What I Would Improve With More Time

- **Drift detection:** Implement statistical drift monitoring across days using longer rolling windows, to catch gradual degradation not visible in event-level features
- **Adaptive thresholding:** Replace the fixed contamination parameter with a validation-set calibrated threshold, or per-station adaptive contamination based on historical fault rates
- **Semi-supervised refinement:** Even 200–300 labeled examples from NOC ticket history would enable PU Learning or label propagation to significantly improve precision
- **Explainability layer:** Add SHAP values per prediction so NOC engineers understand which features drove each flag, improving operator trust and feedback quality
- **Streaming pipeline:** Convert batch preprocessing to an incremental, streaming-ready design for real-time NOC alerting
- **Alert prioritization scoring:** Combine anomaly score, station fault history, and session-level context into a composite priority score for smarter NOC queue management
