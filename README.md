#  Telemetry Anomaly Detection Framework — v3.0

A production-grade, two-phase anomaly detection pipeline for multi-device telemetry.  
Detects latency spikes, CPU anomalies, thermal events, battery drain, network issues, and memory pressure across heterogeneous device fleets.

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║               TELEMETRY ANOMALY DETECTION FRAMEWORK  v3.0                       ║
╠═══════════════════════════════╦════════════════════════════════════════════════╣
║   PHASE 1 · MODELS OF NORMAL  ║   PHASE 2 · ANOMALY DETECTION                 ║
╠═══════════════════════════════╬════════════════════════════════════════════════╣
║                               ║                                                ║
║  Datasets                     ║  Datasets + Algorithm Fits                     ║
║     │                         ║  + Compressed Feature Vectors                  ║
║     ▼                         ║            │                                   ║
║  ┌──────────────────────┐     ║            ▼                                   ║
║  │     Algorithms       │     ║  ┌──────────────────────────────────────────┐ ║
║  │  ┌──────────────┐    │     ║  │         Anomaly Definitions              │ ║
║  │  │ Rolling Mean │    │─────╫─►│  σ's From Mean of Data                  │ ║
║  │  ├──────────────┤    │     ║  │  σ's From Mean of Errors                │ ║
║  │  │    ARIMA     │    │     ║  │  Nonparametric Dynamic Thresholding      │ ║
║  │  ├──────────────┤    │     ║  │  RRCF Anomaly Scores ◄──────────────────┤ ║
║  │  │ Autoencoder  │    │     ║  └──────────────────────┬───────────────────┘ ║
║  │  └──────────────┘    │     ║                          │                     ║
║  └──────────────────────┘     ║                          ▼                     ║
║     │                         ║            More Data Files + Plots             ║
║     ▼                         ║                          │                     ║
║  Data Files + Plots           ║                          ▼                     ║
║     │                         ║            Counts of Anomalous Points          ║
║     ▼                         ║                                                ║
║  Correlation Coefficients     ║                                                ║
╚═══════════════════════════════╩════════════════════════════════════════════════╝
```

---

## Table of Contents

- [Architecture](#architecture)
- [What's New in v3](#whats-new-in-v3)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Phase 1 — Models of Normal](#phase-1--models-of-normal)
  - [Phase 2 — Anomaly Definitions](#phase-2--anomaly-definitions)
  - [Fusion & Ensemble](#fusion--ensemble)
  - [Outputs](#outputs)
- [Configuration Reference](#configuration-reference)
- [Output Files](#output-files)
- [Detectors Reference](#detectors-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualizations](#visualizations)
- [Extending the Framework](#extending-the-framework)

---

## What's New in v3

| Area | v2 | v3 |
|---|---|---|
| Phase 1 models | Rolling Z only | Rolling Mean + ARIMA + Autoencoder |
| Phase 2 detectors | IsoForest, LCAI, Cohort | σ/Data, σ/Errors, NDT, RRCF |
| Compressed vectors | None | Autoencoder 8-D latent space |
| Algorithm fits saved | No | Yes — CSV + pickle |
| Correlation matrix | None | Full model × signal Pearson r table |
| Anomaly counts table | Basic | Per-detector × per-device × per-type |
| RRCF | No | Yes — shingled online forest |
| Dynamic thresholding | No | Yes — NASA NDT algorithm |
| Output directory | None | `telemetry_v3_outputs/` with 5 sub-folders |

---

## Requirements

- Python **3.9+**
- Jupyter Notebook or JupyterLab

### Python packages

```
numpy
pandas
scipy
scikit-learn
statsmodels
plotly
rrcf
```

---

## Installation

```bash
# Clone or download the notebook
git clone <your-repo-url>
cd telemetry-anomaly-v3

# Install dependencies
pip install numpy pandas scipy scikit-learn statsmodels plotly rrcf

# Launch notebook
jupyter notebook anamoly_v3.ipynb
```

> **Note:** The notebook automatically attempts `pip install rrcf` if the package is not found.  
> A pure-Python fallback (IsolationForest-based approximation) is used if installation fails.

---

## Project Structure

```
anamoly_v3.ipynb                    ← Main pipeline notebook
telemetry_v3_outputs/               ← All outputs (auto-created on first run)
├── models/
│   ├── rolling_mean_stats.csv      ← Per-device per-signal baseline stats
│   ├── arima_fits.csv              ← ARIMA predicted values & residuals
│   ├── arima_models.pkl            ← Serialised ARIMAResults objects
│   └── autoencoder_models.pkl      ← Serialised encoder / decoder / scalers
├── features/
│   └── compressed_feature_vectors.csv  ← 8-D latent space per data point
├── reports/
│   ├── correlation_coefficients.csv    ← Phase 1 model correlation table
│   ├── anomaly_counts.csv              ← Phase 2 anomaly counts by detector
│   └── pipeline_report.txt             ← Full plain-text summary report
└── anomalies/
    └── alerts.csv                      ← All alerts with tier, severity, metadata
```

---

## Pipeline Walkthrough

### Phase 1 — Models of Normal

The left side of the architecture. Three complementary algorithms are fitted on device telemetry to model what **normal behaviour looks like**.

#### Rolling Mean Model

Fits a sliding-window mean and standard deviation over each signal per device.

- **Window size:** configurable via `rolling_window` (default: 50 samples)
- **Signals covered:** `latency`, `cpu_usage`, `temperature`, `packet_loss`
- **Outputs per signal:** `rm_pred_{signal}`, `rm_resid_{signal}`, `rm_z_{signal}`
- **Saved to:** `models/rolling_mean_stats.csv`

#### ARIMA Model

Fits an ARIMA(p,d,q) model on each device–signal combination. Trained on the first `arima_train_frac` of the time series; residuals are computed for the full series.

- **Default order:** ARIMA(2, 1, 2)
- **Signals covered:** `latency`, `cpu_usage` (configurable via `arima_signals`)
- **Outputs per signal:** `arima_pred_{signal}`, `arima_resid_{signal}`
- **Saved to:** `models/arima_fits.csv`, `models/arima_models.pkl`

> ARIMA is particularly effective at capturing auto-regressive patterns and seasonal drift in latency and CPU load time series.

#### Autoencoder Model — Compressed Feature Vectors

An MLP encoder-decoder trained to reconstruct multivariate telemetry through a compressed 8-dimensional bottleneck. High reconstruction error signals the model has encountered something outside the normal manifold.

- **Architecture:** `input (10) → 32 → 8 (latent) → 32 → input (10)`
- **Latent dimensions:** 8 (configurable via `ae_latent_dim`)
- **Input features:** latency, cpu_usage, memory_usage, temperature, packet_loss, jitter, disk_io, gpu_usage, drain_rate, app_load_time
- **Outputs:** `ae_latent_0…7`, `ae_recon_error`, `ae_recon_error_z`
- **Saved to:** `features/compressed_feature_vectors.csv`, `models/autoencoder_models.pkl`

#### Correlation Coefficients

After all Phase 1 models are fitted, a Pearson correlation table is computed between each model's predictions and the raw signal values. This is a key Phase 1 diagnostic output.

- **Saved to:** `reports/correlation_coefficients.csv`

---

### Phase 2 — Anomaly Definitions

The right side of the architecture. Four definitions are applied to the model fits and compressed feature vectors from Phase 1.

#### σ's From Mean of Data

Flags raw data points that lie more than **k standard deviations** from the per-device rolling mean. Also flags extreme autoencoder reconstruction error.

- **Threshold:** `sigma_data_k` (default: 3.0)
- **Column:** `det_sigma_data`
- **Best for:** point anomalies with sharp magnitude changes (e.g. latency spikes, CPU spikes)

#### σ's From Mean of Errors

Flags data points whose **model residuals** (actual − predicted) deviate more than k standard deviations from the mean residual. Operates on both Rolling Mean and ARIMA residuals.

- **Threshold:** `sigma_error_k` (default: 3.0)
- **Column:** `det_sigma_errors`
- **Best for:** anomalies that are hard to detect in raw data but create large prediction errors (e.g. gradual thermal drift)

#### Nonparametric Dynamic Thresholding (NASA NDT)

Implementation of the NASA Telemetry Anomaly Detection algorithm (Hundman et al., ICML 2018). The threshold adapts to the local error distribution at each timestep using a trailing window — no assumption of normality required.

**Algorithm:**
1. Aggregate residuals from all Phase 1 models
2. Apply Savitzky-Golay smoothing to the error signal
3. For each point, compute `threshold = μ_window + k × σ_window`
4. Flag the point if its error exceeds the adaptive threshold

- **Window:** `ndt_window` (default: 50)
- **z-score:** `ndt_z_score` (default: 2.5)
- **Column:** `det_ndt`
- **Best for:** contextual anomalies and gradual degradation patterns

#### RRCF Anomaly Scores — Robust Random Cut Forest

Online streaming anomaly detector based on Guha et al. (ICML 2016). Maintains a forest of Robust Random Cut Trees over a sliding shingle (window) of recent points. The **CoDisp score** (Collusive Displacement) measures how much a point disrupts the tree structure — high CoDisp means the point is anomalous.

**Algorithm:**
1. Shingle the time series into overlapping windows of size `rrcf_shingle_size`
2. Maintain `rrcf_num_trees` RCTrees, each capped at `rrcf_tree_size` points
3. For each new point: insert → compute CoDisp → remove oldest if buffer full
4. Flag points above the `rrcf_threshold_pct` percentile of CoDisp scores

- **Trees:** `rrcf_num_trees` (default: 40)
- **Tree size:** `rrcf_tree_size` (default: 256)
- **Shingle size:** `rrcf_shingle_size` (default: 8)
- **Column:** `det_rrcf`
- **Score column:** `rrcf_codisp`
- **Best for:** streaming detection of novel anomaly patterns without retraining

---

### Fusion & Ensemble

After all four anomaly definitions fire independently, their votes are combined with two additional signals:

| Vote source | Description |
|---|---|
| `det_sigma_data` | σ from mean of data |
| `det_sigma_errors` | σ from mean of errors |
| `det_ndt` | Nonparametric dynamic threshold |
| `det_rrcf` | RRCF CoDisp score |
| `vote_risk` | Multi-signal risk score > threshold |
| `vote_csaf` | Cross-Modal Semantic Anomaly Fusion score |

A row is marked **`final_anomaly = 1`** when `vote_total ≥ ensemble_min_votes` (default: 2).

**Severity score:**
```
severity = 0.50 × risk_score + 0.30 × vote_confidence + 0.20 × csaf_score
```

**Severity tiers** (assigned via DBSCAN incident clustering):

| Tier | Severity threshold |
|---|---|
| CRITICAL | ≥ 0.80 |
| HIGH | ≥ 0.60 |
| MEDIUM | ≥ 0.40 |
| LOW | < 0.40 |

---

### Outputs

**Phase 1 — Data Files & Plots**
- Model fit CSVs + pickled model objects
- 9 interactive Plotly visualizations
- Correlation coefficients table

**Phase 2 — More Data Files & Plots**
- Anomaly counts table (per-detector × per-device)
- RRCF CoDisp score timeline
- Anomaly definition agreement heatmap
- Severity tier distribution
- Priority-ranked `alerts.csv`
- Plain-text `pipeline_report.txt`

---

## Configuration Reference

All parameters live in `TelemetryConfig` (§ 1). Edit once; the entire pipeline picks up the change.

```python
@dataclass
class TelemetryConfig:
    # Data
    n_points_per_device : int   = 3_000
    anomaly_rate        : float = 0.05
    devices             : List[str] = ["iPhone", "MacBook", "Apple Watch", "iPad"]

    # Phase 1 — Models of Normal
    rolling_window      : int   = 50
    arima_order         : tuple = (2, 1, 2)
    arima_train_frac    : float = 0.80
    arima_signals       : List[str] = ["latency", "cpu_usage"]
    ae_latent_dim       : int   = 8
    ae_hidden_dim       : int   = 32
    ae_max_iter         : int   = 80

    # Phase 2 — Anomaly Definitions
    sigma_data_k        : float = 3.0
    sigma_error_k       : float = 3.0
    ndt_window          : int   = 50
    ndt_z_score         : float = 2.5
    rrcf_num_trees      : int   = 40
    rrcf_tree_size      : int   = 256
    rrcf_shingle_size   : int   = 8
    rrcf_threshold_pct  : float = 95.0

    # Ensemble
    ensemble_min_votes  : int   = 2
    risk_threshold      : float = 0.70
    csaf_threshold      : float = 0.65
```

---

## Output Files

| File | Description |
|---|---|
| `models/rolling_mean_stats.csv` | Per-device per-signal mean, std, p95 baselines |
| `models/arima_fits.csv` | Full time-indexed ARIMA predictions and residuals |
| `models/arima_models.pkl` | Fitted `ARIMAResults` objects keyed by (device, signal) |
| `models/autoencoder_models.pkl` | Encoder, decoder, and scaler objects per device |
| `features/compressed_feature_vectors.csv` | 8-D latent vectors + reconstruction error per row |
| `reports/correlation_coefficients.csv` | Pearson r: model predictions vs raw signals |
| `reports/anomaly_counts.csv` | Counts of flagged rows per detector per device |
| `reports/pipeline_report.txt` | Human-readable end-to-end summary |
| `anomalies/alerts.csv` | Ranked alerts with tier, severity, detectors fired, CSAF status |

---

## Detectors Reference

| Detector | Class | Column | Approach |
|---|---|---|---|
| σ from Mean of Data | `SigmaFromMeanOfData` | `det_sigma_data` | Rolling z-score on raw signal |
| σ from Mean of Errors | `SigmaFromMeanOfErrors` | `det_sigma_errors` | z-score on model residuals |
| Nonparametric Dynamic Threshold | `NonparametricDynamicThreshold` | `det_ndt` | Adaptive window threshold on smoothed errors |
| RRCF | `RRCFDetector` | `det_rrcf` | CoDisp score from shingled random cut forest |
| Risk Vote | `RiskScorer` | `vote_risk` | Weighted multi-signal percentile score |
| CSAF | `CSAFFusion` | `vote_csaf` | Telemetry + UX signal fusion |

---

## Evaluation Metrics

The `PipelineEvaluator` (§ 9) produces the following against injected ground-truth labels:

| Metric | Description |
|---|---|
| ROC-AUC | Area under the receiver operating characteristic curve |
| Average Precision | Area under precision-recall curve (better for imbalanced data) |
| Precision | TP / (TP + FP) — how many flagged anomalies are real |
| Recall | TP / (TP + FN) — how many real anomalies were caught |
| F1-Score | Harmonic mean of precision and recall |
| False Positive Rate | FP / (FP + TN) — noise in the alert stream |
| False Negative Rate | FN / (FN + TP) — missed anomalies |

Per-detector metrics are printed for all six vote columns, allowing comparison of each anomaly definition in isolation.

---

## Visualizations

9 interactive Plotly charts are rendered inline in the notebook:

| Plot | What it shows |
|---|---|
| 1 · Multi-Model Fit | Raw latency vs Rolling Mean vs ARIMA predictions + anomaly overlay |
| 2 · Residual Comparison | Rolling Mean residuals vs ARIMA residuals side-by-side |
| 3 · Autoencoder | Reconstruction error timeline + latent space scatter (dim-0 vs dim-1) |
| 4 · Definition Agreement | Pearson r heatmap between all six detector vote columns |
| 5 · Anomaly Counts | Grouped bar: counts per detector per device (Phase 2 output) |
| 6 · ROC + PR Curves | Ensemble evaluation curves with AUC/AP annotations |
| 7 · Severity Tiers | Bar chart of CRITICAL / HIGH / MEDIUM / LOW / NORMAL distribution |
| 8 · Correlation Coefficients | Heatmap of Phase 1 model r values (Phase 1 output) |
| 9 · RRCF CoDisp Timeline | CoDisp score over time with adaptive threshold band |

---

## Extending the Framework

**Adding a new detector** — implement the `AnomalyDetector` ABC and add your column to `EnsembleDetector.DETECTOR_COLS`:

```python
class MyDetector:
    def detect(self, df: pd.DataFrame) -> pd.Series:
        # return int Series (1 = anomaly)
        ...

df["det_my"] = MyDetector(CFG).detect(df)
EnsembleDetector.DETECTOR_COLS.append("det_my")
```

**Adding a new Phase 1 model** — fit your model, write predictions/residuals into the DataFrame, then add the residual column name to the error aggregation in `NonparametricDynamicThreshold.detect()` (it auto-collects any column matching `*resid*`).

**Swapping in real data** — replace the `generate_telemetry()` call in § 2 with your own DataFrame loader. Required columns: `device`, `timestamp`, plus any signals listed in `ae_features` and `arima_signals`. `anomaly_label` is optional (evaluation will be skipped if absent).

---

## References

- Hundman, K. et al. (2018). [Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://arxiv.org/abs/1802.04431). *KDD 2018.*
- Guha, S. et al. (2016). [Robust Random Cut Forest Based Anomaly Detection on Streams](https://proceedings.mlr.press/v48/guha16.html). *ICML 2016.*
- Breunig, M. et al. (2000). LOF: Identifying Density-Based Local Outliers. *ACM SIGMOD 2000.*

---

## License

MIT — free to use, modify, and distribute.


## AUTHOR   

DATTU NAIK MUDAVATH, SENIOR DATA SCIENTIST
