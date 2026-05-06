# Apple Watch Telemetry Anomaly Detection & RCA Pipeline

## Overview

This project is a **production-style AI anomaly detection and root-cause analysis pipeline** for wearable telemetry data such as:

* ECG
* Heart Rate
* Signal Quality
* RR Interval
* Motion Noise
* Device Reliability Signals

The system simulates how an intelligent healthcare monitoring platform can:

* detect anomalies in physiological signals
* identify AFib-related failures
* evaluate signal degradation
* perform ensemble anomaly detection
* generate AI-assisted root cause analysis (RCA)

The notebook demonstrates an enterprise-style architecture using:

* Statistical anomaly detection
* Machine learning
* Structural anomaly detection
* Ensemble intelligence
* LLM-powered RCA (Groq / Ollama / TinyLlama)

---

# Key Features

| Capability                                 | Description                             |
| ------------------------------------------ | --------------------------------------- |
| Synthetic Apple Watch telemetry generation | Simulates wearable sensor data          |
| ECG anomaly simulation                     | Injects AFib/noisy conditions           |
| Sigma threshold detection                  | Statistical anomaly detection           |
| RRCF detection                             | Structural anomaly detection            |
| Autoencoder detection                      | Reconstruction-based anomaly detection  |
| Ensemble anomaly scoring                   | Combines multiple detectors             |
| Signal quality analytics                   | Detects degraded wearable signals       |
| LLM Root Cause Analysis                    | AI-generated incident explanation       |
| KPI evaluation                             | Precision / Recall / F1 analysis        |
| Visualization                              | Multi-stage telemetry plots             |
| Production logging                         | Structured runtime logs                 |
| Retry handling                             | Resilient LLM inference                 |
| Enterprise architecture                    | Demonstrates scalable monitoring design |

---

# Architecture

```text
Apple Watch Telemetry
        ↓
Signal Processing
        ↓
Feature Engineering
        ↓
Anomaly Detection Layer
   ├── Sigma Threshold
   ├── RRCF
   └── Autoencoder
        ↓
Ensemble Detection
        ↓
Root Cause Analysis Engine
   ├── Groq LLM
   ├── Ollama Local LLM
   └── Rule-based RCA
        ↓
Human-readable Incident Report
```

---

# Project Structure

| Section | Description                    |
| ------- | ------------------------------ |
| S1.1    | Synthetic telemetry generation |
| S1.2    | Signal preprocessing           |
| S1.3    | Multi-model anomaly detection  |
| S1.4    | Ensemble intelligence          |
| S1.5    | LLM root cause engine          |
| S1.6    | KPI evaluation                 |
| S1.7    | Visualization & reporting      |

---

# Technologies Used

| Technology         | Purpose                  |
| ------------------ | ------------------------ |
| Python             | Core development         |
| NumPy              | Numerical computation    |
| Pandas             | Telemetry analytics      |
| Matplotlib         | Visualization            |
| Scikit-learn       | Metrics & ML utilities   |
| TensorFlow / Keras | Autoencoder              |
| RRCF               | Robust Random Cut Forest |
| Groq API           | Cloud LLM inference      |
| Ollama             | Local LLM runtime        |
| TinyLlama          | Lightweight local LLM    |

---

# Detection Models

## 1. Sigma Threshold Detection

Statistical anomaly detection using rolling mean and standard deviation.

### Purpose

Detect:

* sudden ECG spikes
* abnormal signal deviations
* noise bursts

### Formula

x > \mu + k\sigma

---

## 2. RRCF (Robust Random Cut Forest)

Structural anomaly detection using random partitioning trees.

### Purpose

Detect:

* unusual telemetry structure
* abnormal waveform behavior
* sequence irregularities

### Strength

Excellent for:

* time-series anomalies
* streaming telemetry
* irregular wearable signals

---

## 3. Autoencoder Detection

Neural-network reconstruction anomaly detection.

### Purpose

Detect:

* unseen ECG patterns
* latent-space abnormalities
* reconstruction failures

### Flow

```text
Input Signal
      ↓
Encoder
      ↓
Latent Space
      ↓
Decoder
      ↓
Reconstruction Error
```

---

# Ensemble Detection

The system combines all detectors:

```python
ensemble_flags = (
    sigma_flags +
    detected_rrcf +
    detected_ae
) >= 2
```

This uses majority voting to improve reliability.

---

# Root Cause Analysis (RCA)

The pipeline supports:

| RCA Mode       | Description                    |
| -------------- | ------------------------------ |
| Rule-based RCA | Offline deterministic analysis |
| Groq LLM RCA   | Cloud-hosted LLM reasoning     |
| Ollama RCA     | Fully local LLM inference      |

---

# Example RCA Output

```text
ROOT CAUSE:
High ECG noise and degraded signal quality caused unstable AFib feature extraction.

FAILURE MODE:
RR interval irregularity was masked by telemetry corruption.

SEVERITY:
HIGH

RECOMMENDATION:
Apply adaptive denoising and confidence-aware anomaly gating.
```

---

# Installation

## 1. Create Virtual Environment

```bash
python -m venv venv
```

---

## 2. Activate Environment

### Windows

```bash
.\venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Recommended requirements.txt

```text
numpy
pandas
matplotlib
scikit-learn
tensorflow
rrcf
requests
groq
jupyter
notebook
```

---

# Optional Local LLM Setup (Recommended)

Install Ollama

Download:

```text
https://ollama.com/download
```

Pull TinyLlama:

```bash
ollama run tinyllama
```

---

# Optional Groq Setup

Get free API key from:

Groq

```text
https://console.groq.com
```

Set API key:

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="gsk_xxxxxxxxx"
```

---

# SSL Certificate Issues (Corporate Networks)

If running behind:

* VPN
* office proxy
* enterprise firewall

you may encounter:

```text
SSLCertVerificationError
```

Temporary PoC workaround:

```python
import os
import urllib3

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)
```

---

# Running the Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
AppleWatch_Anomaly_Production.ipynb
```

Run cells sequentially.

---

# Example Outputs

## Detection Metrics

| Detector        | Precision | Recall | F1   |
| --------------- | --------- | ------ | ---- |
| Sigma Threshold | 0.81      | 0.72   | 0.76 |
| RRCF            | 0.88      | 0.79   | 0.83 |
| Autoencoder     | 0.85      | 0.82   | 0.83 |
| Ensemble        | 0.91      | 0.87   | 0.89 |

---

# Example Use Cases

| Industry                  | Application              |
| ------------------------- | ------------------------ |
| Healthcare AI             | AFib monitoring          |
| Wearables                 | Signal reliability       |
| Telemetry AI              | Sensor anomaly detection |
| Remote patient monitoring | Incident detection       |
| AI observability          | Root cause analytics     |
| Medical IoT               | Signal intelligence      |

---

# Enterprise Extensions

Potential future enhancements:

* Multi-patient telemetry scaling
* Real-time streaming analytics
* Kafka-based ingestion
* Edge-device anomaly detection
* Federated wearable learning
* Drift monitoring
* Multimodal sensor fusion
* Clinical event prediction
 
---

# Author
DATTU NAIK MUDAVATH - SENIOR DATA SCIENTIST
Apple Watch AI Anomaly Detection PoC
Production-style Healthcare Telemetry Intelligence System
