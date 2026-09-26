# NEXORA — Working Software Prototype Documentation
**Project:** NEXORA — Low-Latency Edge Voice Activator  
**Challenge / Problem ID:** SIH26172  
**Target Wake Word:** `"NEXORA"`  
**Implementation:** Standalone Software Prototype (Laptop Microphone & CPU)  
**Hardware Compatibility:** Pre-architected for ESP32-S3 + I2S MEMS Microphone + TinyML INT8 deployment.

---

## 1. System Architecture & Dataflow

```
Laptop Microphone (Realtek Audio Array)
   ↓ 16 kHz Mono PCM (16-bit)
Sliding Window Framing (1.0s Window / 16000 samples, 200ms Step / 3200 samples)
   ↓ Pre-emphasis (0.97) + Hann Windowing (400 samples / 25ms, 160 hop / 10ms)
512-point FFT & 40-Mel Filterbank Energy
   ↓ DCT-II & Cepstral Mean Subtraction (CMS)
MFCC Feature Extraction (98 frames × 13 coefficients = 1274 scalar dimensions)
   ↓ StandardScaler Feature Normalization
Small Custom Neural Classifier (MLP: 1274 → 64 [ReLU] → 32 [ReLU] → 3 [Softmax])
   ↓ Posterior Probabilities: [SILENCE, UNKNOWN, NEXORA]
Temporal Verification (Sliding window + 2 consecutive positive frames + cooldown)
   ↓
>>> NEXORA CONFIRMED <<<
   ↓
Command Audio Capture (DORMANT before wake word; records 4.0s post-trigger)
   ↓
Local Offline ASR Engine (Vosk Open-Source Recognizer — 100% offline, zero cloud)
   ↓
Final Command Transcript
```

### Strict Constraint Compliance Checklist
- [x] **No Picovoice / Porcupine**
- [x] **No Alexa / Google wake-word models**
- [x] **No commercial wake-word SDKs or proprietary APIs**
- [x] **No pretrained global wake-word models**
- [x] **No cloud speech APIs**
- [x] **Custom wake word detector trained on NEXORA acoustic recordings**
- [x] **Zero command audio routed to ASR prior to NEXORA confirmation**
- [x] **100% Offline execution on host machine**

---

## 2. Repository File Structure

```
NEXORA-SOFTWARE-MVP/
├── software/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py             # Central system constants and parameters
│   │   ├── audio_capture.py      # Thread-safe 16 kHz microphone ring buffer
│   │   ├── features.py           # 16 kHz framing, FFT, Mel filterbank & MFCC
│   │   ├── kws.py                # Keyword spotting inference & latency timing
│   │   ├── temporal.py           # Multi-frame temporal verification & cooldown
│   │   ├── command_capture.py    # Post-wake audio recording to recordings/
│   │   ├── asr.py                # Local offline Vosk speech recognition
│   │   ├── intent.py             # Edge intent gatekeeper (offline rule classifier)
│   │   ├── actions.py            # Safe simulated action executor & response layer
│   │   └── main.py               # Real-time terminal demonstration UI
│   │
│   ├── dataset/
│   │   ├── nexora/               # Target wake-word recordings (16 kHz mono WAV)
│   │   ├── unknown/              # Confuser & command recordings (16 kHz mono WAV)
│   │   ├── silence/              # Ambient mic & background noise (16 kHz mono WAV)
│   │   ├── dataset_features.npz  # Preprocessed MFCC features & train/val/test splits
│   │   └── dataset_report.json   # Integrity, sample rate & duration report
│   │
│   ├── training/
│   │   ├── collect_samples.py    # Interactive mic sample collection tool
│   │   ├── bootstrap_samples.py  # Seed speech acoustic dataset generator
│   │   ├── prepare_dataset.py    # Validation, feature extraction & stratified split
│   │   ├── train.py              # Lightweight TinyML-compatible MLP training
│   │   ├── evaluate.py           # Evaluation report & confusion matrix generator
│   │   └── export.py             # C Header exporter for ESP32-S3 deployment
│   │
│   ├── benchmark.py              # Latency, memory, and model size profiler
│   ├── test_e2e.py               # Complete end-to-end pipeline verification test
│   └── requirements.txt          # Minimal Python dependencies
│
├── models/
│   ├── nexora_model.joblib       # Trained scaler + MLP model pipeline
│   ├── model_metadata.json       # Parameter count, layers, accuracy metadata
│   └── nexora_model_weights.h    # Exported C arrays for ESP32-S3 firmware
│
├── recordings/                   # Captured post-wake command WAV files
├── logs/
│   ├── dataset_report.json       # Dataset audit report
│   ├── evaluation_report.json    # Precision, recall, and false positive metrics
│   ├── confusion_matrix.png      # Plotted confusion matrix
│   └── benchmark_report.json     # Software performance benchmarks
│
├── docs/
│   └── software_prototype.md     # This comprehensive documentation
└── requirements.txt
```

---

## 3. Step-by-Step Runnable Commands

### Step 1: Install Dependencies
Open PowerShell in the project directory:
```powershell
# Create and activate virtual environment (if not already active)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r software/requirements.txt
```

### Step 2: Test Microphone Access
Verify that the laptop microphone is detected and responsive:
```powershell
python software/training/collect_samples.py --test-mic
```

### Step 3: Collect Custom Audio Samples (Interactive)
Record custom voice samples directly through the laptop microphone:
```powershell
# Record 10 repetitions of "NEXORA"
python software/training/collect_samples.py --class nexora --count 10

# Record 10 negative confuser words ("NEBULA", "NECTAR", "ALEXA", etc.)
python software/training/collect_samples.py --class unknown --count 10

# Record ambient room noise / fan / keyboard typing
python software/training/collect_samples.py --class silence --auto-silence --count 10
```

*(Note: To seed baseline speech acoustic variations automatically, run `python software/training/bootstrap_samples.py`)*

### Step 4: Prepare & Validate Dataset
Validates 16 kHz sample rate, mono channels, audio integrity, extracts MFCC features, and splits into 70% Train, 15% Val, 15% Test:
```powershell
python software/training/prepare_dataset.py
```
*Outputs: `software/dataset/dataset_features.npz` and `software/dataset/dataset_report.json`*

### Step 5: Train Custom NEXORA Wake Word Model
Trains the custom lightweight MLP classifier (1274 → 64 → 32 → 3):
```powershell
python software/training/train.py
```
*Outputs: `models/nexora_model.joblib` and `models/model_metadata.json`*

### Step 6: Evaluate Model & Generate Metrics
Evaluates model on the held-out test split and generates the confusion matrix:
```powershell
python software/training/evaluate.py
```
*Outputs: `logs/evaluation_report.json` and `logs/confusion_matrix.png`*

### Step 7: Export Model for ESP32-S3 TinyML Deployment
Generates C header file with weights, biases, and normalization scaling for ESP-IDF / Arduino ESP32 firmware:
```powershell
python software/training/export.py
```
*Outputs: `models/nexora_model_weights.h`*

### Step 8: Run Software Performance Benchmark
Profiles feature extraction, model inference, temporal verification, and local ASR latencies:
```powershell
python software/benchmark.py
```
*Outputs: `logs/benchmark_report.json`*

### Step 9: Run Automated End-to-End Pipeline Verification Test
Executes a comprehensive non-interactive test verifying silence rejection, confuser rejection, NEXORA confirmation, and Vosk transcription:
```powershell
python software/test_e2e.py
```

### Step 10: Run Real-Time NEXORA Terminal Demo
Launches the continuous real-time listener on the laptop microphone:
```powershell
python software/app/main.py
```
*Optional flags:*
- `--threshold 0.75` (Confidence threshold, default: 0.75)
- `--consecutive 2` (Consecutive positive frames required, default: 2)
- `--command-duration 4.0` (Command recording duration in seconds, default: 4.0)

---

## 4. Software Performance Measurements

> [!NOTE]
> **SOFTWARE MEASUREMENT DISCLAIMER:**  
> The metrics below were measured on the host Windows laptop CPU (Intel/AMD x86_64, 16 kHz audio stream).  
> These are **SOFTWARE MEASUREMENTS** and are **NOT** claimed to be ESP32-S3 hardware measurements.  
> On-target ESP32-S3 measurements will be determined once deployed to physical hardware.

| Metric | Measured Value (Host PC) | Measurement Unit |
| :--- | :--- | :--- |
| **Acoustic Sampling Rate** | 16,000 | Hz (Mono PCM) |
| **Audio Frame Length** | 25.0 (400 samples) | ms |
| **Audio Frame Step** | 10.0 (160 samples) | ms |
| **Mel Filterbanks** | 40 | filters (0 - 8000 Hz) |
| **MFCC Feature Dimensions** | 13 × 98 frames = 1274 | scalars |
| **Feature Extraction Latency (Mean)** | **0.74** | ms |
| **Model Inference Latency (Mean)** | **1.16** | ms |
| **Temporal Verification Overhead** | **0.0050** | ms |
| **Total Local KWS Detection Latency** | **2.07** | ms |
| **Total Local KWS Latency (P95)** | **2.34** | ms |
| **Local ASR Latency (2.0s audio)** | **242.65** | ms |
| **KWS Model Parameter Count** | 83,779 | parameters |
| **Estimated INT8 Weight Size** | **81.82** | KB (fits in ESP32 512KB SRAM) |
| **Process RAM (Resident Set Size)** | **249.29** | MB (including Vosk model) |
| **Python Traced Heap Peak** | **43.85** | MB |
| **Process CPU Utilization (Benchmark)** | **107.2%** | % (multi-core burst during 100 iters) |
| **CPU Time per Audio Window** | **2.19** | ms CPU time |
| **Energy Proxy (Active Duty Cycle)** | **1.03%** | active duty cycle (98.97% sleep window) |

> [!NOTE]
> **Energy Proxy Clarification:**  
> Direct physical energy consumption (Joules/milliwatts) cannot be measured without hardware current measurement shunts. Active duty cycle ($T_{active} / T_{step} = 2.07\text{ ms} / 200\text{ ms} = 1.03\%$) serves as the standard embedded proxy for energy efficiency.


---

## 5. Model Evaluation Metrics

From `logs/evaluation_report.json` (evaluated on held-out test split):

- **Overall Test Accuracy:** `94.44%`
- **NEXORA Precision:** `85.71%`
- **NEXORA Recall (TPR):** `100.00%` (0 missed detections)
- **NEXORA F1-Score:** `0.9231`
- **False NEXORA Detections:** `1`
- **Missed NEXORA Detections:** `0`

### Confusion Matrix
```
               Predicted:  SILENCE   UNKNOWN   NEXORA
Actual: SILENCE               5         0         0
Actual: UNKNOWN               0         6         1
Actual: NEXORA                0         0         6
```

---

## 6. Edge Intent Gatekeeper & Action Layer

### Architecture & Pipeline
```
Vosk Offline ASR Transcript
         ↓
Intent Gatekeeper (software/app/intent.py)
- Input: Text transcript
- Normalization: Lowercase, punctuation removal, whitespace trimming
- Matcher: Compiled edge regex / keyword rule mapping
         ↓
Intent Classification (GREETING, LIGHT_ON, LIGHT_OFF, SYSTEM_STATUS, HELP, UNKNOWN_INTENT)
         ↓
Action Executor (software/app/actions.py)
- Whitelisted intent dispatch table
- Safe state mutation (Simulated Light State: ON/OFF)
- Response generator
         ↓
Local Text Response & Telemetry Output
         ↓
Return to LISTENING
```

### Supported Intents & Sample Commands
| Intent | Example Commands | Action Executed | Local Response |
| :--- | :--- | :--- | :--- |
| **`GREETING`** | *"hello"*, *"hi"*, *"how are you"*, *"good morning"* | `NONE` (Acknowledged) | *"I'm ready and listening."* |
| **`LIGHT_ON`** | *"turn on the lights"*, *"switch on the light"*, *"lights on"* | `LIGHT_ON` (Simulated Light = ON) | *"Light turned on."* |
| **`LIGHT_OFF`** | *"turn off the lights"*, *"switch off lights"*, *"lights off"* | `LIGHT_OFF` (Simulated Light = OFF) | *"Light turned off."* |
| **`SYSTEM_STATUS`** | *"what is your status"*, *"system status"*, *"are you working"* | `SYSTEM_STATUS` (Telemetry audit) | *"All systems operational. Microphone active, KWS ready, offline ASR online, simulated light is {ON/OFF}."* |
| **`HELP`** | *"help"*, *"what can you do"*, *"show available commands"* | `HELP` (Catalog listing) | *"Available commands: 'turn on the lights', 'turn off the lights', 'system status', 'how are you', 'help'."* |
| **`UNKNOWN_INTENT`** | Arbitrary or unrecognized phrases | `NONE` (Zero execution) | *"Command not recognized. Say 'help' for available commands."* |

### Action Safety Boundaries
- **Zero Arbitrary Execution:** Speech transcripts are NEVER passed to `eval()`, `exec()`, `os.system()`, or shell subprocesses.
- **Whitelist Dispatch Table:** Only explicitly enumerated intents can trigger state changes.
- **Safe Fallback:** All unrecognized inputs automatically route to `UNKNOWN_INTENT` with `action: NONE` and zero side effects.

### Test Results
- **Software-Only Intent Test (`python software/app/main.py --intent-test`):** 6/6 test phrases classified and executed with 100% accuracy.
- **End-to-End Suite (`python software/test_e2e.py`):** All 12 integration checks passed (Silence, Confuser, KWS, Temporal, ASR, 5 Intents, Unknown handling, and Action isolation).
- **Latency Impact:** Intent classification latency is **~0.016 ms** and action execution is **~0.011 ms** (total post-ASR latency: **0.027 ms**), introducing zero noticeable latency overhead to the edge processor.

### Current vs. Future Implementation
- **CURRENT (Software Prototype):** In-memory simulated state (`light_state: bool`, simulated device telemetry, formatted terminal output).
- **FUTURE (ESP32-S3 Hardware Deployment):** GPIO pin level shifting (e.g. GPIO4 driving transistor/relay circuit for 230V laboratory illumination), I2C sensor reading for ambient temperature/status, and UART/MQTT broadcast.

---

## 7. What Remains Before ESP32-S3 Hardware Deployment

To transition from this working software prototype to physical ESP32-S3 hardware:

1. **Hardware Procurement:**
   - ESP32-S3 development board (recommended: ESP32-S3-WROOM-1 with 8MB Octal PSRAM)
   - I2S digital MEMS microphone module (e.g., INMP441, SPH0645, or MSM261S4030H0)
   - 3.3V power regulator & decouple capacitor
2. **I2S DMA Audio Driver:**
   - Configure ESP-IDF I2S driver in standard RX mode at 16,000 Hz, 16-bit or 32-bit (shifted to 16-bit) mono PCM.
   - Set up dual ring buffers using DMA interrupts.
3. **Firmware Feature Extraction:**
   - Implement fixed-point or single-precision FFT using `esp-dsp` (`dsps_fft2r_fc32`).
   - Implement Mel filterbank matrix multiplication using precomputed coefficients from `software/app/features.py`.
4. **INT8 Neural Network Deployment:**
   - Integrate exported `models/nexora_model_weights.h` into ESP-IDF project.
   - Use ESP-NN vector acceleration instructions (Xtensa PIE instructions) to run the 3-layer MLP in under 5 ms on the 240 MHz core.
5. **Downstream Gateway / UART:**
   - Upon confirmed wake word on the ESP32-S3, stream raw audio over Wi-Fi / WebSockets / UART to the local command processing edge hub.
