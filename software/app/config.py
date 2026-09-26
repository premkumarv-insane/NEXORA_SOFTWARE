"""
NEXORA System Configuration
Defines central parameters for audio processing, feature extraction,
KWS classification, temporal verification, command capture, and local ASR.
Compatible with eventual ESP32-S3 + I2S MEMS + TinyML deployment.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SOFTWARE_DIR = BASE_DIR / "software"
APP_DIR = SOFTWARE_DIR / "app"
DATASET_DIR = SOFTWARE_DIR / "dataset"
TRAINING_DIR = SOFTWARE_DIR / "training"
MODELS_DIR = BASE_DIR / "models"
RECORDINGS_DIR = BASE_DIR / "recordings"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DATASET_DIR, MODELS_DIR, RECORDINGS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
for cls_dir in ["nexora", "unknown", "silence"]:
    (DATASET_DIR / cls_dir).mkdir(parents=True, exist_ok=True)

# Audio Configuration
SAMPLE_RATE = 16000          # 16 kHz standard for speech / TinyML
CHANNELS = 1                 # Mono
DTYPE = "float32"
WINDOW_DURATION_SEC = 1.0    # 1.0 second sliding window for KWS
STEP_DURATION_SEC = 0.20     # 200 ms inference hop in real-time mode
SAMPLES_PER_WINDOW = int(SAMPLE_RATE * WINDOW_DURATION_SEC)  # 16000 samples
SAMPLES_PER_STEP = int(SAMPLE_RATE * STEP_DURATION_SEC)      # 3200 samples

# Feature Extraction Configuration (TinyML & ESP32-S3 compatible)
# Framing: 25 ms window, 10 ms step
FRAME_LENGTH_MS = 25.0
FRAME_STEP_MS = 10.0
FRAME_LENGTH_SAMPLES = int(SAMPLE_RATE * (FRAME_LENGTH_MS / 1000.0))  # 400
FRAME_STEP_SAMPLES = int(SAMPLE_RATE * (FRAME_STEP_MS / 1000.0))      # 160
N_FFT = 512                  # 512-point FFT (radix-2)
N_MELS = 40                  # 40 Mel frequency filterbanks
N_MFCC = 13                  # 13 MFCC coefficients (standard for edge KWS)
PREEMPHASIS_COEFF = 0.97     # High-frequency boost pre-emphasis filter
FEATURE_TYPE = "mfcc"        # "mfcc" or "logmel"

# Expected Number of Frames in 1.0 second:
# num_frames = floor((16000 - 400) / 160) + 1 = 98 frames
EXPECTED_NUM_FRAMES = ((SAMPLES_PER_WINDOW - FRAME_LENGTH_SAMPLES) // FRAME_STEP_SAMPLES) + 1
FEATURE_TENSOR_SHAPE = (EXPECTED_NUM_FRAMES, N_MFCC)  # (98, 13)
FLATTENED_FEATURE_DIM = EXPECTED_NUM_FRAMES * N_MFCC  # 1274 features

# Classification Target Classes
CLASSES = ["SILENCE", "UNKNOWN", "NEXORA"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

# Model Filepaths & Metadata
MODEL_PATH = MODELS_DIR / "nexora_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
DATASET_REPORT_PATH = DATASET_DIR / "dataset_report.json"
EVALUATION_REPORT_PATH = LOGS_DIR / "evaluation_report.json"
CONFUSION_MATRIX_IMAGE = LOGS_DIR / "confusion_matrix.png"

# KWS Inference & Thresholding
KWS_CONFIDENCE_THRESHOLD = 0.75  # Minimum probability for candidate trigger

# Temporal Verification
TEMPORAL_CONSECUTIVE_FRAMES = 2  # Must exceed threshold for N consecutive frames
TEMPORAL_WINDOW_SIZE = 4         # Lookback window for temporal smoothing
TEMPORAL_COOLDOWN_SEC = 2.5      # Cooldown duration after confirmed activation

# Command Capture
COMMAND_DURATION_SEC = 4.0       # Duration of command recording after wake word
COMMAND_VAD_ENABLED = True       # Voice activity detection to stop on silence
COMMAND_SILENCE_TIMEOUT_SEC = 1.2 # Silence duration to terminate command capture early
COMMAND_MIN_DURATION_SEC = 1.0   # Minimum command audio duration

# Local Offline ASR
ASR_ENGINE = "vosk"              # 100% offline open-source ASR engine
ASR_LANG = "en-us"
