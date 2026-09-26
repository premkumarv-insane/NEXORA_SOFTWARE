"""
NEXORA TinyML Model Exporter
Exports trained KWS neural network weights and normalization parameters to:
1. C Header file (models/nexora_model_weights.h) for direct ESP32-S3 firmware inclusion.
2. Quantized INT8 parameters and scaling metadata.
"""

import sys
import json
import joblib
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from software.app import config


def export_c_header(
    model_path: Path = config.MODEL_PATH,
    output_header: Path = config.MODELS_DIR / "nexora_model_weights.h"
):
    """
    Exports model parameters as C arrays for ESP32-S3 deployment.
    """
    print(f"Exporting model from {model_path} to {output_header}...")
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_path} not found.")

    pipeline = joblib.load(str(model_path))
    scaler = pipeline.named_steps["scaler"]
    mlp = pipeline.named_steps["classifier"]

    # Scaler params
    mean = scaler.mean_
    scale = scaler.scale_

    # Weights and biases
    coefs = mlp.coefs_          # [W1, W2, W3]
    intercepts = mlp.intercepts_ # [b1, b2, b3]

    header_lines = [
        "/* NEXORA WAKE WORD DETECTION MODEL WEIGHTS */",
        "/* Auto-generated for ESP32-S3 / TinyML deployment */",
        f"/* Input features: {config.FLATTENED_FEATURE_DIM} (MFCC 13 x 98 frames) */",
        f"/* Classes: {', '.join(config.CLASSES)} */",
        "#ifndef NEXORA_MODEL_WEIGHTS_H",
        "#define NEXORA_MODEL_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define NEXORA_INPUT_DIM {config.FLATTENED_FEATURE_DIM}",
        f"#define NEXORA_NUM_CLASSES {len(config.CLASSES)}",
        f"#define NEXORA_LAYER1_OUT {coefs[0].shape[1]}",
        f"#define NEXORA_LAYER2_OUT {coefs[1].shape[1]}",
        "",
        "/* Feature Normalization (StandardScaler) */",
    ]

    def format_c_array(name: str, arr: np.ndarray, dtype: str = "float") -> str:
        flat = arr.flatten()
        elems = ", ".join(f"{v:.7f}f" for v in flat)
        return f"static const {dtype} {name}[{flat.size}] = {{\n    {elems}\n}};\n"

    header_lines.append(format_c_array("nexora_scaler_mean", mean))
    header_lines.append(format_c_array("nexora_scaler_scale", scale))

    for idx, (w, b) in enumerate(zip(coefs, intercepts), 1):
        header_lines.append(f"/* Layer {idx}: Shape ({w.shape[0]}, {w.shape[1]}) */")
        header_lines.append(format_c_array(f"nexora_w{idx}", w))
        header_lines.append(format_c_array(f"nexora_b{idx}", b))

    # INT8 Quantization calculation for Layer 1
    w1_max = np.max(np.abs(coefs[0]))
    scale_w1 = 127.0 / (w1_max + 1e-8)
    w1_int8 = np.clip(np.round(coefs[0] * scale_w1), -128, 127).astype(np.int8)

    header_lines.append("/* INT8 Quantization Reference for Layer 1 */")
    header_lines.append(f"#define NEXORA_W1_INT8_SCALE {scale_w1:.6f}f")
    int8_elems = ", ".join(str(v) for v in w1_int8.flatten()[:100]) # First 100 sample
    header_lines.append(f"// Sample 100 INT8 quantized weights:\n// {int8_elems} ...\n")

    header_lines.append("#endif /* NEXORA_MODEL_WEIGHTS_H */\n")

    output_header.parent.mkdir(parents=True, exist_ok=True)
    with open(output_header, "w") as f:
        f.write("\n".join(header_lines))

    print(f"[OK] Exported C Header to: {output_header}")


if __name__ == "__main__":
    export_c_header()
