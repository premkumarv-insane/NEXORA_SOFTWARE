"""
NEXORA Bootstrap Speech Sample Generator
Generates initial seed acoustic speech samples for NEXORA and UNKNOWN confusers
using Windows SAPI SpeechSynthesizer configured for 16 kHz Mono PCM.
This provides a baseline dataset allowing immediate training, validation,
and end-to-end testing of the software pipeline prior to user mic collection.

Note:
All generated files are clearly labeled with sample rate: 16000 Hz, 1 channel, 16-bit PCM.
Users can record additional live microphone samples anytime with collect_samples.py.
"""

import os
import sys
import tempfile
import subprocess
import soundfile as sf
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from software.app import config

NEXORA_VARIATIONS = [
    ("Nexora", -3, 80),
    ("Nexora", -2, 90),
    ("Nexora", -1, 100),
    ("Nexora", 0, 75),
    ("Nexora", 0, 90),
    ("Nexora", 0, 100),
    ("Nexora", 1, 80),
    ("Nexora", 1, 95),
    ("Nexora", 2, 85),
    ("Nexora", 2, 100),
    ("Nexora", 3, 90),
    ("Nex-o-ra", -1, 90),
    ("Nex-o-ra", 0, 100),
    ("Nex-o-ra", 1, 85),
    ("NEXORA", 0, 95),
    ("NEXORA", 2, 100),
    ("Nexora", -2, 70),
    ("Nexora", 1, 70),
    ("Nexora", -1, 85),
    ("Nexora", 3, 100),
]

UNKNOWN_VARIATIONS = [
    ("Nebula", 0, 90),
    ("Nebula", 1, 85),
    ("Nectar", 0, 90),
    ("Nectar", -1, 85),
    ("Nokia", 0, 90),
    ("Nokia", 2, 95),
    ("Aurora", 0, 90),
    ("Aurora", -2, 85),
    ("Pandora", 0, 90),
    ("Pandora", 1, 95),
    ("Alexa", 0, 90),
    ("Alexa", -1, 85),
    ("Next one", 0, 90),
    ("Next one", 2, 95),
    ("Excellent", 0, 90),
    ("Computer", 0, 90),
    ("Assistant", 0, 90),
    ("Turn on the lights", 0, 90),
    ("Laboratory lights", 0, 90),
    ("Check temperature", 0, 90),
    ("Stop operation", 0, 90),
    ("System online", 0, 90),
    ("Hello there", 0, 90),
    ("What time is it", 0, 90),
    ("Emergency alert", 0, 90),
]


def render_tts_wav(text: str, rate: int, volume: int, target_path: Path):
    """
    Renders speech audio to 16 kHz Mono WAV using PowerShell System.Speech.
    """
    path_str = str(target_path).replace("\\", "/")
    ps_script = f"""
Add-Type -AssemblyName System.Speech
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = {rate}
$synth.Volume = {volume}
$synth.SetOutputToWaveFile('{path_str}', $format)
$synth.Speak('{text}')
$synth.Dispose()
"""
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False) as f:
        f.write(ps_script)
        script_file = f.name

    try:
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", script_file], check=True, capture_output=True)
    finally:
        if os.path.exists(script_file):
            os.remove(script_file)

    # Validate output file
    if target_path.exists():
        data, sr = sf.read(str(target_path), dtype="float32")
        # Ensure audio is padded or formatted cleanly
        if sr != 16000 or data.ndim > 1:
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            sf.write(str(target_path), data, 16000, subtype="PCM_16")


def bootstrap_dataset():
    print("==================================================")
    print(" NEXORA BOOTSTRAP DATASET GENERATION (16 kHz Mono)")
    print("==================================================")

    nexora_dir = config.DATASET_DIR / "nexora"
    unknown_dir = config.DATASET_DIR / "unknown"
    nexora_dir.mkdir(parents=True, exist_ok=True)
    unknown_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(NEXORA_VARIATIONS)} acoustic speech samples for NEXORA...")
    for idx, (text, rate, vol) in enumerate(NEXORA_VARIATIONS, 1):
        filename = f"nexora_seed_{idx:03d}.wav"
        filepath = nexora_dir / filename
        render_tts_wav(text, rate, vol, filepath)
        print(f"  [NEXORA {idx:02d}/{len(NEXORA_VARIATIONS):02d}] '{text}' (rate={rate}, vol={vol}) -> {filename}")

    print(f"\nGenerating {len(UNKNOWN_VARIATIONS)} acoustic speech samples for UNKNOWN...")
    for idx, (text, rate, vol) in enumerate(UNKNOWN_VARIATIONS, 1):
        filename = f"unknown_seed_{idx:03d}.wav"
        filepath = unknown_dir / filename
        render_tts_wav(text, rate, vol, filepath)
        print(f"  [UNKNOWN {idx:02d}/{len(UNKNOWN_VARIATIONS):02d}] '{text}' (rate={rate}, vol={vol}) -> {filename}")

    print("\n[SUCCESS] Bootstrap dataset generated successfully.")
    print("Dataset directory:")
    print(f"  - NEXORA : {len(list(nexora_dir.glob('*.wav')))} samples")
    print(f"  - UNKNOWN: {len(list(unknown_dir.glob('*.wav')))} samples")
    print(f"  - SILENCE: {len(list((config.DATASET_DIR / 'silence').glob('*.wav')))} samples")


if __name__ == "__main__":
    bootstrap_dataset()
