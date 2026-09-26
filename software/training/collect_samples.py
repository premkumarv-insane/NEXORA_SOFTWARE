"""
NEXORA Audio Sample Collection Tool
Records 16 kHz Mono PCM WAV audio files from the laptop microphone.
Collects data for 3 classes:
- NEXORA  : Target wake word with varied distances, volumes, speeds, positions
- UNKNOWN : Negative words/phrases, including phonetically similar confusers
- SILENCE : Background ambient conditions, fan, keyboard, distance room noise
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf

# Allow imports from parent project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from software.app import config

NEXORA_PROMPTS = [
    "Say 'NEXORA' (Normal conversational volume, centered)",
    "Say 'NEXORA' (Slightly louder volume, 1 meter away)",
    "Say 'NEXORA' (Quiet / soft volume, close to mic)",
    "Say 'NEXORA' (Fast pace: 'Nex-o-ra')",
    "Say 'NEXORA' (Deliberate pace: 'NEX - O - RA')",
    "Say 'NEXORA' (From left side of laptop)",
    "Say 'NEXORA' (From right side of laptop)",
    "Say 'NEXORA' (High pitch voice tone)",
    "Say 'NEXORA' (Deep voice tone)",
    "Say 'NEXORA' (Natural speaking tone with brief pause before and after)",
]

UNKNOWN_PROMPTS = [
    "Say 'NEBULA' (Confuser - similar prefix)",
    "Say 'NECTAR' (Confuser - similar prefix)",
    "Say 'NOKIA' (Confuser - similar phonetics)",
    "Say 'AURORA' (Confuser - similar ending)",
    "Say 'PANDORA' (Confuser - similar ending)",
    "Say 'ALEXA' (Alternative wake word)",
    "Say 'NEXT ONE' (Common command start)",
    "Say 'EXCELLENT' (Phonetically overlapping sounds)",
    "Say 'TURN ON THE LIGHT' (Command phrase)",
    "Say 'CHECK TEMPERATURE' (Command phrase)",
    "Say 'OPEN LABORATORY' (Command phrase)",
    "Say 'COMPUTER' (Alternative keyword)",
    "Say 'ASSISTANT' (Alternative keyword)",
    "Say 'HELLO SYSTEM' (Common greeting)",
    "Say 'ONE TWO THREE' (Numbers)",
]

SILENCE_PROMPTS = [
    "Remain completely silent (Ambient room noise)",
    "Remain silent with laptop fan running",
    "Type moderately on your laptop keyboard (Typing noise)",
    "Shuffle papers or touch the desk near the laptop (Desk handling noise)",
    "Room with background distant chatter or ambient hum",
    "Click mouse buttons repeatedly",
    "Breathe normally near the microphone",
]


def test_microphone():
    """Verify that the microphone can be opened and captured."""
    print("Testing microphone access...")
    try:
        duration = 1.0
        recording = sd.rec(
            int(duration * config.SAMPLE_RATE),
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype="float32"
        )
        sd.wait()
        rms = np.sqrt(np.mean(recording ** 2))
        max_val = np.max(np.abs(recording))
        print(f"[OK] Microphone responsive! Signal RMS: {rms:.4f}, Peak: {max_val:.4f}")
        return True
    except Exception as e:
        print(f"[ERROR] Microphone test failed: {e}")
        return False


def record_sample(
    duration: float = 1.5,
    sample_rate: int = config.SAMPLE_RATE,
    channels: int = config.CHANNELS
) -> np.ndarray:
    """Records a single audio sample from the default microphone."""
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32"
    )
    sd.wait()
    return audio.flatten()


def save_wav(audio: np.ndarray, filepath: Path, sample_rate: int = config.SAMPLE_RATE):
    """Saves audio buffer as 16 kHz Mono 16-bit PCM WAV."""
    # Ensure -1.0 to 1.0 bounds
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(filepath), audio, sample_rate, subtype="PCM_16")


def collect_class_interactive(cls_name: str, count: int, duration: float):
    """Guides user through interactive recording for a target class."""
    target_dir = config.DATASET_DIR / cls_name.lower()
    target_dir.mkdir(parents=True, exist_ok=True)

    if cls_name == "nexora":
        prompts = NEXORA_PROMPTS
    elif cls_name == "unknown":
        prompts = UNKNOWN_PROMPTS
    else:
        prompts = SILENCE_PROMPTS

    print("=" * 60)
    print(f"COLLECTING SAMPLES FOR CLASS: [{cls_name.upper()}]")
    print(f"Destination: {target_dir}")
    print(f"Target Count: {count} samples | Duration per sample: {duration}s")
    print("=" * 60)

    recorded = 0
    for i in range(count):
        prompt = prompts[i % len(prompts)]
        print(f"\n[{i + 1}/{count}] Prompt: >>> {prompt} <<<")
        input("Press [ENTER] to record (speak immediately when recording starts)... ")

        print("  ● RECORDING...", end="", flush=True)
        audio = record_sample(duration=duration)
        print(" DONE!")

        rms = np.sqrt(np.mean(audio ** 2))
        max_val = np.max(np.abs(audio))

        # Check for potential audio issues
        if cls_name != "silence" and max_val < 0.02:
            print("  [WARNING] Audio signal very faint. Please speak closer or louder.")
        elif max_val > 0.98:
            print("  [WARNING] Possible clipping detected. Consider lowering mic gain.")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{cls_name.lower()}_{timestamp}_{i+1:03d}.wav"
        filepath = target_dir / filename
        save_wav(audio, filepath)
        recorded += 1
        print(f"  ✓ Saved to: {filename} (RMS: {rms:.3f}, Peak: {max_val:.3f})")

    print(f"\n[COMPLETED] Collected {recorded} samples for {cls_name.upper()}.")


def main():
    parser = argparse.ArgumentParser(description="NEXORA Dataset Audio Collection Tool")
    parser.add_argument(
        "--class",
        dest="target_class",
        choices=["nexora", "unknown", "silence", "all"],
        default="all",
        help="Audio class to record"
    )
    parser.add_argument("--count", type=int, default=10, help="Number of samples to collect per class")
    parser.add_argument("--duration", type=float, default=1.5, help="Duration of each recording in seconds")
    parser.add_argument("--test-mic", action="store_true", help="Test microphone and exit")
    parser.add_argument("--auto-silence", action="store_true", help="Record silence samples without pressing enter each time")

    args = parser.parse_args()

    print("==================================================")
    print(" NEXORA DATA COLLECTION TOOL (16 kHz Mono PCM)")
    print("==================================================")

    if not test_microphone():
        print("[ERROR] Cannot access microphone. Check system settings.")
        sys.exit(1)

    if args.test_mic:
        print("[OK] Microphone test complete.")
        return

    classes_to_record = ["nexora", "unknown", "silence"] if args.target_class == "all" else [args.target_class]

    for cls in classes_to_record:
        if cls == "silence" and args.auto_silence:
            target_dir = config.DATASET_DIR / "silence"
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nCollecting {args.count} silence samples automatically (pause between samples)...")
            for i in range(args.count):
                prompt = SILENCE_PROMPTS[i % len(SILENCE_PROMPTS)]
                print(f"[{i+1}/{args.count}] Recording silence: {prompt}...")
                audio = record_sample(duration=args.duration)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"silence_{timestamp}_{i+1:03d}.wav"
                save_wav(audio, target_dir / filename)
                print(f"  Saved {filename}")
                time.sleep(0.5)
        else:
            collect_class_interactive(cls, args.count, args.duration)

    print("\n[SUCCESS] Audio collection session finished.")
    print("Next step: Run 'prepare_dataset.py' to validate and extract features.")


if __name__ == "__main__":
    main()
