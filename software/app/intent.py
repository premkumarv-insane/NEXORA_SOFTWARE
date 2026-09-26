"""
NEXORA Edge Intent Gatekeeper
Lightweight, deterministic local intent classification module.
Classifies ASR transcripts into supported action intents without cloud APIs or LLMs.

Supported Intents:
1. GREETING       - Conversational greetings ("hello", "hi", "how are you")
2. LIGHT_ON       - Commands to activate lights ("turn on the light", "switch on lights")
3. LIGHT_OFF      - Commands to deactivate lights ("turn off the light", "switch off lights")
4. SYSTEM_STATUS  - Queries regarding device health ("system status", "are you working")
5. HELP           - Request for supported commands ("help", "what can you do")
6. UNKNOWN_INTENT - Unrecognized commands (safely rejected)
"""

import re
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class IntentResult:
    intent: str
    confidence: float
    confidence_label: str
    matched_rule: str
    raw_transcript: str
    normalized_transcript: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IntentGatekeeper:
    """
    Deterministic rule-based intent gatekeeper designed for low-latency edge deployment.
    Architecture allows future drop-in replacement with TinyML / embedded intent models.
    """

    INTENT_GREETING = "GREETING"
    INTENT_LIGHT_ON = "LIGHT_ON"
    INTENT_LIGHT_OFF = "LIGHT_OFF"
    INTENT_SYSTEM_STATUS = "SYSTEM_STATUS"
    INTENT_HELP = "HELP"
    INTENT_UNKNOWN = "UNKNOWN_INTENT"

    def __init__(self):
        # Compile patterns for fast edge matching
        self._patterns = [
            # 1. LIGHT_ON patterns
            (
                self.INTENT_LIGHT_ON,
                re.compile(
                    r"\b(turn\s+on\s+(the\s+)?lights?|switch\s+on\s+(the\s+)?lights?|turn\s+(the\s+)?lights?\s+on|lights?\s+on|enable\s+(the\s+)?lights?)\b",
                    re.IGNORECASE,
                ),
                1.0,
            ),
            # 2. LIGHT_OFF patterns
            (
                self.INTENT_LIGHT_OFF,
                re.compile(
                    r"\b(turn\s+off\s+(the\s+)?lights?|switch\s+off\s+(the\s+)?lights?|turn\s+(the\s+)?lights?\s+off|lights?\s+off|disable\s+(the\s+)?lights?)\b",
                    re.IGNORECASE,
                ),
                1.0,
            ),
            # 3. GREETING patterns
            (
                self.INTENT_GREETING,
                re.compile(
                    r"\b(hello|hi|hey|how\s+are\s+you|good\s+(morning|afternoon|evening)|greetings)\b",
                    re.IGNORECASE,
                ),
                0.95,
            ),
            # 4. SYSTEM_STATUS patterns
            (
                self.INTENT_SYSTEM_STATUS,
                re.compile(
                    r"\b(what\s+is\s+your\s+status|system\s+status|are\s+you\s+working|status\s+report|check\s+status|device\s+status)\b",
                    re.IGNORECASE,
                ),
                0.95,
            ),
            # 5. HELP patterns
            (
                self.INTENT_HELP,
                re.compile(
                    r"\b(help|what\s+can\s+you\s+do|show\s+available\s+commands|available\s+commands|commands|options)\b",
                    re.IGNORECASE,
                ),
                0.95,
            ),
        ]

    def normalize(self, text: str) -> str:
        """Normalizes transcript: lowercase, removes punctuation, trims extra whitespace."""
        if not text:
            return ""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return " ".join(cleaned.split())

    def classify(self, transcript: str) -> IntentResult:
        """
        Classifies an ASR transcript into a structured IntentResult.
        Guarantees deterministic execution and immediate rejection of unsupported speech.
        """
        normalized = self.normalize(transcript)

        if not normalized or normalized in ["no audible speech detected", "silence"]:
            return IntentResult(
                intent=self.INTENT_UNKNOWN,
                confidence=0.0,
                confidence_label="NONE",
                matched_rule="empty_or_silence",
                raw_transcript=transcript,
                normalized_transcript=normalized,
            )

        # Rule evaluation in priority order
        for intent, pattern, base_conf in self._patterns:
            match = pattern.search(normalized)
            if match:
                matched_str = match.group(0)
                # If exact full match, confidence is highest
                is_exact = matched_str.strip() == normalized.strip()
                conf = 1.0 if is_exact else base_conf
                conf_label = "HIGH" if conf >= 0.90 else "MEDIUM"
                return IntentResult(
                    intent=intent,
                    confidence=conf,
                    confidence_label=conf_label,
                    matched_rule=f"regex:{pattern.pattern}",
                    raw_transcript=transcript,
                    normalized_transcript=normalized,
                )

        # Fallback to UNKNOWN_INTENT
        return IntentResult(
            intent=self.INTENT_UNKNOWN,
            confidence=0.0,
            confidence_label="LOW",
            matched_rule="no_matching_pattern",
            raw_transcript=transcript,
            normalized_transcript=normalized,
        )
