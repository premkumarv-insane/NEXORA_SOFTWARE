"""
NEXORA Action Executor & Response Layer
Executes safe, simulated edge actions in response to classified user intents.
Guarantees absolute safety boundaries:
- Software-only simulation (No GPIO or physical hardware access in prototype)
- Whitelisted intent dispatch (Zero eval, exec, shell, or arbitrary code execution)
- Safe fallback for unknown intents ([ACTION] NONE)
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

try:
    from software.app.intent import IntentGatekeeper
except ImportError:
    from intent import IntentGatekeeper


@dataclass
class ActionResult:
    action: str
    action_executed: bool
    status_message: str
    response_text: str
    device_state: Dict[str, Any]
    execution_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ActionExecutor:
    """
    Simulated Edge Device Action Executor.
    Maintains local state for simulated smart-laboratory actuators (e.g. Lights, Status).
    """

    def __init__(self):
        # Simulated device peripheral states
        self.light_state = False  # False = OFF, True = ON
        self.total_actions = 0
        self.last_action_timestamp = 0.0

    def get_device_state(self) -> Dict[str, Any]:
        """Returns the current state dictionary of simulated hardware."""
        return {
            "light": "ON" if self.light_state else "OFF",
            "total_actions_executed": self.total_actions,
        }

    def execute(self, intent: str, context: Optional[Dict[str, Any]] = None) -> ActionResult:
        """
        Safely executes the action associated with the given intent.
        Strict whitelist dispatch — no arbitrary code interpretation.
        """
        t0 = time.perf_counter()
        context = context or {}

        # 1. LIGHT_ON
        if intent == IntentGatekeeper.INTENT_LIGHT_ON:
            self.light_state = True
            self.total_actions += 1
            self.last_action_timestamp = time.time()
            action_name = "LIGHT_ON"
            status_msg = "Simulated light switched ON"
            response_msg = "Light turned on."
            action_executed = True

        # 2. LIGHT_OFF
        elif intent == IntentGatekeeper.INTENT_LIGHT_OFF:
            self.light_state = False
            self.total_actions += 1
            self.last_action_timestamp = time.time()
            action_name = "LIGHT_OFF"
            status_msg = "Simulated light switched OFF"
            response_msg = "Light turned off."
            action_executed = True

        # 3. GREETING
        elif intent == IntentGatekeeper.INTENT_GREETING:
            action_name = "NONE"
            status_msg = "Greeting acknowledged"
            response_msg = "I'm ready and listening."
            action_executed = False

        # 4. SYSTEM_STATUS
        elif intent == IntentGatekeeper.INTENT_SYSTEM_STATUS:
            kws_loaded = context.get("kws_loaded", True)
            asr_engine = context.get("asr_engine", "Offline Local Vosk")
            light_str = "ON" if self.light_state else "OFF"
            action_name = "SYSTEM_STATUS"
            status_msg = (
                f"KWS: {'Loaded' if kws_loaded else 'Uninitialized'} | "
                f"ASR: {asr_engine} | Light: {light_str}"
            )
            response_msg = (
                f"All systems operational. Microphone active, KWS ready, "
                f"offline ASR online, simulated light is {light_str}."
            )
            action_executed = True

        # 5. HELP
        elif intent == IntentGatekeeper.INTENT_HELP:
            action_name = "HELP"
            status_msg = "Supported commands catalog listed"
            response_msg = (
                "Available commands: 'turn on the lights', 'turn off the lights', "
                "'system status', 'how are you', 'help'."
            )
            action_executed = True

        # 6. UNKNOWN_INTENT / Fallback (Zero Execution Safety Boundary)
        else:
            action_name = "NONE"
            status_msg = "No action performed (intent unrecognized)"
            response_msg = "Command not recognized. Say 'help' for available commands."
            action_executed = False

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        return ActionResult(
            action=action_name,
            action_executed=action_executed,
            status_message=status_msg,
            response_text=response_msg,
            device_state=self.get_device_state(),
            execution_latency_ms=latency_ms,
        )
