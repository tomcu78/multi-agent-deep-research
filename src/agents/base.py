"""Base agent class defining common lifecycle, telemetry, and logging."""
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from src.llm import get_llm

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for specialized autonomous research agents."""
    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        llm: Optional[Any] = None,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm = llm or get_llm()
        self.event_callback = event_callback

    def emit_event(self, event_type: str, message: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Emit a structured event for UI/streaming and state activity logs."""
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "role": self.role,
            "type": event_type,
            "message": message,
            "payload": payload or {}
        }
        logger.info(f"[{self.name}] {message}")
        if self.event_callback:
            try:
                self.event_callback(event_data)
            except Exception as e:
                logger.warning(f"Error in event callback: {e}")
        return event_data
