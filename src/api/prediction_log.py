"""Append-only JSONL log of served predictions. Phase 4 drift monitoring reads this file."""
from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)


class PredictionLog:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        """Never raises: a full disk must not turn a good prediction into a 500."""
        if self.path is None:
            return
        line = json.dumps({"ts": datetime.now(UTC).isoformat(), **record}, separators=(",", ":"), default=str)
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError:
            LOG.exception("Could not write prediction log %s", self.path)
