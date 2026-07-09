"""Explicit cache for costly causal prior forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .result_store import stable_hash


@dataclass
class PriorForecastCache:
    store: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get(self, key: str) -> dict[str, Any] | None:
        value = self.store.get(key)
        return None if value is None else dict(value)

    def set(self, key: str, value: dict[str, Any]) -> None:
        self.store[key] = dict(value)

    def make_key(
        self,
        *,
        prior_name: str,
        prior_params: dict[str, Any] | None,
        history_values: Any,
        horizon: int = 1,
    ) -> str:
        return stable_hash(
            {
                "prior_name": prior_name,
                "prior_params": prior_params or {},
                "history_values": list(map(float, history_values)),
                "horizon": int(horizon),
            }
        )
