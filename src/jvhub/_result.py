"""Common behaviour of the analysis result objects."""

from __future__ import annotations

from dataclasses import field, fields
from typing import Any, Dict

import numpy as np


class JVHubWarning(UserWarning):
    """Warning issued by jvhub analyses (e.g. a fitted parameter not determined by the data)."""


def array_field():
    """A dataclass field holding an array: excluded from ``repr`` and ``to_dict``."""
    return field(repr=False, metadata={"array": True})


class ResultMixin:
    """Adds :meth:`to_dict` and read-only arrays to frozen result dataclasses."""

    def __post_init__(self) -> None:
        for f in fields(self):  # type: ignore[arg-type]
            if f.metadata.get("array"):
                value = np.array(getattr(self, f.name), dtype=float)
                value.setflags(write=False)
                object.__setattr__(self, f.name, value)

    def to_dict(self) -> Dict[str, Any]:
        """Scalar results as a plain ``dict`` (handy for building tables with pandas)."""
        return {f.name: getattr(self, f.name) for f in fields(self) if not f.metadata.get("array")}  # type: ignore[arg-type]
