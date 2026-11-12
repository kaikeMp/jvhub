"""The :class:`JVCurve` container and unit handling.

Every analysis in jvhub receives a :class:`JVCurve`. Internally a curve is always
stored in SI units -- voltage in volts and current density in A/m^2 -- together
with the device area in m^2, so unit conversions happen in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["JVCurve", "CURRENT_UNITS", "AREA_UNITS"]

#: Accepted units for the current column.
CURRENT_UNITS = ("A", "mA", "A/m2", "mA/cm2")
#: Accepted units for the device area.
AREA_UNITS = ("cm2", "m2")

_UNIT_ALIASES = {
    "a": "A",
    "ma": "mA",
    "a/m2": "A/m2",
    "ma/cm2": "mA/cm2",
    "cm2": "cm2",
    "m2": "m2",
}


def _normalize_unit(unit: str) -> str:
    key = str(unit).strip().replace("²", "2").replace("^2", "2").replace(" ", "").lower()
    if key not in _UNIT_ALIASES:
        raise ValueError(
            f"Unknown unit {unit!r}. Current units: {', '.join(CURRENT_UNITS)}; "
            f"area units: {', '.join(AREA_UNITS)}."
        )
    return _UNIT_ALIASES[key]


def area_to_m2(area: float, unit: str = "cm2") -> float:
    """Convert a device area given in ``unit`` (``"cm2"`` or ``"m2"``) to m^2."""
    unit = _normalize_unit(unit)
    if unit not in AREA_UNITS:
        raise ValueError(f"{unit!r} is not an area unit; use 'cm2' or 'm2'.")
    area = float(area)
    if not np.isfinite(area) or area <= 0:
        raise ValueError(f"The device area must be a positive number, got {area}.")
    return area * 1e-4 if unit == "cm2" else area


def to_current_density(current, unit: str, area_m2: Optional[float]) -> np.ndarray:
    """Convert a current (``"A"``, ``"mA"``) or current density (``"A/m2"``,
    ``"mA/cm2"``) array to current density in A/m^2."""
    unit = _normalize_unit(unit)
    current = np.asarray(current, dtype=float)
    if unit == "A/m2":
        return current.copy()
    if unit == "mA/cm2":
        return current * 1e-3 * 1e4
    if unit not in ("A", "mA"):
        raise ValueError(f"{unit!r} is not a current unit; use one of {CURRENT_UNITS}.")
    if area_m2 is None:
        raise ValueError(f"The device area is required to convert a current given in {unit} to current density.")
    if unit == "mA":
        return (current / 1000) / area_m2
    return current / area_m2


@dataclass(frozen=True)
class JVCurve:
    """A measured current density-voltage curve.

    Parameters
    ----------
    voltage
        Applied voltage in V.
    current_density
        Current density in A/m^2 (1 A/m^2 = 0.1 mA/cm^2), same length as ``voltage``.
    area
        Device (active) area in m^2. Optional, but required by the analyses that work
        with absolute currents (:func:`jvhub.pv_parameters` and
        :func:`jvhub.circuital_model`).

    Notes
    -----
    The arrays are stored as read-only copies, so analyses can never modify the
    caller's data. Use :meth:`from_arrays` or :func:`jvhub.read_jv` to build a curve
    from data in other units.
    """

    voltage: np.ndarray
    current_density: np.ndarray
    area: Optional[float] = None

    def __post_init__(self) -> None:
        v = np.array(self.voltage, dtype=float).ravel()
        j = np.array(self.current_density, dtype=float).ravel()
        if v.shape != j.shape:
            raise ValueError(f"voltage and current_density must have the same length ({v.size} != {j.size}).")
        if v.size < 2:
            raise ValueError("A J-V curve needs at least two points.")
        if not (np.all(np.isfinite(v)) and np.all(np.isfinite(j))):
            raise ValueError("The J-V data contain NaN or infinite values.")
        v.setflags(write=False)
        j.setflags(write=False)
        object.__setattr__(self, "voltage", v)
        object.__setattr__(self, "current_density", j)
        if self.area is not None:
            object.__setattr__(self, "area", area_to_m2(self.area, "m2"))

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_arrays(
        cls,
        voltage,
        current,
        unit: str = "mA",
        area: Optional[float] = None,
        area_unit: str = "cm2",
    ) -> "JVCurve":
        """Build a curve from a voltage array (V) and a current array in any supported unit.

        Parameters
        ----------
        voltage
            Voltage in V.
        current
            Current or current density, in ``unit``.
        unit
            One of ``"A"``, ``"mA"``, ``"A/m2"`` or ``"mA/cm2"``.
        area
            Device area, in ``area_unit``. Required when ``unit`` is ``"A"`` or ``"mA"``.
        area_unit
            ``"cm2"`` (default) or ``"m2"``.

        Examples
        --------
        >>> curve = JVCurve.from_arrays([0.0, 0.5, 1.0], [-1.2, -1.0, 0.8], unit="mA", area=0.09)
        >>> round(float(curve.current_density_mA_cm2[0]), 3)
        -13.333
        """
        area_m2 = None if area is None else area_to_m2(area, area_unit)
        return cls(np.asarray(voltage, dtype=float), to_current_density(current, unit, area_m2), area_m2)

    # --------------------------------------------------------------- properties
    @property
    def area_cm2(self) -> Optional[float]:
        """Device area in cm^2 (``None`` if unknown)."""
        return None if self.area is None else self.area * 1e4

    @property
    def current_density_mA_cm2(self) -> np.ndarray:
        """Current density in mA/cm^2."""
        return self.current_density * 0.1

    @property
    def current(self) -> np.ndarray:
        """Current in A (requires the device area)."""
        return self.current_density * self._require_area()

    @property
    def current_mA(self) -> np.ndarray:
        """Current in mA (requires the device area)."""
        return self.current * 1000

    def _require_area(self, purpose: str = "this quantity") -> float:
        if self.area is None:
            raise ValueError(f"The device area is required for {purpose}; create the curve with area=...")
        return self.area

    def __len__(self) -> int:
        return self.voltage.size

    def to_dataframe(self):
        """Return the curve as a :class:`pandas.DataFrame` with explicit units in the column names."""
        import pandas as pd

        data = {
            "Voltage (V)": self.voltage,
            "Current Density (A/m2)": self.current_density,
            "Current Density (mA/cm2)": self.current_density_mA_cm2,
        }
        if self.area is not None:
            data["Current (mA)"] = self.current_mA
        return pd.DataFrame(data)
