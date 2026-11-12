"""Injection barrier from thermionic emission (Richardson-Schottky model).

Implements the *Richardson-Schottky* analysis of JV Analysis Hub. The forward-bias
dark current in a voltage window is fitted to the diode form

    J = J0 [exp(m V) - 1],

the saturation current density J0 gives the barrier height through the Richardson
equation

    phi = (k_B T / q) ln(A* T^2 / J0),

and, when the active-layer thickness L is known, the image-force lowered barrier is

    phi_eff(V) = phi - sqrt(q E / (4 pi eps_0 eps_r)),   E = V / L.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from ._result import JVHubWarning, ResultMixin, array_field
from .constants import (
    BOLTZMANN,
    ELEMENTARY_CHARGE,
    RICHARDSON_CONSTANT_FREE_ELECTRON,
    RELATIVE_PERMITTIVITY,
    ROOM_TEMPERATURE,
    VACUUM_PERMITTIVITY,
)
from .curve import JVCurve

__all__ = ["RichardsonSchottkyResult", "richardson_schottky", "barrier_height", "image_force_lowered_barrier"]

#: Default effective Richardson constant A* = lambda_R A_0 with lambda_R = 0.5 (about 6.0e5 A m^-2 K^-2).
RICHARDSON_CONSTANT = 0.5 * RICHARDSON_CONSTANT_FREE_ELECTRON


@dataclass(frozen=True)
class RichardsonSchottkyResult(ResultMixin):
    """Result of :func:`richardson_schottky`.

    Attributes
    ----------
    phi_eV
        Zero-field injection barrier height, in eV.
    j0_A_m2
        Saturation current density J0 from the fit, in A/m^2.
    exponent_per_V
        Fitted exponent m of J = J0[exp(mV) - 1], in 1/V.
    ideality_factor
        n = q / (m k_B T). Thermionic emission gives n close to 1 (up to about 2 for real
        interfaces); much larger values mean that the current in the window is not
        limited by the injection barrier, and phi is then not meaningful.
    r_squared
        Coefficient of determination of the fit of J in the window.
    phi_eff_min_eV
        Minimum image-force lowered barrier in the fitted window, in eV
        (``None`` when no thickness was given).
    temperature_K, richardson_constant, thickness_m, relative_permittivity
        Inputs used in the calculation.
    voltage, current_density
        Data inside the fitting window (V, A/m^2).
    voltage_fit, current_density_fit
        The fitted diode curve on 200 points.
    electric_field, phi_eff
        E = V/L (V/m) and phi_eff(V) (eV) for the fitted window (empty without thickness).
    """

    phi_eV: float
    j0_A_m2: float
    exponent_per_V: float
    ideality_factor: float
    r_squared: float
    phi_eff_min_eV: Optional[float]
    temperature_K: float
    richardson_constant: float
    thickness_m: Optional[float]
    relative_permittivity: float
    voltage: np.ndarray = array_field()
    current_density: np.ndarray = array_field()
    voltage_fit: np.ndarray = array_field()
    current_density_fit: np.ndarray = array_field()
    electric_field: np.ndarray = array_field()
    phi_eff: np.ndarray = array_field()


def _diode(voltage, j0, m):
    return j0 * (np.exp(m * voltage) - 1)


def barrier_height(j0: float, temperature: float = ROOM_TEMPERATURE, richardson_constant: float = RICHARDSON_CONSTANT) -> float:
    """Barrier height phi = (k_B T/q) ln(A* T^2 / J0), in eV, for J0 in A/m^2."""
    return float((BOLTZMANN * temperature / ELEMENTARY_CHARGE) * np.log(richardson_constant * temperature**2 / j0))


def image_force_lowered_barrier(phi: float, voltage, thickness: float, relative_permittivity: float = RELATIVE_PERMITTIVITY):
    """Electric field E = V/L and effective barrier phi - sqrt(qE/(4 pi eps_0 eps_r)) (eV)."""
    field = np.asarray(voltage, dtype=float) / thickness
    lowering = np.sqrt(ELEMENTARY_CHARGE * field / (4 * np.pi * VACUUM_PERMITTIVITY * relative_permittivity))
    return field, phi - lowering


def richardson_schottky(
    curve: JVCurve,
    thickness: Optional[float] = None,
    *,
    voltage_window: Tuple[float, float] = (0.2, 0.8),
    temperature: float = ROOM_TEMPERATURE,
    richardson_constant: float = RICHARDSON_CONSTANT,
    relative_permittivity: float = RELATIVE_PERMITTIVITY,
    max_ideality: float = 2.0,
) -> RichardsonSchottkyResult:
    """Injection barrier height of a dark J-V curve from thermionic emission.

    Parameters
    ----------
    curve
        A dark J-V curve.
    thickness
        Active-layer thickness in m. When given, the image-force lowered barrier
        phi_eff(V) is also computed.
    voltage_window
        Open interval (V_min, V_max) of the forward-bias data that is fitted.
    temperature
        Temperature in K (default 298.15 K).
    richardson_constant
        Effective Richardson constant A* in A m^-2 K^-2 (default 0.5 A_0, with A_0 the free-electron
        Richardson constant, about 6.0e5).
    relative_permittivity
        Relative permittivity used for the image-force lowering (default 3.0; use the value of
        your material).
    max_ideality
        Ideality factor above which a warning is issued (default 2).

    Returns
    -------
    RichardsonSchottkyResult

    Warns
    -----
    JVHubWarning
        If the ideality factor exceeds ``max_ideality`` (the thermionic-emission
        interpretation, and therefore phi, is then questionable).
    """
    if thickness is not None and not thickness > 0:
        raise ValueError("thickness must be positive (in metres) or None.")
    low, high = voltage_window
    inside = (curve.voltage > low) & (curve.voltage < high)
    voltage = curve.voltage[inside]
    density = curve.current_density[inside]
    if voltage.size < 3:
        raise ValueError(f"Fewer than 3 data points between {low} V and {high} V; adjust voltage_window.")

    with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", OptimizeWarning)
        (j0, m), _ = curve_fit(_diode, voltage, density, p0=[np.max(density), 1.0], maxfev=10000)

    voltage_fit = np.linspace(voltage.min(), voltage.max(), 200)
    phi = barrier_height(j0, temperature, richardson_constant)
    ideality = float(ELEMENTARY_CHARGE / (m * BOLTZMANN * temperature))
    predicted = _diode(voltage, j0, m)
    r_squared = float(1 - np.sum((density - predicted) ** 2) / np.sum((density - np.mean(density)) ** 2))
    if not 0 < ideality <= max_ideality:
        warnings.warn(
            f"Ideality factor n = {ideality:.3g} in the fitted window ({low}-{high} V): the current there is "
            "not limited by thermionic emission over the injection barrier, so phi is not reliable.",
            JVHubWarning,
            stacklevel=2,
        )

    field = phi_eff = np.array([])
    phi_eff_min = None
    if thickness is not None:
        field, phi_eff = image_force_lowered_barrier(phi, voltage, thickness, relative_permittivity)
        phi_eff_min = float(np.min(phi_eff))

    return RichardsonSchottkyResult(
        phi_eV=phi,
        j0_A_m2=float(j0),
        exponent_per_V=float(m),
        ideality_factor=ideality,
        r_squared=r_squared,
        phi_eff_min_eV=phi_eff_min,
        temperature_K=float(temperature),
        richardson_constant=float(richardson_constant),
        thickness_m=None if thickness is None else float(thickness),
        relative_permittivity=float(relative_permittivity),
        voltage=voltage,
        current_density=density,
        voltage_fit=voltage_fit,
        current_density_fit=_diode(voltage_fit, j0, m),
        electric_field=field,
        phi_eff=phi_eff,
    )
