"""Charge-carrier mobility from space-charge-limited current (Mott-Gurney law).

Implements the *Mott-Gurney* analysis of JV Analysis Hub with the dynamic
power-law method of Felekidis et al. [1]_: writing J = k V^m, the local exponent

    m(V) = d ln J / d ln V = (V / J) dJ/dV

is computed for every point and the trap-free SCLC regime is identified at the
voltage V' where m = 2. The mobility follows from the Mott-Gurney law,

    mu = 8 J' L^3 / (9 eps_r eps_0 V'^2).

.. [1] N. Felekidis, A. Melianas, M. Kemerink, *Automated open-source software for
   charge transport analysis in single-carrier organic semiconductor diodes*,
   Org. Electron. 61 (2018) 318-328.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._result import ResultMixin, array_field
from .constants import RELATIVE_PERMITTIVITY, VACUUM_PERMITTIVITY
from .curve import JVCurve

__all__ = ["MottGurneyResult", "mott_gurney", "local_exponent"]


@dataclass(frozen=True)
class MottGurneyResult(ResultMixin):
    """Result of :func:`mott_gurney`.

    Attributes
    ----------
    mobility_cm2_Vs
        Charge-carrier mobility, in cm^2/(V s).
    voltage_V
        Voltage V' at which m(V) is closest to the target exponent (2), in V.
    current_density_A_m2
        Current density J' at V', in A/m^2.
    exponent
        The value of m(V') actually found (ideally close to 2).
    thickness_m, relative_permittivity
        Inputs used in the calculation.
    voltage, current_density, m
        Positive-voltage data (V, A/m^2) and the local exponent m(V) for each point.
    """

    mobility_cm2_Vs: float
    voltage_V: float
    current_density_A_m2: float
    exponent: float
    thickness_m: float
    relative_permittivity: float
    voltage: np.ndarray = array_field()
    current_density: np.ndarray = array_field()
    m: np.ndarray = array_field()


def local_exponent(voltage, current_density) -> np.ndarray:
    """Local power-law exponent m(V) = (V/J) dJ/dV, with dJ/dV from ``numpy.gradient``.

    Points where J = 0 give ``inf``/``nan`` and are ignored by :func:`mott_gurney`.
    """
    voltage = np.asarray(voltage, dtype=float)
    current_density = np.asarray(current_density, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.gradient(current_density, voltage)
        return voltage / current_density * slope


def mott_gurney(
    curve: JVCurve,
    thickness: float,
    *,
    relative_permittivity: float = RELATIVE_PERMITTIVITY,
    target_exponent: float = 2.0,
) -> MottGurneyResult:
    """Carrier mobility of a single-carrier (dark) J-V curve from the Mott-Gurney law.

    Only points with V > 0 are used. The point whose local exponent m(V) is closest to
    ``target_exponent`` gives (V', J') and the mobility
    ``mu = (8/9) L^3 J' / (eps_0 eps_r V'^2)``.

    Parameters
    ----------
    curve
        A dark J-V curve.
    thickness
        Active-layer thickness L, in m (e.g. ``100e-9`` for 100 nm).
    relative_permittivity
        Relative permittivity of the active layer (default 3.0; use the value of your material).
    target_exponent
        Exponent identifying the trap-free SCLC regime (2 for Mott-Gurney).

    Returns
    -------
    MottGurneyResult
    """
    if not thickness > 0:
        raise ValueError("thickness must be positive (in metres, e.g. 100e-9).")
    positive = curve.voltage > 0
    voltage = curve.voltage[positive]
    density = curve.current_density[positive]
    if voltage.size < 2:
        raise ValueError("The Mott-Gurney analysis needs at least two points with V > 0.")

    m = local_exponent(voltage, density)
    distance = np.abs(m - target_exponent)
    if np.all(np.isnan(distance)):
        raise ValueError("m(V) could not be computed for any point (is the current density zero?).")
    i = int(np.nanargmin(distance))
    v_prime, j_prime = float(voltage[i]), float(density[i])

    mobility = (8 / 9) * (thickness**3) / (VACUUM_PERMITTIVITY * relative_permittivity) * j_prime / (v_prime**2)
    return MottGurneyResult(
        mobility_cm2_Vs=float(mobility * 1e4),
        voltage_V=v_prime,
        current_density_A_m2=j_prime,
        exponent=float(m[i]),
        thickness_m=float(thickness),
        relative_permittivity=float(relative_permittivity),
        voltage=voltage,
        current_density=density,
        m=m,
    )
