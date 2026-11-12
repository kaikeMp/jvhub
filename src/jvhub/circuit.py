"""Equivalent-circuit model of dark J-V curves with a space-charge-limited series element.

Implements the *Circuital Model* of JV Analysis Hub, an approximation based on the
equivalent circuit of Koehler et al. [1]_, with a robust numerical solution. The
device is a diode in parallel with a shunt resistance R_sh, in series with the bulk of
the active layer, whose current is space-charge limited. For an applied voltage V,
with V_SC the voltage dropped across the bulk,

    J = J0 [exp(q (V - V_SC) / k_B T) - 1] + (V - V_SC) / (R_sh A)      (junction)
    V_SC = (1/3) sqrt(8 J L^3 / (eps mu_eff))                            (bulk)

The bulk equation is the Mott-Gurney law, J = 9 eps mu_eff V_SC^2 / (8 L^3). Eliminating
J gives, for every voltage, one equation in V_SC whose left side increases and right
side decreases monotonically on 0 <= V_SC <= V, so it has exactly one root. jvhub finds
it by bisection to machine precision, which makes the model evaluation robust and
deterministic.

(R_sh, J0, mu_eff) are fitted by nonlinear least squares on ln J (log-log fit), with the
logarithms of the parameters as variables -- the parameters are therefore always
positive -- and their standard uncertainties are estimated from the Jacobian.

Notes
-----
* Only the forward-bias data (V > 0 and J > 0) are fitted.
* mu_eff and the thickness L enter the model only through mu_eff / L^3, so the data
  determine that ratio, not L. The thickness must be measured independently, and the
  reported mu_eff scales as L^3. (JV Analysis Hub v1.0.0 scanned L to "optimize" it;
  that scan cannot identify L and is not offered here.)

.. [1] M. Koehler, N. Yamamoto, A. Macedo, D. Grodniski, L. Roman, M. Da Luz,
   *The current-voltage characteristics of polymer/C60 diodes in the dark: a direct way
   to assess photovoltaic devices efficiency parameters*, Appl. Phys. Lett. 103 (2013).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from ._result import JVHubWarning, ResultMixin, array_field
from .constants import BOLTZMANN as _K
from .constants import ELEMENTARY_CHARGE as _Q
from .constants import RELATIVE_PERMITTIVITY, ROOM_TEMPERATURE
from .constants import VACUUM_PERMITTIVITY as _EPS0
from .curve import JVCurve

__all__ = ["CircuitalResult", "circuital_model", "circuital_current_density"]

_MAX_EXPONENT = 700.0  # keeps exp() finite
_BISECTION_STEPS = 200  # far more than needed to reach machine precision on [0, V]
_UNDETERMINED = 1.0  # relative uncertainty above which a parameter is reported as undetermined


@dataclass(frozen=True)
class CircuitalResult(ResultMixin):
    """Result of :func:`circuital_model`.

    Attributes
    ----------
    rsh_ohm, rsh_std_ohm
        Shunt resistance R_sh and its standard uncertainty, in ohm.
    j0_A_m2, j0_std_A_m2
        Saturation current density J0 and its standard uncertainty, in A/m^2.
    mu_eff_cm2_Vs, mu_eff_std_cm2_Vs
        Effective mobility and its standard uncertainty, in cm^2/(V s), for the given
        thickness (mu_eff is proportional to thickness^3).
    r_squared
        Coefficient of determination of J in the fitted region.
    r_squared_log
        Coefficient of determination of ln J (the quantity that is fitted).
    thickness_m, temperature_K, relative_permittivity
        Inputs used in the model.
    n_points
        Number of data points fitted.
    voltage, current_density
        Fitted data: forward bias, V > 0 and J > 0 (V, A/m^2).
    current_density_fit, space_charge_voltage
        Model current density (A/m^2) and voltage across the bulk V_SC (V) at each point.

    Notes
    -----
    The uncertainties are 1-sigma estimates from the curvature of the least-squares
    problem. A relative uncertainty of order 100 % or more means that the data do not
    constrain the parameter; :func:`circuital_model` then issues a
    :class:`~jvhub.JVHubWarning`.
    """

    rsh_ohm: float
    rsh_std_ohm: float
    j0_A_m2: float
    j0_std_A_m2: float
    mu_eff_cm2_Vs: float
    mu_eff_std_cm2_Vs: float
    r_squared: float
    r_squared_log: float
    thickness_m: float
    temperature_K: float
    relative_permittivity: float
    n_points: int
    voltage: np.ndarray = array_field()
    current_density: np.ndarray = array_field()
    current_density_fit: np.ndarray = array_field()
    space_charge_voltage: np.ndarray = array_field()


def _solve(voltage, rsh, j0, mu_eff, thickness, area, temperature, permittivity):
    """Voltage across the bulk V_SC and current density J of the model at each voltage (V > 0)."""
    eps = permittivity * _EPS0
    sclc = 9 * eps * mu_eff / (8 * thickness**3)  # J = sclc * V_SC^2
    thermal = _Q / (_K * temperature)

    def excess(vsc):  # bulk current minus junction current; increasing in vsc
        exponent = np.minimum(thermal * (voltage - vsc), _MAX_EXPONENT)
        return sclc * vsc**2 - (j0 * np.expm1(exponent) + (voltage - vsc) / (rsh * area))

    low = np.zeros_like(voltage)
    high = voltage.copy()
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        positive = excess(middle) > 0
        high = np.where(positive, middle, high)
        low = np.where(positive, low, middle)
        if np.all(high - low <= 4 * np.finfo(float).eps * np.maximum(high, 1e-300)):
            break
    vsc = 0.5 * (low + high)
    return vsc, sclc * vsc**2


def circuital_current_density(
    voltage,
    rsh: float,
    j0: float,
    mu_eff: float,
    thickness: float,
    area: float,
    temperature: float = ROOM_TEMPERATURE,
    relative_permittivity: float = RELATIVE_PERMITTIVITY,
) -> np.ndarray:
    """Current density (A/m^2) predicted by the circuital model for forward voltages.

    Parameters
    ----------
    voltage
        Applied voltages in V (must be > 0).
    rsh
        Shunt resistance in ohm.
    j0
        Saturation current density in A/m^2.
    mu_eff
        Effective mobility in m^2/(V s).
    thickness, area
        Active-layer thickness (m) and device area (m^2).
    temperature, relative_permittivity
        Temperature (K) and relative permittivity of the active layer.
    """
    voltage = np.asarray(voltage, dtype=float)
    if np.any(voltage <= 0):
        raise ValueError("The circuital model is evaluated for forward voltages (V > 0) only.")
    if min(rsh, j0, mu_eff, thickness, area) <= 0:
        raise ValueError("rsh, j0, mu_eff, thickness and area must be positive.")
    return _solve(voltage, rsh, j0, mu_eff, thickness, area, temperature, relative_permittivity)[1]


def _r_squared(observed, predicted) -> float:
    return float(1 - np.sum((observed - predicted) ** 2) / np.sum((observed - np.mean(observed)) ** 2))


def circuital_model(
    curve: JVCurve,
    thickness: float,
    *,
    temperature: float = ROOM_TEMPERATURE,
    relative_permittivity: float = RELATIVE_PERMITTIVITY,
    initial_guess: Sequence[float] = (1e3, 1e-7, 1e-8),
    max_nfev: int = 2000,
) -> CircuitalResult:
    """Fit the space-charge equivalent-circuit model to a dark J-V curve.

    Parameters
    ----------
    curve
        A dark J-V curve with the device ``area`` set. The forward-bias points
        (V > 0, J > 0) are fitted.
    thickness
        Measured active-layer thickness L, in m (e.g. ``105e-9``). mu_eff scales as L^3.
    temperature
        Temperature in K (default 298.15 K).
    relative_permittivity
        Relative permittivity of the active layer (default 3.0; use the value of your material).
    initial_guess
        Starting values (R_sh [ohm], J0 [A/m^2], mu_eff [m^2/(V s)]), all positive.
    max_nfev
        Maximum number of model evaluations of the least-squares solver.

    Returns
    -------
    CircuitalResult

    Raises
    ------
    RuntimeError
        If the least-squares fit does not converge.

    Warns
    -----
    JVHubWarning
        If a parameter is not determined by the data (relative uncertainty >= 100 %).
    """
    area = curve._require_area("the circuital model")
    if not thickness > 0:
        raise ValueError("thickness must be positive (in metres, e.g. 105e-9).")
    if len(initial_guess) != 3 or min(initial_guess) <= 0:
        raise ValueError("initial_guess must be three positive numbers (R_sh, J0, mu_eff).")

    forward = (curve.voltage > 0) & (curve.current_density > 0)
    voltage = curve.voltage[forward]
    density = curve.current_density[forward]
    if voltage.size < 4:
        raise ValueError("The circuital model needs at least 4 points with V > 0 and J > 0.")
    log_density = np.log(density)

    def residuals(log_params):
        rsh, j0, mu_eff = np.exp(log_params)
        model = _solve(voltage, rsh, j0, mu_eff, thickness, area, temperature, relative_permittivity)[1]
        return np.log(np.maximum(model, 1e-300)) - log_density

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        solution = least_squares(residuals, np.log(initial_guess), method="trf", x_scale="jac", max_nfev=max_nfev)
    if not solution.success:
        raise RuntimeError(f"The circuital-model fit did not converge: {solution.message}")

    rsh, j0, mu_eff = np.exp(solution.x)
    dof = max(voltage.size - 3, 1)
    variance = 2 * solution.cost / dof
    covariance = np.linalg.pinv(solution.jac.T @ solution.jac) * variance
    relative = np.sqrt(np.clip(np.diag(covariance), 0, None))  # sigma of ln(p) = relative sigma of p

    for name, value in zip(("R_sh", "J0", "mu_eff"), relative):
        if value >= _UNDETERMINED:
            warnings.warn(
                f"{name} is not determined by the data (relative uncertainty {value:.0%}); "
                "treat its value as an order-of-magnitude estimate at most.",
                JVHubWarning,
                stacklevel=2,
            )

    vsc, fitted = _solve(voltage, rsh, j0, mu_eff, thickness, area, temperature, relative_permittivity)
    return CircuitalResult(
        rsh_ohm=float(rsh),
        rsh_std_ohm=float(rsh * relative[0]),
        j0_A_m2=float(j0),
        j0_std_A_m2=float(j0 * relative[1]),
        mu_eff_cm2_Vs=float(mu_eff * 1e4),
        mu_eff_std_cm2_Vs=float(mu_eff * 1e4 * relative[2]),
        r_squared=_r_squared(density, fitted),
        r_squared_log=_r_squared(log_density, np.log(fitted)),
        thickness_m=float(thickness),
        temperature_K=float(temperature),
        relative_permittivity=float(relative_permittivity),
        n_points=int(voltage.size),
        voltage=voltage,
        current_density=density,
        current_density_fit=fitted,
        space_charge_voltage=vsc,
    )
