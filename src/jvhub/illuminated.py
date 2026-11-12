"""Photovoltaic parameters of illuminated J-V curves.

Implements the *PV Illuminated* analysis of JV Analysis Hub: short-circuit current
density, open-circuit voltage, fill factor, power conversion efficiency, maximum
power point and slope estimates of the series and shunt resistances.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from ._result import JVHubWarning, ResultMixin, array_field
from .constants import BOLTZMANN, ELEMENTARY_CHARGE, ROOM_TEMPERATURE
from .curve import JVCurve

_MIN_FORWARD_POINTS = 5

__all__ = ["PVResult", "pv_parameters", "interpolate_curve", "series_resistance_diode_fit"]


@dataclass(frozen=True)
class PVResult(ResultMixin):
    """Result of :func:`pv_parameters`.

    Attributes
    ----------
    jsc_mA_cm2
        Short-circuit current density |J(V=0)|, in mA/cm^2.
    voc_V
        Open-circuit voltage V(J=0), in V.
    ff
        Fill factor, as a fraction between 0 and 1.
    pce_percent
        Power conversion efficiency, in %.
    rs_ohm
        Series resistance, in ohm (see ``rs_method`` of :func:`pv_parameters`).
    rsh_ohm
        Shunt resistance, in ohm, from the inverse slope dV/dI of the current for V < 0.1 V.
    rs_ohm_cm2, rsh_ohm_cm2
        The same resistances normalized by the device area (R x A), in ohm cm^2, the
        usual unit for comparing devices of different sizes.
    ideality_factor
        Diode ideality factor n from the forward-bias fit used for R_s
        (``None`` with ``rs_method="slope"``).
    rs_fit_r_squared
        R^2 of that forward-bias fit (``None`` with ``rs_method="slope"``).
    rs_method
        Method used for R_s.
    vmax_V, jmax_mA_cm2
        Voltage and current density at the maximum power point (J is negative in the
        power-generating quadrant).
    incident_power_mW_cm2
        Incident light power used for the PCE, in mW/cm^2.
    voltage, current_density_mA_cm2
        The interpolated (and smoothed) curve used for the calculation.
    """

    jsc_mA_cm2: float
    voc_V: float
    ff: float
    pce_percent: float
    rs_ohm: float
    rsh_ohm: float
    rs_ohm_cm2: float
    rsh_ohm_cm2: float
    ideality_factor: Optional[float]
    rs_fit_r_squared: Optional[float]
    rs_method: str
    vmax_V: float
    jmax_mA_cm2: float
    incident_power_mW_cm2: float
    voltage: np.ndarray = array_field()
    current_density_mA_cm2: np.ndarray = array_field()


def interpolate_curve(voltage, values, num_points: int = 1500, kind: str = "cubic"):
    """Resample a curve on a uniform voltage grid.

    The data are sorted by voltage, repeated voltages are dropped (the first occurrence
    after sorting is kept) and the curve is interpolated with ``scipy.interpolate.interp1d``
    on ``num_points`` equally spaced voltages between the minimum and maximum voltage.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The new voltage grid and the interpolated values.
    """
    voltage = np.asarray(voltage, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(voltage)
    voltage_unique, first = np.unique(voltage[order], return_index=True)
    values_unique = values[order][first]
    minimum = {"linear": 2, "quadratic": 3, "cubic": 4}.get(kind, 2)
    if voltage_unique.size < minimum:
        raise ValueError(f"{kind} interpolation needs at least {minimum} distinct voltages.")
    grid = np.linspace(voltage_unique.min(), voltage_unique.max(), num_points)
    return grid, interp1d(voltage_unique, values_unique, kind=kind)(grid)


def _smooth(values: np.ndarray, max_window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing with the window rule used by JV Analysis Hub."""
    window = min(max_window, len(values) // 4)
    if window % 2 == 0:
        window += 1
    window = max(window, 5)
    if len(values) < window:
        return values
    return savgol_filter(values, window, polyorder)


def _value_at(x, y, target: float) -> float:
    """Linear interpolation (with extrapolation) of y at x = target."""
    return float(interp1d(x, y, kind="linear", fill_value="extrapolate")(target))


def _inverse_slope(x: np.ndarray, y: np.ndarray, mask: np.ndarray, what: str) -> float:
    if not np.any(mask):
        raise ValueError(f"No data points in the voltage range used for the {what} resistance.")
    return float(1 / stats.linregress(x[mask], y[mask]).slope)


def series_resistance_diode_fit(
    voltage,
    current,
    voc: float,
    isc: float,
    rsh: float,
    temperature: float = ROOM_TEMPERATURE,
) -> Tuple[float, float, float]:
    """Series resistance and ideality factor from the forward bias of an illuminated curve.

    For V > Voc the one-diode model gives

        V = R_s I + n (k_B T / q) ln(I_d) + const,   I_d = I + I_sc - (V - R_s I) / R_sh,

    where I_d is the diode current. With R_sh known, this is linear in (R_s, n, const) and
    is solved by least squares; the small shunt correction of I_d depends on R_s and is
    iterated to convergence. Unlike the inverse slope dV/dI, which equals
    R_s + n k_B T/(q I_d) and therefore overestimates R_s, this separates the series
    resistance from the diode's own differential resistance.

    Parameters
    ----------
    voltage, current
        Measured voltage (V) and current (A, positive in forward bias).
    voc, isc
        Open-circuit voltage (V) and short-circuit current magnitude (A).
    rsh
        Shunt resistance (ohm).
    temperature
        Temperature in K.

    Returns
    -------
    (float, float, float)
        R_s (ohm), ideality factor n, and R^2 of the fit of V.
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    forward = voltage > voc
    v, i = voltage[forward], current[forward]
    if v.size < _MIN_FORWARD_POINTS:
        raise ValueError(f"Fewer than {_MIN_FORWARD_POINTS} points above Voc for the series-resistance fit.")
    thermal = BOLTZMANN * temperature / ELEMENTARY_CHARGE
    rs = 0.0
    for _ in range(100):
        diode_current = i + isc - (v - i * rs) / rsh
        if np.any(diode_current <= 0):
            raise ValueError("Non-positive diode current above Voc; the forward-bias data are not diode-like.")
        design = np.column_stack([i, np.log(diode_current), np.ones_like(i)])
        coefficients = np.linalg.lstsq(design, v, rcond=None)[0]
        converged = abs(coefficients[0] - rs) <= 1e-12 * max(abs(coefficients[0]), 1.0)
        rs = float(coefficients[0])
        if converged:
            break
    predicted = design @ coefficients
    r_squared = float(1 - np.sum((v - predicted) ** 2) / np.sum((v - v.mean()) ** 2))
    return rs, float(coefficients[1] / thermal), r_squared


def pv_parameters(
    curve: JVCurve,
    incident_power: float = 100.0,
    *,
    rs_method: str = "diode",
    temperature: float = ROOM_TEMPERATURE,
    num_points: int = 1500,
    smooth: bool = True,
    smoothing_window: int = 21,
    smoothing_polyorder: int = 3,
    rsh_voltage_limit: float = 0.1,
    rs_voc_fraction: float = 0.9,
) -> PVResult:
    """Extract the photovoltaic parameters of an illuminated J-V curve.

    Algorithm:

    1. The current density (mA/cm^2) and the current (mA) are interpolated with cubic
       splines on ``num_points`` uniformly spaced voltages and, if ``smooth`` is true,
       smoothed with a Savitzky-Golay filter (window ``min(smoothing_window, N/4)``,
       forced odd and at least 5; polynomial order ``smoothing_polyorder``).
    2. Jsc = |J(V=0)| and Voc = V(J=0) by linear interpolation.
    3. The maximum power point (Vmax, Jmax) is the point of maximum generated power
       ``-J*V`` in 0 <= V <= Voc; the same point is used for the fill factor.
    4. FF = Jmax*Vmax/(Jsc*Voc).
    5. PCE = Voc*Jsc*FF/P_in * 100.
    6. Rsh = 1/(dI/dV) from a linear fit of I(V) for V < ``rsh_voltage_limit``.
    7. Rs, with ``rs_method``:

       * ``"diode"`` (default): one-diode fit of the measured points above Voc, which
         separates Rs from the diode's differential resistance n k_B T/(q I) and also
         gives the ideality factor (:func:`series_resistance_diode_fit`). On synthetic
         one-diode curves it recovers Rs within 0.5 % without noise. Check
         ``rs_fit_r_squared``: the fit assumes diode-like behaviour above Voc (it does not
         apply, e.g., to S-shaped curves).
       * ``"slope"`` (JV Analysis Hub v1.0.0): 1/(dI/dV) from a linear fit of the
         interpolated current for V > ``rs_voc_fraction`` * Voc. The slope also contains
         the diode's differential resistance, so it overestimates Rs (by 30-75 % on the
         same synthetic curves).

    With ``rs_method="slope"`` the results are those of JV Analysis Hub v1.0.0 (v1.0.0
    searched Vmax/Jmax over the whole curve and the FF point in 0 <= V <= Voc separately;
    both coincide for normal illuminated curves).

    Parameters
    ----------
    curve
        An illuminated J-V curve with the device ``area`` set (needed for Rs and Rsh).
    incident_power
        Incident light power density in mW/cm^2 (100 mW/cm^2 = 1 sun).
    rs_method
        ``"diode"`` (default) or ``"slope"`` (v1.0.0 method); see step 7. If the curve has
        fewer than 5 points above Voc, ``"diode"`` falls back to ``"slope"`` with a
        :class:`~jvhub.JVHubWarning`.
    temperature
        Temperature in K, used for the ideality factor (default 298.15 K).
    num_points
        Number of points of the interpolated curve.
    smooth, smoothing_window, smoothing_polyorder
        Savitzky-Golay smoothing of the interpolated curve.
    rsh_voltage_limit
        Upper voltage (V) of the region fitted for the shunt resistance.
    rs_voc_fraction
        With ``rs_method="slope"``, the series resistance is fitted for V above this
        fraction of Voc.

    Returns
    -------
    PVResult

    Raises
    ------
    ValueError
        If the curve does not cross the power-generating quadrant (e.g. a dark curve).
    """
    if incident_power <= 0:
        raise ValueError("incident_power must be positive (mW/cm^2).")
    if rs_method not in ("diode", "slope"):
        raise ValueError("rs_method must be 'diode' or 'slope'.")
    curve._require_area("the series and shunt resistances")

    voltage, density = interpolate_curve(curve.voltage, curve.current_density_mA_cm2, num_points)
    _, current_mA = interpolate_curve(curve.voltage, curve.current_mA, num_points)
    if smooth:
        density = _smooth(density, smoothing_window, smoothing_polyorder)
        current_mA = _smooth(current_mA, smoothing_window, smoothing_polyorder)

    jsc = _value_at(voltage, density, 0.0)
    voc = _value_at(density, voltage, 0.0)

    in_quadrant = (voltage >= 0) & (voltage <= voc)
    if not np.any(in_quadrant):
        raise ValueError(
            "No data points between V = 0 and Voc: the curve does not look like an illuminated J-V curve."
        )
    v_q, j_q = voltage[in_quadrant], density[in_quadrant]
    i_mp = int(np.argmax(-j_q * v_q))  # maximum power point, used for Vmax/Jmax and the FF
    vmax, jmax = float(v_q[i_mp]), float(j_q[i_mp])
    ff = abs((vmax * jmax) / (voc * jsc))
    pce = abs(voc * jsc * ff / incident_power) * 100

    current_A = current_mA / 1000
    rsh = _inverse_slope(voltage, current_A, voltage < rsh_voltage_limit, "shunt")

    ideality = rs_r2 = None
    method = rs_method
    if rs_method == "diode":
        isc_A = abs(jsc) * curve.area_cm2 / 1000
        try:
            rs, ideality, rs_r2 = series_resistance_diode_fit(
                curve.voltage, curve.current, voc, isc_A, rsh, temperature
            )
        except ValueError as exc:
            warnings.warn(
                f"Diode fit for R_s not possible ({exc}); using the inverse-slope method, which overestimates R_s.",
                JVHubWarning,
                stacklevel=2,
            )
            method = "slope"
    if method == "slope":
        rs = _inverse_slope(voltage, current_A, voltage > voc * rs_voc_fraction, "series")
    area_cm2 = curve.area_cm2

    return PVResult(
        jsc_mA_cm2=abs(jsc),
        voc_V=voc,
        ff=float(ff),
        pce_percent=float(pce),
        rs_ohm=rs,
        rsh_ohm=rsh,
        rs_ohm_cm2=rs * area_cm2,
        rsh_ohm_cm2=rsh * area_cm2,
        ideality_factor=ideality,
        rs_fit_r_squared=rs_r2,
        rs_method=method,
        vmax_V=vmax,
        jmax_mA_cm2=jmax,
        incident_power_mW_cm2=float(incident_power),
        voltage=voltage,
        current_density_mA_cm2=density,
    )
