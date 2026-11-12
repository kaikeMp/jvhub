"""Generate the synthetic J-V curves in this folder (reproducible: fixed random seed).

Run from the repository root:  python tests/data/make_synthetic.py

* illuminated_synthetic_*.txt -- one-diode solar cell with photocurrent, series and shunt
  resistance, written like the J-V system files (header ``[Volt (V)],[Current (mA)]``).
* dark_synthetic*.{csv,txt} -- dark single-carrier-like device: diode + shunt in series
  with a space-charge-limited bulk (forward bias), shunt-dominated reverse bias.

The parameters are of the same order as those of the real devices in ``*_real.*``.
"""

from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).parent
Q = 1.602176634e-19
K = 1.380649e-23
EPS0 = 8.8541878188e-12
T = 298.15
VT = K * T / Q
RNG = np.random.default_rng(20261112)


def one_diode_current(voltage, iph, i0, n, rs, rsh):
    """Terminal current (A, positive in forward bias) of an illuminated one-diode cell."""

    def residual(i, v):
        return i - (i0 * np.expm1((v - i * rs) / (n * VT)) + (v - i * rs) / rsh - iph)

    with np.errstate(over="ignore"):
        return np.array([brentq(residual, -1.0, 1.0, args=(v,), xtol=1e-16) for v in voltage])


def dark_current_density(voltage, rsh, j0, mu, thickness, area, eps_r=3.0):
    """Current density (A/m^2): junction (diode + shunt) in series with a Mott-Gurney bulk."""
    sclc = 9 * eps_r * EPS0 * mu / (8 * thickness**3)
    out = np.empty_like(voltage)
    for k, v in enumerate(voltage):
        if v <= 0:  # reverse bias: shunt current
            out[k] = v / (rsh * area)
            continue

        def excess(vsc):
            return sclc * vsc**2 - (j0 * np.expm1((v - vsc) / VT) + (v - vsc) / (rsh * area))

        vsc = brentq(excess, 0.0, v, xtol=1e-15)
        out[k] = sclc * vsc**2
    return out


def noisy(values, relative, absolute):
    return values * (1 + relative * RNG.standard_normal(values.size)) + absolute * RNG.standard_normal(values.size)


def write_illuminated(name, iph, i0, n, rs, rsh):
    voltage = np.round(np.arange(-1.2, 1.2001, 0.02), 6)
    current_mA = noisy(one_diode_current(voltage, iph, i0, n, rs, rsh), 0.003, 2e-7) * 1e3
    lines = ["[Volt (V)],[Current (mA)]"] + [f"{v:.12f},{i:.12f}" for v, i in zip(voltage, current_mA)]
    (HERE / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dark(name, sep, header, rsh, j0, mu, thickness=105e-9, area=0.09e-4):
    voltage = np.round(np.arange(-1.0, 4.0001, 0.02), 6)
    current_mA = noisy(dark_current_density(voltage, rsh, j0, mu, thickness, area) * area, 0.005, 1e-7) * 1e3
    lines = ([header] if header else []) + [f"{v:.12g}{sep}{i:.12f}" for v, i in zip(voltage, current_mA)]
    (HERE / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    # device area 0.066 cm^2 and 0.09 cm^2 respectively
    write_illuminated("illuminated_synthetic_1.txt", iph=1.21e-3, i0=2e-10, n=1.8, rs=12.0, rsh=6000.0)
    write_illuminated("illuminated_synthetic_2.txt", iph=1.92e-3, i0=5e-13, n=1.6, rs=10.0, rsh=2e4)
    # device area 0.09 cm^2, thickness 105 nm
    write_dark("dark_synthetic.csv", ",", None, rsh=80.0, j0=1e-14, mu=5.5e-8)
    write_dark("dark_synthetic_tab_header.txt", "\t", "volt\tcurrent", rsh=70.0, j0=3e-14, mu=6e-8)
    print("written:", *sorted(p.name for p in HERE.glob("*synthetic*") if p.suffix != ".py"))
