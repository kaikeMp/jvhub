"""Physical constants used by jvhub, in SI units.

The elementary charge, the Boltzmann constant and the Planck constant are exact in the SI
since its 2019 revision; the other values are the CODATA 2022 recommended values. All
analyses take their constants from this module, so the models are mutually consistent.
"""

from math import pi

__all__ = [
    "ELEMENTARY_CHARGE",
    "BOLTZMANN",
    "PLANCK",
    "ELECTRON_MASS",
    "VACUUM_PERMITTIVITY",
    "RICHARDSON_CONSTANT_FREE_ELECTRON",
    "ROOM_TEMPERATURE",
    "RELATIVE_PERMITTIVITY",
]

#: Elementary charge e, in C (exact).
ELEMENTARY_CHARGE = 1.602176634e-19
#: Boltzmann constant k_B, in J/K (exact).
BOLTZMANN = 1.380649e-23
#: Planck constant h, in J s (exact).
PLANCK = 6.62607015e-34
#: Electron mass m_e, in kg (CODATA 2022).
ELECTRON_MASS = 9.1093837139e-31
#: Vacuum permittivity eps_0, in F/m (CODATA 2022).
VACUUM_PERMITTIVITY = 8.8541878188e-12
#: Richardson constant of free electrons A_0 = 4 pi m_e k_B^2 e / h^3, in A m^-2 K^-2 (about 1.2017e6).
RICHARDSON_CONSTANT_FREE_ELECTRON = 4 * pi * ELECTRON_MASS * BOLTZMANN**2 * ELEMENTARY_CHARGE / PLANCK**3
#: Default temperature of the dark-curve analyses, in K (25 degC).
ROOM_TEMPERATURE = 298.15
#: Default relative permittivity of the active layer (typical of organic semiconductors).
#: It is a material property: pass the value of your material when it is known.
RELATIVE_PERMITTIVITY = 3.0
