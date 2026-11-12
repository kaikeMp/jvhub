"""jvhub: multi-model analysis of current density-voltage (J-V) curves of solar cells.

jvhub is the Python library of JV Analysis Hub. It provides, without any graphical
interface, the calculations of the program:

* :func:`pv_parameters` -- Jsc, Voc, FF, PCE, maximum power point, Rs and Rsh of
  illuminated curves;
* :func:`mott_gurney` -- charge-carrier mobility from space-charge-limited current;
* :func:`circuital_model` -- equivalent-circuit fit (R_sh, J0, mu_eff) of dark curves;
* :func:`richardson_schottky` -- injection barrier from thermionic emission.

Data enter as a :class:`JVCurve`, built with :func:`read_jv` or
:meth:`JVCurve.from_arrays`.

If you use jvhub in published work, please cite the paper returned by
:func:`citation`.
"""

from ._citation import __citation__, citation
from ._result import JVHubWarning
from .circuit import CircuitalResult, circuital_current_density, circuital_model
from .curve import JVCurve
from .illuminated import PVResult, pv_parameters
from .io import read_jv, read_table
from .sclc import MottGurneyResult, mott_gurney
from .thermionic import RichardsonSchottkyResult, richardson_schottky

__version__ = "0.1.0"

__all__ = [
    "JVCurve",
    "read_jv",
    "read_table",
    "pv_parameters",
    "PVResult",
    "mott_gurney",
    "MottGurneyResult",
    "circuital_model",
    "circuital_current_density",
    "CircuitalResult",
    "JVHubWarning",
    "richardson_schottky",
    "RichardsonSchottkyResult",
    "citation",
    "__citation__",
    "__version__",
]
