"""jvhub must reproduce the numbers of the JV Analysis Hub v1.0.0 program.

``tests/data/reference_v1_0_0.json`` was produced by running the original ``opvtools`` code of
JV Analysis Hub v1.0.0 (GitHub tag v1.0.0) exactly as its graphical interface does, on the files in
``tests/data`` (device area 0.09 cm^2 unless stated; see tests/data/README.md).

Where jvhub changed a default value on purpose (SI/CODATA constants), the test passes the v1.0.0
value explicitly (Richardson constant, eps_r = 3.9), so that the algorithm itself is compared. The circuital model is not compared:
its v1.0.0 results were invalid (see CHANGELOG).
"""

import pytest
from pytest import approx

import jvhub

# Deterministic calculations (interpolation, smoothing, linear regression, Mott-Gurney) agree with
# v1.0.0 to rounding error. Quantities from iterative nonlinear fits (Richardson-Schottky) are only
# converged to the solver tolerance (scipy curve_fit: xtol = ftol = 1.5e-8) and differ in the last
# digits between platforms (BLAS/libm), so they are compared at 1e-6.
RTOL = 1e-9
RTOL_FIT = 1e-6

PV_FILES = {
    "illuminated_real.txt": 0.09,
    "illuminated_synthetic_1.txt": 0.066,
    "illuminated_synthetic_2.txt": 0.09,
}
PV_KEYS = {
    "Jsc_mAcm2": "jsc_mA_cm2",
    "Voc_V": "voc_V",
    "FF": "ff",
    "PCE_percent": "pce_percent",
    "Rs_ohm": "rs_ohm",
    "Rsh_ohm": "rsh_ohm",
    "Vmax_V": "vmax_V",
    "Jmax_mAcm2": "jmax_mA_cm2",
}


def check(result, expected, keys, rtol=RTOL):
    for old, new in keys.items():
        assert getattr(result, new) == approx(expected[old], rel=rtol), new


@pytest.mark.parametrize("name", PV_FILES)
def test_pv_parameters(data_dir, reference, name):
    curve = jvhub.read_jv(data_dir / name, ["V", "mA"], area=PV_FILES[name])
    check(jvhub.pv_parameters(curve, rs_method="slope"), reference["illuminated"][name], PV_KEYS)


@pytest.mark.parametrize("unit", ["A", "mA/cm2"])
def test_pv_parameters_other_units(data_dir, reference, unit):
    raw = jvhub.read_table(data_dir / "illuminated_real.txt")
    current_mA = raw[1].to_numpy()
    values = current_mA / 1000 if unit == "A" else current_mA / 0.09
    curve = jvhub.JVCurve.from_arrays(raw[0], values, unit=unit, area=0.09)
    check(jvhub.pv_parameters(curve, rs_method="slope"), reference["illuminated"][f"illuminated_real.txt[{unit}]"], PV_KEYS)


def test_pv_parameters_incident_power(data_dir, reference):
    curve = jvhub.read_jv(data_dir / "illuminated_real.txt", ["V", "mA"], area=0.09)
    check(jvhub.pv_parameters(curve, incident_power=50, rs_method="slope"), reference["illuminated"]["illuminated_real.txt[Pin=50]"], PV_KEYS)


@pytest.mark.parametrize("name", ["dark_mott_gurney_real.csv", "dark_synthetic_tab_header.txt"])
def test_mott_gurney(data_dir, reference, name):
    curve = jvhub.read_jv(data_dir / name, ["V", "mA"], area=0.09)
    check(
        jvhub.mott_gurney(curve, thickness=100e-9, relative_permittivity=3.9),
        reference["mott_gurney"][name],
        {"Mobility_cm2Vs": "mobility_cm2_Vs", "Voltage_m2": "voltage_V", "CurrentDensity_m2": "current_density_A_m2"},
    )


@pytest.mark.parametrize("name", ["dark_richardson_schottky_real.csv", "dark_synthetic.csv"])
@pytest.mark.parametrize("thickness", [None, 350e-9])
def test_richardson_schottky(data_dir, reference, name, thickness):
    curve = jvhub.read_jv(data_dir / name, ["V", "mA"], area=0.09)
    # v1.0.0 used A* = 0.5 x 1.2e6 and eps_r = 3.9; pass them explicitly. These dark curves give
    # n of about 21 in the 0.2-0.8 V window, so jvhub warns that phi is not reliable.
    with pytest.warns(jvhub.JVHubWarning, match="Ideality factor"):
        result = jvhub.richardson_schottky(
            curve, thickness=thickness, richardson_constant=0.5 * 1.2e6, relative_permittivity=3.9
        )
    key = name if thickness is None else name + "[L=350nm]"
    keys = {"phi_eV": "phi_eV", "J0_A_m2": "j0_A_m2"}
    if thickness is not None:
        keys["phi_eff_min"] = "phi_eff_min_eV"
    else:
        assert result.phi_eff_min_eV is None
    check(result, reference["richardson_schottky"][key], keys, rtol=RTOL_FIT)


# The circuital model is intentionally NOT compared with v1.0.0: in v1.0.0 the inner solver
# never converged and the "fit" returned the measured curve itself (see CHANGELOG, 0.1.0).
# Its correctness is tested against synthetic data in test_models.py.
