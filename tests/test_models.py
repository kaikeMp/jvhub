"""Behaviour of the analyses beyond the v1.0.0 parity checks."""

import numpy as np
import pytest
from pytest import approx

import jvhub
from jvhub import JVCurve
from jvhub.constants import VACUUM_PERMITTIVITY


@pytest.fixture
def light(data_dir):
    return jvhub.read_jv(data_dir / "illuminated_real.txt", ["V", "mA"], area=0.09)


@pytest.fixture
def dark(data_dir):
    return jvhub.read_jv(data_dir / "dark_mott_gurney_real.csv", ["V", "mA"], area=0.09)


# -------------------------------------------------------------- PV parameters
def test_pv_result_object(light):
    result = jvhub.pv_parameters(light)
    assert 0 < result.ff < 1
    assert result.jmax_mA_cm2 < 0 < result.vmax_V < result.voc_V
    assert result.pce_percent == approx(result.jsc_mA_cm2 * result.voc_V * result.ff, rel=1e-12)
    assert len(result.voltage) == len(result.current_density_mA_cm2) == 1500
    data = result.to_dict()
    assert set(data) >= {"jsc_mA_cm2", "voc_V", "ff", "pce_percent", "rs_ohm", "rsh_ohm"}
    assert "voltage" not in data
    with pytest.raises(AttributeError):
        result.ff = 1.0  # results are immutable


def test_pv_does_not_modify_input(light):
    before = light.current_density.copy()
    jvhub.pv_parameters(light)
    assert np.array_equal(light.current_density, before)


def test_pv_needs_area(light):
    curve = JVCurve(light.voltage, light.current_density)
    with pytest.raises(ValueError, match="area"):
        jvhub.pv_parameters(curve)


def test_pv_rejects_dark_curve(dark):
    with pytest.raises(ValueError):
        jvhub.pv_parameters(dark)


def test_pv_without_smoothing_is_close(light):
    smooth = jvhub.pv_parameters(light)
    raw = jvhub.pv_parameters(light, smooth=False)
    assert raw.voc_V == approx(smooth.voc_V, rel=1e-3)
    assert raw.jsc_mA_cm2 == approx(smooth.jsc_mA_cm2, rel=1e-3)


def test_pv_bad_incident_power(light):
    with pytest.raises(ValueError, match="incident_power"):
        jvhub.pv_parameters(light, incident_power=0)


# ---------------------------------------------------------------- Mott-Gurney
def test_mott_gurney_recovers_ideal_sclc():
    thickness, eps_r, mu = 100e-9, 3.9, 1e-8  # m, -, m^2/(V s)
    voltage = np.linspace(0.05, 3, 200)
    density = 9 / 8 * eps_r * VACUUM_PERMITTIVITY * mu * voltage**2 / thickness**3
    result = jvhub.mott_gurney(JVCurve(voltage, density), thickness, relative_permittivity=eps_r)
    assert result.exponent == approx(2, abs=1e-3)
    assert result.mobility_cm2_Vs == approx(mu * 1e4, rel=1e-3)


def test_mott_gurney_mobility_scales_with_thickness_and_permittivity(dark):
    a = jvhub.mott_gurney(dark, 100e-9)
    b = jvhub.mott_gurney(dark, 200e-9, relative_permittivity=6.0)
    assert b.mobility_cm2_Vs == approx(a.mobility_cm2_Vs * 8 / 2, rel=1e-12)


def test_mott_gurney_uses_positive_voltages(dark):
    result = jvhub.mott_gurney(dark, 100e-9)
    assert np.all(result.voltage > 0)
    assert result.m.shape == result.voltage.shape


@pytest.mark.parametrize("thickness", [0, -1e-9])
def test_mott_gurney_bad_thickness(dark, thickness):
    with pytest.raises(ValueError, match="thickness"):
        jvhub.mott_gurney(dark, thickness)


# ------------------------------------------------------- Richardson-Schottky
def test_richardson_schottky_recovers_synthetic_barrier():
    from jvhub.thermionic import BOLTZMANN, ELEMENTARY_CHARGE, RICHARDSON_CONSTANT

    temperature, phi = 298.15, 0.6
    j0 = RICHARDSON_CONSTANT * temperature**2 * np.exp(-ELEMENTARY_CHARGE * phi / (BOLTZMANN * temperature))
    voltage = np.linspace(0, 1, 101)
    density = j0 * (np.exp(voltage / 0.05) - 1)
    result = jvhub.richardson_schottky(JVCurve(voltage, density), thickness=100e-9)
    assert result.phi_eV == approx(phi, rel=1e-6)
    assert result.exponent_per_V == approx(20, rel=1e-6)
    assert result.ideality_factor == approx(ELEMENTARY_CHARGE / (20 * BOLTZMANN * temperature), rel=1e-6)
    assert result.r_squared == approx(1, abs=1e-12)
    assert np.all(result.phi_eff < result.phi_eV)
    assert result.phi_eff_min_eV == approx(result.phi_eff.min())


@pytest.mark.filterwarnings("ignore::jvhub.JVHubWarning")
def test_richardson_schottky_window(dark):
    result = jvhub.richardson_schottky(dark, voltage_window=(0.3, 0.7))
    assert result.voltage.min() > 0.3 and result.voltage.max() < 0.7
    assert result.electric_field.size == 0


def test_richardson_schottky_empty_window(dark):
    with pytest.raises(ValueError, match="voltage_window"):
        jvhub.richardson_schottky(dark, voltage_window=(10, 11))


# ---------------------------------------------------------- circuital model
CIRCUIT = dict(rsh=1e4, j0=1e-6, mu_eff=1e-8, thickness=100e-9, area=9e-6)  # ohm, A/m^2, m^2/Vs, m, m^2


def _synthetic_circuit_curve():
    voltage = np.linspace(0.02, 3, 150)
    density = jvhub.circuital_current_density(voltage, **CIRCUIT)
    return JVCurve(voltage, density, CIRCUIT["area"])


def test_circuital_model_solution_satisfies_both_equations():
    from jvhub.circuit import _EPS0, _K, _Q

    voltage = np.linspace(0.02, 3, 50)
    p = CIRCUIT
    from jvhub.circuit import _solve

    vsc, j = _solve(voltage, p["rsh"], p["j0"], p["mu_eff"], p["thickness"], p["area"], 300.0, 3.0)
    assert np.all((vsc > 0) & (vsc < voltage))
    junction = p["j0"] * np.expm1(_Q * (voltage - vsc) / (_K * 300.0)) + (voltage - vsc) / (p["rsh"] * p["area"])
    bulk_vsc = np.sqrt(8 * j * p["thickness"] ** 3 / (3.0 * _EPS0 * p["mu_eff"])) / 3
    assert j == approx(junction, rel=1e-9)
    assert vsc == approx(bulk_vsc, rel=1e-9)


def test_circuital_model_recovers_synthetic_parameters():
    result = jvhub.circuital_model(_synthetic_circuit_curve(), thickness=CIRCUIT["thickness"])
    assert result.rsh_ohm == approx(CIRCUIT["rsh"], rel=1e-4)
    assert result.j0_A_m2 == approx(CIRCUIT["j0"], rel=1e-4)
    assert result.mu_eff_cm2_Vs == approx(CIRCUIT["mu_eff"] * 1e4, rel=1e-4)
    assert result.r_squared == approx(1, abs=1e-10)


def test_circuital_model_is_independent_of_initial_guess():
    curve = _synthetic_circuit_curve()
    a = jvhub.circuital_model(curve, 100e-9)
    b = jvhub.circuital_model(curve, 100e-9, initial_guess=(1e6, 1e-9, 1e-6))
    for key in ("rsh_ohm", "j0_A_m2", "mu_eff_cm2_Vs"):
        assert getattr(b, key) == approx(getattr(a, key), rel=1e-4)


def test_circuital_mobility_scales_with_thickness_cubed():
    curve = _synthetic_circuit_curve()
    a = jvhub.circuital_model(curve, 100e-9)
    b = jvhub.circuital_model(curve, 120e-9)
    assert b.mu_eff_cm2_Vs == approx(a.mu_eff_cm2_Vs * 1.2**3, rel=1e-4)
    assert b.rsh_ohm == approx(a.rsh_ohm, rel=1e-4)
    assert b.r_squared == approx(a.r_squared, abs=1e-8)


def test_circuital_model_measured_curve(data_dir):
    raw = jvhub.read_table(data_dir / "dark_circuital_real.txt")
    curve = jvhub.read_jv(data_dir / "dark_circuital_real.txt", ["V", "mA"], area=0.09)
    with pytest.warns(jvhub.JVHubWarning, match="J0 is not determined"):
        result = jvhub.circuital_model(curve, thickness=105e-9)
    assert min(result.rsh_ohm, result.j0_A_m2, result.mu_eff_cm2_Vs) > 0
    assert result.r_squared > 0.95
    assert result.n_points == np.sum((raw[0] > 0) & (raw[1] > 0))
    assert result.mu_eff_std_cm2_Vs / result.mu_eff_cm2_Vs < 0.2
    assert result.j0_std_A_m2 / result.j0_A_m2 > 1
    # stable against last-bit changes of the input (v1.0.0 changed J0 tenfold)
    area = 0.09e-4
    other = JVCurve(raw[0].to_numpy(), raw[1].to_numpy() * 1e-3 / area, area)
    with pytest.warns(jvhub.JVHubWarning):
        again = jvhub.circuital_model(other, thickness=105e-9)
    for key in ("rsh_ohm", "j0_A_m2", "mu_eff_cm2_Vs", "r_squared"):
        assert getattr(again, key) == approx(getattr(result, key), rel=1e-6)


def test_circuital_input_validation(dark):
    with pytest.raises(ValueError, match="area"):
        jvhub.circuital_model(JVCurve(dark.voltage, dark.current_density), 100e-9)
    with pytest.raises(ValueError, match="thickness"):
        jvhub.circuital_model(dark, 0)
    with pytest.raises(ValueError, match="initial_guess"):
        jvhub.circuital_model(dark, 100e-9, initial_guess=(1, -1, 1))
    with pytest.raises(ValueError, match="V > 0"):
        jvhub.circuital_current_density([0.0, 1.0], 1e4, 1e-6, 1e-8, 1e-7, 9e-6)


# ------------------------------------------------------------------ citation
def test_citation():
    assert "10.1016/j.cpc.2026.110129" in jvhub.citation()
    assert "Comput. Phys. Commun. 324 (2026) 110129" in jvhub.citation("text")
    assert jvhub.__citation__ == jvhub.citation()
    with pytest.raises(ValueError):
        jvhub.citation("apa")


# ----------------------------------------------------------------- constants
def test_constants_are_si_values():
    from jvhub import constants as c

    assert c.ELEMENTARY_CHARGE == 1.602176634e-19
    assert c.BOLTZMANN == 1.380649e-23
    assert c.VACUUM_PERMITTIVITY == 8.8541878188e-12
    assert c.RICHARDSON_CONSTANT_FREE_ELECTRON == approx(1.20173e6, rel=1e-5)


def test_models_share_constants():
    from jvhub import circuit, constants, sclc, thermionic

    assert sclc.VACUUM_PERMITTIVITY is constants.VACUUM_PERMITTIVITY
    assert thermionic.VACUUM_PERMITTIVITY is constants.VACUUM_PERMITTIVITY
    assert circuit._EPS0 is constants.VACUUM_PERMITTIVITY
    assert circuit._Q is thermionic.ELEMENTARY_CHARGE is constants.ELEMENTARY_CHARGE


@pytest.mark.filterwarnings("ignore::jvhub.JVHubWarning")
def test_default_richardson_constant_shift(dark):
    # changing A* only shifts phi by (kT/q) ln(A*_new / A*_old)
    from jvhub.constants import BOLTZMANN, ELEMENTARY_CHARGE
    from jvhub.thermionic import RICHARDSON_CONSTANT

    new = jvhub.richardson_schottky(dark)
    old = jvhub.richardson_schottky(dark, richardson_constant=0.6e6)
    shift = BOLTZMANN * 298.15 / ELEMENTARY_CHARGE * np.log(RICHARDSON_CONSTANT / 0.6e6)
    assert new.phi_eV - old.phi_eV == approx(shift, rel=1e-9)
    assert new.j0_A_m2 == old.j0_A_m2


def test_area_normalized_resistances(light):
    result = jvhub.pv_parameters(light)
    assert result.rs_ohm_cm2 == approx(result.rs_ohm * 0.09)
    assert result.rsh_ohm_cm2 == approx(result.rsh_ohm * 0.09)


# ------------------------------------------------- series resistance accuracy
def _one_diode_curve(rs, rsh, n, i0, iph, area_cm2=0.09, noise=0.0, seed=0):
    from scipy.optimize import brentq

    from jvhub.constants import BOLTZMANN, ELEMENTARY_CHARGE

    vt = BOLTZMANN * 298.15 / ELEMENTARY_CHARGE
    voltage = np.round(np.arange(-1.2, 1.2001, 0.02), 4)
    with np.errstate(over="ignore"):  # brentq probes |i| up to 1 A, where exp() overflows harmlessly
        current = np.array([
            brentq(lambda i: i - (i0 * np.expm1((v - i * rs) / (n * vt)) + (v - i * rs) / rsh - iph), -1, 1, xtol=1e-16)
            for v in voltage
        ])
    current = current * (1 + noise * np.random.default_rng(seed).standard_normal(voltage.size))
    return JVCurve.from_arrays(voltage, current, unit="A", area=area_cm2)


@pytest.mark.parametrize(
    "rs, rsh, n, i0, iph",
    [(12.0, 6000.0, 1.8, 2e-10, 1.21e-3), (10.0, 2e4, 1.6, 5e-13, 1.92e-3), (40.0, 8000.0, 1.7, 1e-11, 1.5e-3)],
)
def test_series_resistance_diode_fit_is_accurate(rs, rsh, n, i0, iph):
    # noise-free one-diode curve: the fit recovers the true parameters
    curve = _one_diode_curve(rs, rsh, n, i0, iph)
    diode = jvhub.pv_parameters(curve)
    slope = jvhub.pv_parameters(curve, rs_method="slope")
    assert diode.rs_method == "diode"
    assert diode.rs_ohm == approx(rs, rel=0.005)
    assert diode.ideality_factor == approx(n, rel=0.005)
    assert diode.rs_fit_r_squared > 0.99999
    assert diode.rsh_ohm == approx(rsh, rel=0.01)
    assert slope.rs_ohm > 1.3 * rs  # the v1.0.0 slope includes the diode resistance
    assert slope.ideality_factor is None and slope.rs_method == "slope"


@pytest.mark.parametrize(
    "rs, rsh, n, i0, iph",
    [(12.0, 6000.0, 1.8, 2e-10, 1.21e-3), (10.0, 2e4, 1.6, 5e-13, 1.92e-3), (40.0, 8000.0, 1.7, 1e-11, 1.5e-3)],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_series_resistance_with_noise(rs, rsh, n, i0, iph, seed):
    # 1 % multiplicative noise: single curves scatter by a few % (1-sigma about 2 %),
    # still far closer than the slope method (+30 % or more)
    curve = _one_diode_curve(rs, rsh, n, i0, iph, noise=0.01, seed=seed)
    diode = jvhub.pv_parameters(curve)
    slope = jvhub.pv_parameters(curve, rs_method="slope")
    assert diode.rs_ohm == approx(rs, rel=0.07)
    assert diode.ideality_factor == approx(n, rel=0.15)
    assert abs(diode.rs_ohm - rs) < 0.25 * abs(slope.rs_ohm - rs)


def test_series_resistance_falls_back_without_forward_data():
    full = _one_diode_curve(12.0, 6000.0, 1.8, 2e-10, 1.21e-3)
    keep = full.voltage <= 0.74  # Voc is about 0.72 V: only one or two points above Voc
    curve = JVCurve(full.voltage[keep], full.current_density[keep], full.area)
    with pytest.warns(jvhub.JVHubWarning, match="inverse-slope"):
        result = jvhub.pv_parameters(curve)
    assert result.rs_method == "slope" and result.ideality_factor is None


def test_rs_method_validation(light):
    with pytest.raises(ValueError, match="rs_method"):
        jvhub.pv_parameters(light, rs_method="local")


def test_pv_default_is_diode_fit(light):
    result = jvhub.pv_parameters(light)
    assert result.rs_method == "diode"
    assert 1 < result.ideality_factor < 3
    assert result.rs_fit_r_squared > 0.9999


def test_maximum_power_point_is_used_for_fill_factor(light):
    result = jvhub.pv_parameters(light)
    assert 0 <= result.vmax_V <= result.voc_V
    assert result.ff == approx(abs(result.vmax_V * result.jmax_mA_cm2) / (result.voc_V * result.jsc_mA_cm2), rel=1e-12)
    power = -result.voltage * result.current_density_mA_cm2
    inside = (result.voltage >= 0) & (result.voltage <= result.voc_V)
    assert -result.vmax_V * result.jmax_mA_cm2 == approx(power[inside].max(), rel=1e-12)


def test_richardson_schottky_warns_when_not_thermionic(dark):
    with pytest.warns(jvhub.JVHubWarning, match="Ideality factor"):
        result = jvhub.richardson_schottky(dark)
    assert result.ideality_factor > 10
    assert result.r_squared > 0.99  # a good fit is not enough: n shows the model does not apply


def test_permittivity_default_is_shared():
    import inspect

    from jvhub.constants import RELATIVE_PERMITTIVITY

    assert RELATIVE_PERMITTIVITY == 3.0
    for function in (jvhub.mott_gurney, jvhub.richardson_schottky, jvhub.circuital_model):
        assert inspect.signature(function).parameters["relative_permittivity"].default == RELATIVE_PERMITTIVITY
