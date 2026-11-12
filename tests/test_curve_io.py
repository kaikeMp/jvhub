import numpy as np
import pandas as pd
import pytest
from pytest import approx

import jvhub
from jvhub import JVCurve

V = np.array([-0.5, 0.0, 0.5, 1.0])
I_MA = np.array([-1.0, -0.9, -0.2, 2.0])


# ----------------------------------------------------------------- JVCurve
@pytest.mark.parametrize(
    "unit, values",
    [("mA", I_MA), ("A", I_MA / 1000), ("mA/cm2", I_MA / 0.09), ("A/m2", I_MA / 0.09 * 10), ("mA/cm²", I_MA / 0.09)],
)
def test_units_give_same_curve(unit, values):
    curve = JVCurve.from_arrays(V, values, unit=unit, area=0.09)
    assert curve.current_mA == approx(I_MA, rel=1e-12)
    assert curve.current_density_mA_cm2 == approx(I_MA / 0.09, rel=1e-12)
    assert curve.area == approx(9e-6)
    assert curve.area_cm2 == approx(0.09)


def test_area_in_m2():
    curve = JVCurve.from_arrays(V, I_MA, unit="mA", area=9e-6, area_unit="m2")
    assert curve.current_density == approx(I_MA / 1000 / 9e-6)


def test_density_without_area_is_allowed_but_current_is_not():
    curve = JVCurve.from_arrays(V, I_MA, unit="mA/cm2")
    assert curve.area is None
    with pytest.raises(ValueError, match="area"):
        _ = curve.current_mA
    with pytest.raises(ValueError, match="area"):
        JVCurve.from_arrays(V, I_MA, unit="mA")


def test_curve_is_read_only_and_copies_input():
    voltage = V.copy()
    curve = JVCurve.from_arrays(voltage, I_MA, area=0.09)
    voltage[0] = 99
    assert curve.voltage[0] == -0.5
    with pytest.raises(ValueError):
        curve.voltage[0] = 1.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(voltage=[0, 1], current_density=[1]), "same length"),
        (dict(voltage=[0], current_density=[1]), "at least two"),
        (dict(voltage=[0, np.nan], current_density=[1, 2]), "NaN"),
        (dict(voltage=[0, 1], current_density=[1, 2], area=-1), "positive"),
    ],
)
def test_curve_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        JVCurve(**kwargs)


def test_unknown_unit():
    with pytest.raises(ValueError, match="Unknown unit"):
        JVCurve.from_arrays(V, I_MA, unit="uA", area=0.09)


def test_to_dataframe():
    df = JVCurve.from_arrays(V, I_MA, area=0.09).to_dataframe()
    assert list(df.columns) == ["Voltage (V)", "Current Density (A/m2)", "Current Density (mA/cm2)", "Current (mA)"]


# ------------------------------------------------------------------ files
def _write(tmp_path, name, text, encoding="utf-8"):
    path = tmp_path / name
    path.write_text(text, encoding=encoding)
    return path


@pytest.mark.parametrize(
    "text",
    [
        "-0.5,-1\n0,-0.9\n0.5,-0.2\n1,2\n",
        "V;I\n-0.5;-1\n0;-0.9\n0.5;-0.2\n1;2\n",
        "volt\tcurr\n-0.5\t-1\n0\t-0.9\n0.5\t-0.2\n1\t2\n",
        "  -0.5   -1\n 0  -0.9\n 0.5 -0.2\n 1 2\n",
        "\n-0.5,-1\n\n0,-0.9\n0.5,-0.2\n1,2\n\n",
    ],
    ids=["comma", "semicolon+header", "tab+header", "whitespace", "blank-lines"],
)
def test_read_separators_and_headers(tmp_path, text):
    curve = jvhub.read_jv(_write(tmp_path, "a.txt", text), ["V", "mA"], area=0.09)
    assert curve.voltage == approx(V)
    assert curve.current_mA == approx(I_MA)


def test_read_utf8_bom_without_header(tmp_path):
    path = _write(tmp_path, "bom.csv", "-0.5,-1\n0,-0.9\n0.5,-0.2\n1,2\n", encoding="utf-8-sig")
    assert len(jvhub.read_jv(path, ["V", "mA"], area=0.09)) == 4  # first row is kept


def test_read_latin1_header(tmp_path):
    path = _write(tmp_path, "l.txt", "Tensão,Corrente\n-0.5,-1\n0,-0.9\n", encoding="latin1")
    assert len(jvhub.read_jv(path, ["V", "mA"], area=0.09)) == 2


def test_read_excel(tmp_path):
    path = tmp_path / "cell.xlsx"
    pd.DataFrame({"V": V, "J": I_MA / 0.09, "other": 0}).to_excel(path, index=False)
    curve = jvhub.read_jv(path, ["V", "mA/cm2", None])
    assert curve.current_density_mA_cm2 == approx(I_MA / 0.09)


def test_three_columns_first_current_is_used(tmp_path):
    path = _write(tmp_path, "c.csv", "".join(f"{v},{i},{i / 0.09}\n" for v, i in zip(V, I_MA)))
    curve = jvhub.read_jv(path, ["V", "mA", "mA/cm2"], area=0.09)
    assert curve.current_mA == approx(I_MA)


def test_column_order_can_be_swapped(tmp_path):
    path = _write(tmp_path, "s.csv", "".join(f"{i},{v}\n" for v, i in zip(V, I_MA)))
    assert jvhub.read_jv(path, ["mA", "V"], area=0.09).voltage == approx(V)


@pytest.mark.parametrize(
    "text, columns, message",
    [
        ("1,2\n3,4\n", ["V", "mA", None], "has 2 columns"),
        ("1,2\n3,4\n", ["V", None], "at least one current"),
        ("1,2\n3,4\n", ["V", "V"], "More than one voltage"),
        ("1,2\n3,4\n", ["V", "ohm"], "Unknown unit"),
        ("1,2\n3,x\n", ["V", "mA"], "non-numeric"),
        ("1,2\n3,\n", ["V", "mA"], "empty cells"),
        ("1,2\n3,4\n", ["V", "mA"], "area is required"),
    ],
)
def test_read_errors(tmp_path, text, columns, message):
    with pytest.raises(ValueError, match=message):
        jvhub.read_jv(_write(tmp_path, "e.csv", text), columns)


def test_unsupported_extension(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        jvhub.read_jv(_write(tmp_path, "e.json", "{}"), ["V", "mA"])


def test_decimal_comma_is_rejected(tmp_path):
    path = _write(tmp_path, "d.txt", "0,1;0,5\n0,2;0,7\n")
    with pytest.raises(ValueError, match="decimal separator"):
        jvhub.read_jv(path, ["V", "mA"], area=0.09)
