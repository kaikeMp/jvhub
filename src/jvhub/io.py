"""Reading J-V data files (CSV, TXT, DAT and XLSX)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .curve import CURRENT_UNITS, JVCurve, _normalize_unit, area_to_m2, to_current_density

__all__ = ["read_jv", "read_table"]

_TEXT_SUFFIXES = {".csv", ".txt", ".dat", ".tsv"}
_EXCEL_SUFFIXES = {".xlsx", ".xlsm"}
_ENCODINGS = ("utf-8-sig", "latin1", "cp1252")
_SKIP = {None, "", "skip", "-"}

PathLike = Union[str, Path]


def _read_text(path: Path) -> str:
    for encoding in _ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path.name} as UTF-8, Latin-1 or CP1252.")  # pragma: no cover


def _detect_separator(text: str, test_lines: int = 5) -> str:
    """Pick the separator (comma, semicolon or tab) that occurs most in the first lines.

    Falls back to whitespace when none of them is present.
    """
    lines = [line for line in text.splitlines() if line.strip()][:test_lines]
    counts = {sep: sum(line.count(sep) for line in lines) for sep in (",", ";", "\t")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else r"\s+"


def _is_number(value) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def read_table(path: PathLike, sep: Optional[str] = None) -> pd.DataFrame:
    """Read a numeric table from a text or Excel file.

    The separator is detected automatically (comma, semicolon, tab or whitespace) unless
    ``sep`` is given. A header line is detected and removed when the first row
    contains text. All remaining values must be numeric; the decimal separator
    must be a dot (``.``).

    Returns
    -------
    pandas.DataFrame
        Columns are numbered ``0, 1, 2, ...`` in file order.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        import io

        text = _read_text(path)
        separator = sep if sep is not None else _detect_separator(text)
        df = pd.read_csv(io.StringIO(text), sep=separator, header=None, skip_blank_lines=True)
    elif suffix in _EXCEL_SUFFIXES:
        df = pd.read_excel(path, header=None)
    else:
        raise ValueError(f"Unsupported file format {suffix!r}; use .csv, .txt, .dat, .tsv or .xlsx.")

    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{path.name} contains no data.")
    if not all(_is_number(value) for value in df.iloc[0]):
        df = df.iloc[1:].reset_index(drop=True)
    try:
        df = df.apply(pd.to_numeric)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path.name} contains non-numeric values (the decimal separator must be '.').") from exc
    if df.isna().any().any():
        raise ValueError(f"{path.name} has empty cells; every row must have a value in every column.")
    df.columns = range(df.shape[1])
    return df


def read_jv(
    path: PathLike,
    columns: Sequence[Optional[str]] = ("V", "mA"),
    area: Optional[float] = None,
    area_unit: str = "cm2",
    *,
    sep: Optional[str] = None,
) -> JVCurve:
    """Read a J-V data file into a :class:`~jvhub.JVCurve`.

    Parameters
    ----------
    path
        File to read (``.csv``, ``.txt``, ``.dat``, ``.tsv`` or ``.xlsx``).
    columns
        What each column of the file contains, in file order. Use ``"V"`` for the
        voltage (in volts), one of ``"A"``, ``"mA"``, ``"A/m2"``, ``"mA/cm2"`` for the
        current, and ``None`` (or ``"skip"``) for columns to ignore. The number of
        entries must match the number of columns in the file. If more than one
        current column is given, the first one is used.
    area
        Device area in ``area_unit``. Required when the current is given in A or mA.
    area_unit
        ``"cm2"`` (default) or ``"m2"``.
    sep
        Column separator. Detected automatically when omitted.

    Examples
    --------
    A two-column file with voltage (V) and current (mA) of a 0.09 cm^2 device::

        curve = jvhub.read_jv("cell.txt", columns=["V", "mA"], area=0.09)

    A three-column file whose last column is not needed::

        curve = jvhub.read_jv("cell.csv", columns=["V", "mA/cm2", None], area=0.09)
    """
    df = read_table(path, sep=sep)
    columns = list(columns)
    if df.shape[1] != len(columns):
        raise ValueError(
            f"{Path(path).name} has {df.shape[1]} columns, but {len(columns)} were described in columns={columns}."
        )

    voltage_idx, current_idx, current_unit = None, None, None
    for idx, label in enumerate(columns):
        if label in _SKIP:
            continue
        if str(label).strip().upper() == "V":
            if voltage_idx is not None:
                raise ValueError("More than one voltage ('V') column was given.")
            voltage_idx = idx
            continue
        unit = _normalize_unit(label)
        if unit not in CURRENT_UNITS:
            raise ValueError(f"Column label {label!r} is not 'V', a current unit {CURRENT_UNITS} or None.")
        if current_idx is None:
            current_idx, current_unit = idx, unit
    if voltage_idx is None or current_idx is None:
        raise ValueError("columns must contain one 'V' column and at least one current column.")

    area_m2 = None if area is None else area_to_m2(area, area_unit)
    voltage = df[voltage_idx].to_numpy(dtype=float)
    density = to_current_density(df[current_idx].to_numpy(dtype=float), current_unit, area_m2)
    return JVCurve(np.asarray(voltage), density, area_m2)
