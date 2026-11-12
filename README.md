<p align="center">
  <img src="https://raw.githubusercontent.com/kaikeMp/jv-analysis-hub/v1.0.0/PDFs/logo.png" alt="JV Analysis Hub logo" width="260">
</p>

# jvhub

**Multi-model analysis of current density–voltage (J–V) curves of solar cells, in Python.**

📄 **Paper:** [J-V analysis hub: an open-source Python tool for multi-model characterization of organic solar cells](https://doi.org/10.1016/j.cpc.2026.110129),
*Computer Physics Communications* **324** (2026) 110129 — open access.

`jvhub` is the Python library of [JV Analysis Hub](https://github.com/kaikeMp/jv-analysis-hub), the open-source program described in
*Computer Physics Communications* **324** (2026) 110129 ([doi:10.1016/j.cpc.2026.110129](https://doi.org/10.1016/j.cpc.2026.110129)).
It provides the calculations of the program as plain functions, with no graphical interface, so they can be
used in scripts, notebooks, batch processing or your own applications.

| Analysis | Function | Curve | Main results |
|---|---|---|---|
| Photovoltaic parameters | `pv_parameters` | illuminated | J<sub>sc</sub>, V<sub>oc</sub>, FF, PCE, maximum power point, R<sub>s</sub>, R<sub>sh</sub> |
| Mott–Gurney (SCLC) | `mott_gurney` | dark | charge-carrier mobility μ |
| Equivalent circuit with space-charge resistance | `circuital_model` | dark | R<sub>sh</sub>, J<sub>0</sub>, effective mobility μ<sub>eff</sub>, with uncertainties |
| Richardson–Schottky (thermionic emission) | `richardson_schottky` | dark | injection barrier φ, J<sub>0</sub>, image-force lowered barrier φ<sub>eff</sub> |

## Installation

```bash
pip install jvhub
```

Requires Python ≥ 3.9 with NumPy, SciPy, pandas and openpyxl (installed automatically).

## Quick start

```python
import jvhub

# Illuminated curve: two columns, voltage (V) and current (mA), device area 0.09 cm²
light = jvhub.read_jv("light.txt", columns=["V", "mA"], area=0.09, area_unit="cm2")
pv = jvhub.pv_parameters(light, incident_power=100)  # mW/cm²
print(f"Jsc = {pv.jsc_mA_cm2:.2f} mA/cm², Voc = {pv.voc_V:.3f} V, "
      f"FF = {pv.ff:.3f}, PCE = {pv.pce_percent:.2f} %")

# Dark curve of the same device
dark = jvhub.read_jv("dark.txt", columns=["V", "mA"], area=0.09)

mg = jvhub.mott_gurney(dark, thickness=100e-9)               # thickness in m
print(f"mu = {mg.mobility_cm2_Vs:.2e} cm²/(V s) at V' = {mg.voltage_V:.2f} V")

rs = jvhub.richardson_schottky(dark, thickness=350e-9)
print(f"phi = {rs.phi_eV:.3f} eV, J0 = {rs.j0_A_m2:.2e} A/m²")

cm = jvhub.circuital_model(dark, thickness=105e-9)
print(f"Rsh = {cm.rsh_ohm:.3g} ± {cm.rsh_std_ohm:.2g} ohm, "
      f"mu_eff = {cm.mu_eff_cm2_Vs:.3g} ± {cm.mu_eff_std_cm2_Vs:.2g} cm²/(V s), R² = {cm.r_squared:.4f}")
```

Every result is an immutable object with the scalar results as attributes (the unit is part of the
name, e.g. `jsc_mA_cm2`, `voc_V`, `rs_ohm`), the arrays needed to plot the analysis (e.g.
`mg.voltage`, `mg.m`), and a `to_dict()` method. Collecting many curves in a table is therefore easy:

```python
import pandas as pd
from pathlib import Path

rows = []
for path in sorted(Path("measurements").glob("*.txt")):
    curve = jvhub.read_jv(path, columns=["V", "mA"], area=0.09)
    rows.append({"file": path.name, **jvhub.pv_parameters(curve).to_dict()})
table = pd.DataFrame(rows)
```

### Curves from arrays

```python
curve = jvhub.JVCurve.from_arrays(voltage, current, unit="mA/cm2", area=0.09, area_unit="cm2")
```

Supported current units are `"A"`, `"mA"`, `"A/m2"` and `"mA/cm2"`; the area may be in `"cm2"` or `"m2"`.
Internally a `JVCurve` stores voltage (V), current density (A/m²) and area (m²); `curve.current_mA`,
`curve.current_density_mA_cm2` and `curve.to_dataframe()` give other units.

### Reading files

`read_jv` reads `.csv`, `.txt`, `.dat`, `.tsv` and `.xlsx` files. The column separator (comma,
semicolon, tab or spaces) and a header line are detected automatically; the decimal separator must be
a dot. `columns` describes each column of the file in order — `"V"` for voltage, a current unit, or
`None` to ignore a column:

```python
curve = jvhub.read_jv("cell.csv", columns=["V", "mA/cm2", None], area=0.09)
```

## Models and default parameters

The algorithms are those of JV Analysis Hub v1.0.0, and the test suite checks that `jvhub`
reproduces the numbers of the program for the photovoltaic, Mott–Gurney and Richardson–Schottky
analyses. The equivalent-circuit fit was re-implemented (see below). All physical constants are SI
values (CODATA 2022) defined once in `jvhub.constants` and shared by every model; all model parameters
can be changed through keyword arguments.

| Function | Parameter | Default |
|---|---|---|
| `pv_parameters` | incident power | 100 mW/cm² |
| | interpolation / smoothing | cubic, 1500 points; Savitzky–Golay, window 21, order 3 |
| | R<sub>sh</sub> | inverse slope for V < 0.1 V (Ω and Ω·cm²) |
| | R<sub>s</sub> (`rs_method="diode"`, default) | one-diode fit of the points above V<sub>oc</sub>; also gives the ideality factor n |
| | R<sub>s</sub> (`rs_method="slope"`) | inverse slope for V > 0.9 V<sub>oc</sub>, as in v1.0.0 |
| | maximum power point | maximum of −J·V for 0 ≤ V ≤ V<sub>oc</sub>, used for V<sub>max</sub>, J<sub>max</sub> and FF |
| all dark-curve models | relative permittivity ε<sub>r</sub> | 3.0 (typical of organic semiconductors — pass the value of your material) |
| `mott_gurney` | regime criterion | m(V) = d ln J / d ln V = 2 |
| `richardson_schottky` | fitting window | 0.2 V < V < 0.8 V |
| | temperature, Richardson constant A* | 298.15 K, 0.5 A<sub>0</sub> (A<sub>0</sub> = 4πm<sub>e</sub>k<sub>B</sub>²e/h³ ≈ 1.2017 × 10⁶ A m⁻² K⁻²) |
| | validity check | ideality factor n and R² reported; warning if n > 2 |
| `circuital_model` | temperature | 298.15 K |
| | fitted data / fitted quantity | forward bias (V > 0, J > 0) / ln J |

The theory behind each model is described in the paper below.

### Series resistance

Above V<sub>oc</sub> the current crosses two resistances in series: the series resistance R<sub>s</sub>
(contacts, electrodes) and the differential resistance of the diode itself, n k<sub>B</sub>T/(q I<sub>d</sub>),
which decreases as the current grows. The inverse slope dV/dI measures their sum, so its value depends
on where it is taken and overestimates R<sub>s</sub>. By default `pv_parameters` fits the measured points
above V<sub>oc</sub> with the one-diode relation V = R<sub>s</sub> I + n (k<sub>B</sub>T/q) ln I<sub>d</sub> + const,
which is linear in R<sub>s</sub> and n and separates the two contributions; the ideality factor n and the
R² of this fit are returned as well.

* On synthetic one-diode curves the fit recovers R<sub>s</sub> and n within 0.5 % (a few percent with 1 %
  noise), whereas the slope over V > 0.9 V<sub>oc</sub> overestimates R<sub>s</sub> by 30–75 %.
* On measured cells, subtracting the diode term n k<sub>B</sub>T/(q I) from the slope at the end of the
  curve gives the same R<sub>s</sub> as the fit within about 1 %.
* The fit assumes diode-like behaviour above V<sub>oc</sub> (check `rs_fit_r_squared`; it does not apply to
  S-shaped curves) and needs at least 5 measured points above V<sub>oc</sub>; otherwise the slope method is
  used and a `JVHubWarning` is issued.
* `rs_method="slope"` reproduces JV Analysis Hub v1.0.0.

### Richardson–Schottky validity

The barrier height follows from J<sub>0</sub> only if the current in the fitting window is limited by
thermionic emission, which implies an ideality factor n = q/(m k<sub>B</sub>T) close to 1–2.
`richardson_schottky` reports n and R² and issues a `JVHubWarning` when n > 2 (`max_ideality`): a high
R² alone does not show that the model applies.

### Equivalent-circuit model

For each voltage, the junction (diode in parallel with R<sub>sh</sub>) and the space-charge-limited bulk
must carry the same current. `jvhub` writes this as one equation in the voltage across the bulk,
which has exactly one root, and solves it to machine precision. R<sub>sh</sub>, J<sub>0</sub> and
μ<sub>eff</sub> are then fitted on ln J as positive parameters, and their standard uncertainties are
reported (`rsh_std_ohm`, `j0_std_A_m2`, `mu_eff_std_cm2_Vs`). When the data do not constrain a parameter
(relative uncertainty ≥ 100 %), a `jvhub.JVHubWarning` is issued — on typical dark curves this happens
for J<sub>0</sub>, whose contribution is masked by the shunt and space-charge currents.

μ<sub>eff</sub> and the thickness L enter the model only as μ<sub>eff</sub>/L³, so the fit cannot
determine L: use the measured thickness, and note that μ<sub>eff</sub> scales with L³.

> **Note for users of JV Analysis Hub v1.0.0.** The physical model is unchanged; what was corrected is the
> numerical implementation of the desktop program. In that program the inner equation solver did not
> converge for the fitted parameters, so the "fitted" curve was the measured curve itself, the R² was
> artificially close to 1, and the reported R<sub>sh</sub>, J<sub>0</sub> and μ<sub>eff</sub> were not
> determined by the data (they were also negative before the absolute value was taken). The
> thickness "optimization" of the program could not identify the thickness for the reason above.
> Values of this model obtained with v1.0.0 should be recomputed with `jvhub`.

## Citing

If you use `jvhub` in published work, please cite:

> K. Rosivan Maia Pacheco, L. Benatto, E. Keller Lubacheski, J. Paulo Souza Araújo, G. Manika Koeb,
> M. Felipe Fagundes das Neves, V. Mancini, M. G. Menezes, M. Koehler, L. Stolz Roman,
> *J-V analysis hub: an open-source Python tool for multi-model characterization of organic solar cells*,
> Computer Physics Communications **324** (2026) 110129. https://doi.org/10.1016/j.cpc.2026.110129

The BibTeX entry is available from Python with `print(jvhub.citation())`.

## Graphical program

Users who prefer a graphical interface can use the JV Analysis Hub desktop program (Windows and Linux),
available at <https://github.com/kaikeMp/jv-analysis-hub>.

## License

MIT — see [LICENSE](LICENSE). Developed at the Universidade Federal do Paraná (UFPR), Brazil.
