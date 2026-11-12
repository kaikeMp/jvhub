# Changelog

## 0.1.0 (unreleased)

First release. The calculations of the JV Analysis Hub v1.0.0 desktop program as a Python library
without graphical interface.

- `JVCurve`, `read_jv` and `read_table`: J-V data in any of A, mA, A/m², mA/cm² with explicit area units.
- `pv_parameters`, `mott_gurney`, `richardson_schottky`: same algorithms and default parameters as
  JV Analysis Hub v1.0.0; results verified against the program on the bundled test data.
- `circuital_model`: numerical implementation rewritten (the equivalent-circuit model itself is unchanged).
  **Results differ from v1.0.0, whose values were not valid:**
  - v1.0.0: the inner `fsolve` call did not converge for the fitted parameters (0 of 251 points on the
    test data), so it returned its initial guess, which was the measured current density. The "model"
    was the data itself, R² was artificially high (0.998), and R_sh, J0 and mu_eff were arbitrary
    (negative, hidden by `abs()`; a last-bit change of the input changed J0 tenfold).
  - Now: one monotonic equation per point in the bulk voltage V_SC, solved by bisection to machine
    precision; positive parameters fitted on ln J (log-log fit, as described in the paper) over the
    forward-bias data; standard uncertainties reported (`*_std_*`); `JVHubWarning` when a parameter is
    not determined by the data (typically J0); `circuital_current_density` evaluates the model.
  - The thickness scan (`optimize_thickness`, ±5 nm) was removed: mu_eff and L enter only as
    mu_eff/L³, so the data cannot determine L. Give the measured thickness.
  - About 40 times faster (0.3 s instead of 12 s for a 250-point curve).
  - On `data-test-CircuitalModel-1.txt` (0.09 cm², 105 nm): R_sh = 66 ± 5 ohm, mu_eff = (5.9 ± 0.4)e-4
    cm²/(V s), J0 undetermined, R² = 0.978 (v1.0.0 reported 2.97e5 ohm, 8.9e-5 cm²/(V s), 5.0e-8 A/m², 0.998).
- Physical parameters (permittivity, temperature, Richardson constant, fitting windows) exposed as
  keyword arguments.
- `jvhub.constants`: one set of SI constants (e, k_B, h exact; eps_0, m_e CODATA 2022) used by all
  models. v1.0.0 mixed e = 1.602e-19 and 1.602176634e-19 C, eps_0 = 8.854e-12 and 8.854187817e-12 F/m,
  and T = 300 K and 298.15 K. Effect on the test data:
  - Mott-Gurney mobility: relative change 2e-10;
  - Richardson-Schottky: A* = 0.5 A_0 with A_0 = 4 pi m_e k_B^2 e / h^3 = 1.20173e6 instead of 1.2e6,
    so phi and phi_eff increase by 0.037 meV;
  - circuital model: default temperature 298.15 K (was 300 K) and SI constants, R_sh and mu_eff change
    by about 5e-4.
- `pv_parameters`: also reports R_s and R_sh normalized by area (`rs_ohm_cm2`, `rsh_ohm_cm2`).
- `pv_parameters`: **new default for R_s** (`rs_method="diode"`): one-diode least-squares fit of the
  measured points above Voc, V = R_s I + n (k_B T/q) ln I_d + const, which separates R_s from the
  diode's differential resistance n k_B T/(q I) and reports `ideality_factor` and `rs_fit_r_squared`.
  The v1.0.0 inverse slope (`rs_method="slope"`) contains that diode resistance and overestimated R_s by
  30-75 % on synthetic one-diode curves; the fit recovers R_s within 0.5 % (noise-free). On the bundled
  curves R_s goes from 20.5 / 15.1 / 14.3 ohm (v1.0.0) to 9.9 / 9.2 / 8.9 ohm (curveData_01/02/03);
  subtracting the diode term from the end-of-curve slope gives 9.8 / 9.2 / 8.9 ohm, and the J-V system's
  own values (13.4 / 11.4 / 10.9 ohm) match that uncorrected end-of-curve slope. Falls back to the slope
  with a `JVHubWarning` if fewer than 5 points lie above Voc. A local slope at Voc was also evaluated and
  rejected (Rs 3-6 times too high; Rsh at V = 0 dominated by noise).
- `pv_parameters`: a single maximum power point (maximum of -J V in 0 <= V <= Voc) is used for
  Vmax, Jmax and the fill factor. v1.0.0 searched Vmax/Jmax over the whole curve and the FF point in
  0 <= V <= Voc separately; both coincide on all bundled curves, so results are unchanged.
- Relative permittivity: one default, eps_r = 3.0 (`jvhub.constants.RELATIVE_PERMITTIVITY`), for all
  dark-curve models. v1.0.0 used 3.9 in Mott-Gurney and Richardson-Schottky (3.9 is the value of SiO2)
  and 3.0 in the circuital model. With the default, Mott-Gurney mobilities are 3.9/3.0 = 1.3 times the
  v1.0.0 values; pass `relative_permittivity=3.9` to reproduce them, or the value of your material.
- `richardson_schottky`: reports the ideality factor n = q/(m k_B T) and the R^2 of the fit, and warns
  (`JVHubWarning`) when n > 2 (`max_ideality`), i.e. when the current in the window is not limited by
  thermionic emission. On the bundled dark curves n is about 21 although R^2 = 0.9998.
- The PCE difference with respect to the J-V system's own file (6.30 % vs 6.64 % for curveData_01) is not
  a calculation error: that file's efficiency is inconsistent with its own Pmax / (area x P_in) = 6.27 %.
- Result objects with units in attribute names, plotting arrays and `to_dict()`.
- `citation()` returns the reference of the paper.

Differences from the desktop program that do not change results for valid input:

- Input data are never modified (the program altered its internal tables in place).
- The dark-curve analyses use the current unit chosen by the user; the program assumed mA in the
  circuital and Richardson-Schottky tabs.
- File reading accepts files with a UTF-8 byte-order mark, fewer than five lines, whitespace-separated
  columns and `.dat`/`.tsv` extensions, and reports empty cells.
- Rs and Rsh are reported in ohm (`rs_ohm`, `rsh_ohm`), the unit actually computed; the program labelled
  them Ω·m².
