"""How to cite jvhub."""

from __future__ import annotations

__all__ = ["citation", "__citation__"]

_BIBTEX = r"""@article{RosivanMaiaPacheco2026,
  title   = {J-V analysis hub: an open-source Python tool for multi-model characterization of organic solar cells},
  author  = {Rosivan Maia Pacheco, Kaike and Benatto, Leandro and Keller Lubacheski, Elis and
             Paulo Souza Ara{\'u}jo, Jo{\~a}o and Manika Koeb, Gabriel and
             Felipe Fagundes das Neves, Matheus and Mancini, Victoria and Menezes, Marcos G. and
             Koehler, Marlus and Stolz Roman, Lucimara},
  journal = {Computer Physics Communications},
  volume  = {324},
  pages   = {110129},
  year    = {2026},
  doi     = {10.1016/j.cpc.2026.110129}
}"""

_TEXT = (
    "K. Rosivan Maia Pacheco, L. Benatto, E. Keller Lubacheski, J. Paulo Souza Araújo, "
    "G. Manika Koeb, M. Felipe Fagundes das Neves, V. Mancini, M. G. Menezes, M. Koehler, "
    "L. Stolz Roman, J-V analysis hub: an open-source Python tool for multi-model "
    "characterization of organic solar cells, Comput. Phys. Commun. 324 (2026) 110129. "
    "https://doi.org/10.1016/j.cpc.2026.110129"
)

#: BibTeX entry of the paper describing the methods implemented in jvhub.
__citation__ = _BIBTEX


def citation(style: str = "bibtex") -> str:
    """Return the reference to cite when jvhub is used in published work.

    Parameters
    ----------
    style
        ``"bibtex"`` (default) or ``"text"``.

    Examples
    --------
    >>> import jvhub
    >>> print(jvhub.citation("text"))  # doctest: +ELLIPSIS
    K. Rosivan Maia Pacheco, ... Comput. Phys. Commun. 324 (2026) 110129. https://doi.org/10.1016/j.cpc.2026.110129
    """
    if style == "bibtex":
        return _BIBTEX
    if style == "text":
        return _TEXT
    raise ValueError("style must be 'bibtex' or 'text'.")
