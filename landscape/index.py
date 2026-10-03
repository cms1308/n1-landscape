"""The index of a theory on the model: the series of the theory (form.itotal_terms) -> the native
expansion engine (singlet.expand_series_native, the irreducible mode) -> field-resolved expansion ->
physical index in the canonical flavor basis, with the string and LaTeX renderers.

The field-resolved expansion keeps one marker exponent per field (source order) and is complete
through t^{t_order} inclusive.  The physical index is the projection flavor_a = sum_f B[a][f] n_f with
B the Hermite normal form of the flavor lattice in the canonical field order; for a theory without a
canonical identity (ambiguous superpotential terms) the source order is used.
"""
from __future__ import annotations

import subprocess
from fractions import Fraction as F
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from . import form
from .convert import index_string, reduce_index
from .model import (CanonicalForm, FieldResolvedExpansion, FieldResolvedTerm, PhysicalTerm, Theory, canonical_flavor_basis,
                    canonical_form, project)
from . import singlet
from .singlet import ProductProjector
from .store import LabelStore


def open_stores(th: Theory, store_dir: str | Path, lie_runner=None, create: bool = True) -> List[LabelStore]:
    """One LabelStore per node from `store_dir`/charstore_<type><rank>.sqlite; nodes of one
    type share the object."""
    opened: Dict[str, LabelStore] = {}
    out = []
    for node in th.nodes:
        g = f"{node.type}{node.rank}"
        if g not in opened:
            kwargs = {"lie_runner": lie_runner} if lie_runner is not None else {}
            opened[g] = LabelStore(Path(store_dir) / f"charstore_{g}.sqlite", g, create=create, **kwargs)
        out.append(opened[g])
    return out


def field_resolved_native(rows):
    """The rows of the expansion engine -> FieldResolvedTerms (integral coefficients asserted; a row
    beyond 128 bits comes with a Python integer); the rows kept in the extension (a FieldResolvedRows
    object, singlet.NATIVE_POST) -> a FieldResolvedExpansion, the sequence of the same terms built on
    demand, whose native rows the post-processing and model.project read."""
    if not isinstance(rows, list):
        return FieldResolvedExpansion(rows)
    out = []
    for num, den, milli, ypow, markers in rows:
        assert den == 1, f"non-integral coefficient {num}/{den} at t^{milli / 1000} y^{ypow} {markers}"
        out.append(FieldResolvedTerm(F(num), milli, ypow, markers))
    return out


class IndexEngine:
    """The expansion engine for one node list: the nodes' groups from their character stores, the
    projector the slot engine of the checks asks (`engine="slot"`)."""

    def __init__(self, stores: Sequence[LabelStore], *, lie_runner=None, match_timeout: Optional[float] = None,
                 engine: str = "irrep"):
        kwargs = {"lie_runner": lie_runner} if lie_runner is not None else {}
        self.projector = ProductProjector(stores, **kwargs)
        self._match_timeout = match_timeout
        self._engine = engine
        self.stop = None

    def _expand(self, series, basis, flavor_only: bool = False):
        return singlet.expand_series_native(series, self.projector, form.EXPANSION_TIMEOUT_S, self._match_timeout,
                                            native_rows=singlet.NATIVE_POST, basis=basis, exact=singlet.NATIVE_EXACT,
                                            coef64=singlet.NATIVE_COEF64, flavor_only=flavor_only, engine=self._engine,
                                            work_bound=form.WORK_BOUND)

    def expansion(self, th: Theory, charges: Sequence, t_order: int, basis: Optional[Sequence[Sequence[int]]] = None, below=None):
        """Field-resolved expansion through t^t_order (a list of FieldResolvedTerms, or a
        FieldResolvedExpansion holding the rows in the extension when singlet.NATIVE_POST is set);
        None where the expansion stops (form.itotal_terms: an order above form.MAX_ORDER, or a remaining
        field whose letter falls at t^0 on the grid), where the engine forms more than form.WORK_BOUND product terms, or past the
        deadline (form.EXPANSION_TIMEOUT_S, a safety net); `self.stop` then holds the cause
        ({"cause": "expansion-order" | "work-bound" | "deadline"}, with "bound" and the count at the cut, "terms", for the work bound) and is
        None after an expansion that is returned.  With `basis` (the flavor
        basis of the record) and singlet.NATIVE_FLAVOR, the engine carries the monomials above t^6
        flavor-refined and the expansion's rows above t^6 are flavor-refined (the native post pass
        consumes them; the field-resolved terms then exist through t^6 only); `below`, the order-6
        expansion of the same theory, charges and basis, is spliced in for the rows through t^6
        (singlet.NATIVE_SPLICE).  A flavor-refined row beyond 128 bits: the expansion field-resolved
        throughout, whose rows beyond 128 bits come with arbitrary-precision coefficients (the Python
        post-processing reads them)."""
        assert len(th.nodes) == self.projector.n_nodes
        self.stop = None
        series = form.itotal_terms(th, charges, t_order)
        if series is None:
            self.stop = {"cause": "expansion-order"}
            return None
        flavor = [list(map(int, row)) for row in basis] if (basis is not None and singlet.NATIVE_POST and singlet.NATIVE_FLAVOR) else None
        try:
            if singlet.NATIVE_SPLICE and flavor is not None and t_order > 6 and getattr(below, "native", None) is not None:
                try:
                    return field_resolved_native(self._expand(series, flavor, flavor_only=True).spliced(below.native))
                except singlet.NativeOverflow:
                    pass                             # the expansion below
            try:
                return field_resolved_native(self._expand(series, flavor))
            except singlet.NativeOverflow:
                if flavor is None:
                    raise
            return field_resolved_native(self._expand(series, None))
        except singlet.WorkBound as e:
            self.stop = {"cause": "work-bound", "bound": form.WORK_BOUND, "terms": e.terms}   # the record keeps the bound
            return None
        except subprocess.TimeoutExpired:
            self.stop = {"cause": "deadline"}
            return None


def canonical_basis(th: Theory, canonical: Optional[CanonicalForm] = None) -> List[List[int]]:
    """Flavor basis (rows = U(1) generators) with columns in the SOURCE field order: the
    Hermite normal form of the lattice in the canonical field order, its columns carried
    back through field_perm; the source-order lattice when no canonical form is given
    (an ambiguous theory).  The single copy is model.canonical_flavor_basis."""
    return canonical_flavor_basis(th, canonical, ambiguous=canonical is None)


def physical_index(th: Theory, terms: Iterable[FieldResolvedTerm], ambiguous: bool = False):
    """(basis in source columns, physical index) in the canonical flavor basis."""
    basis = canonical_basis(th, None if ambiguous else canonical_form(th))
    return basis, project(terms, basis)


def reduced(terms: Iterable[PhysicalTerm], t_order: int, strict: bool = False) -> List[PhysicalTerm]:
    """(1 - t^3 y)(1 - t^3/y)(I - 1) through t^t_order; strict = the legacy E < t_order."""
    red = reduce_index(terms, t_order)
    if strict:
        red = [t for t in red if t.milli < 1000 * t_order]
    return sorted(red, key=lambda p: (p.milli, p.flavor, p.ypow))


def unrefined(terms: Iterable[PhysicalTerm]) -> List[PhysicalTerm]:
    return project([FieldResolvedTerm(t.coeff, t.milli, t.ypow, ()) for t in terms], [])


def render_19a(terms: Iterable[PhysicalTerm]) -> str:
    return index_string(terms)


def _tex_exponent(milli: int) -> str:
    whole, frac = divmod(milli, 1000)
    return str(whole) if not frac else f"{whole}.{frac:03d}".rstrip("0")


def render_latex(terms: Iterable[PhysicalTerm]) -> str:
    """Terms by ascending t exponent, then flavor exponents, then y: `2 t^{3.5} x_{1}^{-2} y`."""
    out = ""
    for t in sorted(terms, key=lambda p: (p.milli, p.flavor, p.ypow)):
        c = t.coeff
        factors = []
        if t.milli:
            factors.append(f"t^{{{_tex_exponent(t.milli)}}}")
        factors += [f"x_{{{a + 1}}}" + (f"^{{{e}}}" if e != 1 else "") for a, e in enumerate(t.flavor) if e]
        if t.ypow:
            factors.append("y" + (f"^{{{t.ypow}}}" if t.ypow != 1 else ""))
        body = " ".join(factors)
        text = body if (abs(c) == 1 and body) else f"{abs(c)} {body}".strip()
        if out:
            out += (" - " if c < 0 else " + ") + text
        else:
            out = ("-" if c < 0 else "") + text
    return out or "0"
