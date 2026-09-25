"""The index of a theory on the model: FORM program (form.py) -> parser and
product projector (singlet.py) -> field-resolved expansion -> physical index in the
canonical flavor basis, with the string and LaTeX renderers.

The field-resolved expansion keeps one marker exponent per field (source order) and is
complete through t^{t_order} inclusive; FORM's truncation acts on the integer t-power
only, so the raw output also holds incomplete terms slightly above t_order, which are
dropped here.  The physical index is the projection flavor_a = sum_f B[a][f] n_f with B
the Hermite normal form of the flavor lattice in the canonical field order; for
a theory without a canonical identity (ambiguous superpotential terms) the source order
is used.
"""
from __future__ import annotations

import subprocess
from fractions import Fraction as F
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import form
from .convert import index_string, reduce_index
from .model import (CanonicalForm, FieldResolvedExpansion, FieldResolvedTerm, PhysicalTerm, Theory, canonical_flavor_basis,
                    canonical_form, project)
from . import singlet
from .singlet import ProductProjector, Term, parse_term, parse_terms, project_terms
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
    """The rows of singlet.expand_native -> FieldResolvedTerms (integral coefficients, as
    field_resolved asserts); the rows kept in the extension (a FieldResolvedRows object,
    singlet.NATIVE_POST) -> a FieldResolvedExpansion, the sequence of the same terms built on
    demand, whose native rows the post-processing and model.project read."""
    if not isinstance(rows, list):
        return FieldResolvedExpansion(rows)
    out = []
    for num, den, milli, ypow, markers in rows:
        assert den == 1, f"non-integral coefficient {num}/{den} at t^{milli / 1000} y^{ypow} {markers}"
        out.append(FieldResolvedTerm(F(num), milli, ypow, markers))
    return out


def field_resolved(records: Iterable[Tuple[Term, int]], n_fields: int, t_order: int) -> List[FieldResolvedTerm]:
    """(parsed term, singlet multiplicity) pairs -> the field-resolved expansion through
    t^t_order: coefficients summed per (t, y, markers), integrality asserted."""
    pos = {form.marker(f): f for f in range(n_fields)}
    acc: Dict[tuple, F] = {}
    for term, mult in records:
        if not mult or term.milli > 1000 * t_order:
            continue
        markers = [0] * n_fields
        for sym, e in term.fug:
            markers[pos[sym]] = e
        key = (term.milli, term.ypow, tuple(markers))
        acc[key] = acc.get(key, F(0)) + term.coeff * mult
    out = []
    for (milli, ypow, markers), c in sorted(acc.items()):
        if c:
            assert c.denominator == 1, f"non-integral coefficient {c} at t^{milli / 1000} y^{ypow} {markers}"
            out.append(FieldResolvedTerm(c, milli, ypow, markers))
    return out


class IndexEngine:
    """FORM runner + projector for one node list."""

    def __init__(self, stores: Sequence[LabelStore], workdir: str | Path, lie_runner=None, core: int = 1,
                 tform_workers: int = 4, match_timeout: Optional[float] = None):
        kwargs = {"lie_runner": lie_runner} if lie_runner is not None else {}
        self.projector = ProductProjector(stores, **kwargs)
        self.runner = form.FormRunner(workdir, tform_workers=tform_workers)
        self._core = core
        self._match_timeout = match_timeout

    def expansion(self, th: Theory, charges: Sequence, t_order: int, basis: Optional[Sequence[Sequence[int]]] = None):
        """Field-resolved expansion through t^t_order (a list of FieldResolvedTerms, or a
        FieldResolvedExpansion holding the rows in the extension when singlet.NATIVE_POST is
        set); None on the "stop" of form.program or a FORM timeout.  With `basis` (the flavor
        basis of the record) and singlet.NATIVE_FLAVOR, the engine carries the monomials above
        t^6 flavor-refined and the expansion's rows above t^6 are flavor-refined (the native post
        pass consumes them; the field-resolved terms then exist through t^6 only)."""
        assert len(th.nodes) == self.projector.n_nodes
        if singlet.NATIVE_FORM:
            series = form.itotal_terms(th, charges, t_order)
            if series is None:                       # the stop of form.program
                return None
            flavor = [list(map(int, row)) for row in basis] if (basis is not None and singlet.NATIVE_POST and singlet.NATIVE_FLAVOR) else None
            try:
                rows = singlet.expand_series_native(series, self.projector, form.FORM_TIMEOUT_S, self._match_timeout,
                                                    native_rows=singlet.NATIVE_POST, basis=flavor, exact=singlet.NATIVE_EXACT,
                                                    coef64=singlet.NATIVE_COEF64)
                return field_resolved_native(rows)
            except singlet.NativeOverflow:
                pass                                 # the FORM path below
            except subprocess.TimeoutExpired:
                return None                          # as a FORM timeout
        source = form.program(th, charges, t_order)
        if source is None:
            return None
        out = self.runner.run(source, t_order)
        if out is None:
            return None
        if singlet.NATIVE_EXPAND:
            try:
                return field_resolved_native(singlet.expand_native(out, len(th.nodes), th.n_fields(), t_order,
                                                                   self.projector, self._match_timeout,
                                                                   native_rows=singlet.NATIVE_POST))
            except ValueError as e:
                if "overflow" not in str(e):
                    raise
                # a coefficient beyond 128 bits in the combined pass (expansion orders above
                # about 20): the Python parser and projector, arbitrary precision
        terms = parse_terms(out, len(th.nodes))
        return field_resolved(project_terms(terms, self.projector, self._core, self._match_timeout),
                              th.n_fields(), t_order)


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
