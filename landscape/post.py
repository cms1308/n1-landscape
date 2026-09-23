"""Post-processing of the field-resolved expansion: operator extraction
(decoupled, flipped, relevant, marginal), the F-term substitution and the consistency
conditions C1/C2 (the three checks of the original Mathematica post-processing) and
C1'/C3/C4.

A port of the original post-processing from named symbols to positional columns: a term is (t milli-exponent, y power, exponent vector over the
columns); the flavor exponents of a term are basis . exponents, so the old split into
field symbols and g fugacities becomes the split into the column vector and its
projection.  The semantics of the old code are kept, including the two inherited rules
that are conventions rather than derivations:

  * relevant operators: a positive term at (E, flavor) of the F-term-substituted scalar
    part is listed when the scalar part of the flavor-refined reduced index has a nonzero
    coefficient at (E, flavor) or at (E, neutral) (the old NumberQ branch); the positive
    terms this rule leaves out (a boson whose index contribution is cancelled by a fermionic
    term that no F-term substitution removes) are returned as `unlisted`;
  * `Thread[w -> 1]` at t^6: the first superpotential monomial of two or more factors
    whose powers all equal the term's powers is removed once; single-factor monomials are
    removed per factor otherwise.

Operator lists are sets: one entry per exponent vector, with the coefficient of the term
as multiplicity (the old lists are ordered, may repeat an entry and drop the coefficient
except for marginal operators).  A positive term with a negative power of a field is no
entry of any list (the old code lists it) and is returned in
`negative`; dim3, the non-manifest-symmetry flag and the verdicts are index quantities and
are not affected.

`columns` are the fields of a theory; the argument `field_cols` exists for synthetic
inputs whose flavor exponents are given as extra columns instead of through a basis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .model import FieldResolvedTerm

Vec = Tuple[int, ...]
Key = Tuple[int, int, Vec]              # (milli, ypow, exponent vector)
Poly = Dict[Key, F]

_KERNEL = ((0, 0, 1), (3000, 1, -1), (3000, -1, -1), (6000, 0, 1))

# verdicts assigned by the post-processing, in the precedence of the old pipeline
INDEX_VERDICTS = ("consistent", "inconsistent-index", "free-sector-higher-spin-current", "free-sector",
                  "vanishing-index")


class PostProcessingError(RuntimeError):
    """A branch the old code leaves ill-defined (recorded, not mirrored)."""


def verdict_class(old: str) -> Optional[str]:
    """Verdict strings of the original Index stage (the C1'/C3 routing included) -> class."""
    s = " ".join(str(old).split())
    if s in ("consistent", "consistent (duplicated)"):
        return "consistent"
    if s == "inconsistent":
        return "inconsistent-index"
    if s == "inconsistent (free sector: higher-spin current)":
        return "free-sector-higher-spin-current"
    if s == "free sector":
        return "free-sector"
    if s.startswith("inconsistent (vanishing index"):
        return "vanishing-index"
    if s == "inconsistent (negative central charges)":
        return "negative-central-charge"
    if s == "inconsistent (out of Hofman-Maldacena bounds)":
        return "hofman-maldacena-violation"
    if s == "too small Rcharge":
        return "index-not-computed"
    return None


# --------------------------------------------------------------------------- #
# polynomial helpers
# --------------------------------------------------------------------------- #
def _add(poly: Poly, key: Key, c) -> None:
    if not c:
        return
    v = poly.get(key, 0) + c
    if v:
        poly[key] = v
    else:
        poly.pop(key, None)


def _round_half_even(v) -> int:
    v = F(v)
    return v.numerator if v.denominator == 1 else round(v)


def reduced_index(terms: Iterable[FieldResolvedTerm], t_order: Optional[int]) -> Poly:
    """(1 - t^3 y)(1 - t^3/y)(I - 1) on the column grid, E < t_order (None: no truncation),
    coefficients rounded half-even to integers (the old Round[b, 1])."""
    base: Poly = {}
    n = None
    for t in terms:
        n = len(t.markers)
        _add(base, (t.milli, t.ypow, tuple(t.markers)), F(t.coeff))
    if n is None:
        return {}
    _add(base, (0, 0, (0,) * n), F(-1))
    out: Poly = {}
    lim = None if t_order is None else int(round(1000 * t_order))
    for (m, y, v), c in base.items():
        for dm, dy, s in _KERNEL:
            if lim is None or m + dm < lim:
                _add(out, (m + dm, y + dy, v), s * c)
    res: Poly = {}
    for k, c in out.items():
        r = _round_half_even(c)
        if r:
            res[k] = F(r)
    return res


def flavor_of(vec: Vec, basis: Sequence[Sequence[int]]) -> Vec:
    return tuple(sum(b * e for b, e in zip(row, vec)) for row in basis)


def project_flavor(poly: Poly, basis: Sequence[Sequence[int]]) -> Poly:
    """Columns -> flavor exponents (the old `fields -> 1` with the g fugacities kept)."""
    out: Poly = {}
    for (m, y, v), c in poly.items():
        _add(out, (m, y, flavor_of(v, basis)), c)
    return out


def unrefine(poly: Poly) -> Poly:
    out: Poly = {}
    for (m, y, _), c in poly.items():
        _add(out, (m, y, ()), c)
    return out


def _y_max(poly: Poly) -> int:
    if not poly:
        raise PostProcessingError("Exponent[0, y] (the F2 crash input)")
    return max(y for _, y, _ in poly)


def extract_scalar(poly: Poly, p: int) -> Poly:
    """Iterative top-down SU(2)_y character peel (the F3-fixed extractScalar)."""
    qq = dict(poly)
    for kk in range(p, 0, -1):
        top = [((m, v), c) for (m, y, v), c in qq.items() if y == kk]
        for (m, v), c in top:
            for j in range(-kk, kk + 1, 2):
                _add(qq, (m, j, v), -c)
    return qq


def coefficient_t(poly: Poly, milli: int) -> Poly:
    return {k: c for k, c in poly.items() if k[0] == milli}


# --------------------------------------------------------------------------- #
# F-term substitution and the superpotential at t^6
# --------------------------------------------------------------------------- #
Rules = Dict[Tuple[int, int], Dict[int, int]]


def fterm_rules(block: Poly, cols: Iterable[int], w: Sequence[Dict[int, int]]) -> Rules:
    """For every column f of `cols` with a negative power among the terms: rules
    f^j -> (W_f / f)^(-j), j = min power .. -1, W_f the first superpotential monomial
    containing f."""
    rules: Rules = {}
    for f in cols:
        low = min((v[f] for _, _, v in block), default=0)
        if low >= 0:
            continue
        match = next((m for m in w if f in m), None)
        if match is None:
            # a field outside the superpotential has no F-term: its conjugate fermion stays as it is.
            # (The old low-order pass built a rule for every field and was ill-defined here -- pypost
            # raises, the Mathematica original evaluates a broken rule; the old full-order pass
            # already restricts the rules to the fields of W.)
            continue
        for j in range(low, 0):
            rep = {k: p * (-j) for k, p in match.items()}
            rep[f] = rep.get(f, 0) + j
            rules[(f, j)] = rep
    return rules


def apply_rules(poly: Poly, rules: Rules) -> Poly:
    """Every factor of a term that matches a rule is replaced once; results recombine."""
    out: Poly = {}
    for (m, y, v), c in poly.items():
        d = list(v)
        extra: Dict[int, int] = {}
        for f, e in enumerate(v):
            rep = rules.get((f, e)) if e else None
            if rep is not None:
                d[f] = 0
                for k, p in rep.items():
                    extra[k] = extra.get(k, 0) + p
        for k, p in extra.items():
            d[k] += p
        _add(out, (m, y, tuple(d)), c)
    return out


def w_to_one(poly: Poly, w: Sequence[Dict[int, int]]) -> Poly:
    products = [m for m in w if len(m) >= 2]
    singles = [m for m in w if len(m) == 1]
    out: Poly = {}
    for (m, y, v), c in poly.items():
        d = list(v)
        fired = False
        for rule in products:
            if all(d[k] == p for k, p in rule.items()):
                for k in rule:
                    d[k] = 0
                fired = True
                break
        if not fired:
            for rule in singles:
                (k, p), = rule.items()
                if d[k] == p:
                    d[k] = 0
        _add(out, (m, y, tuple(d)), c)
    return out


# --------------------------------------------------------------------------- #
# operators
# --------------------------------------------------------------------------- #
def _operators(block: Poly, field_cols: Sequence[int]) -> Dict[Vec, F]:
    """Positive non-constant terms -> {field exponent vector: coefficient}."""
    out: Dict[Vec, F] = {}
    for (m, y, v), c in block.items():
        if c > 0 and (y or any(v)):
            key = tuple(v[f] for f in field_cols)
            out[key] = out.get(key, 0) + c
    return out


def _split(ops: Dict[Vec, F]) -> Tuple[Dict[Vec, F], Dict[Vec, F]]:
    """(monomials of chiral fields, terms with a negative power).  A negative power is the marker of a
    conjugate-fermion letter that no F-term substitution removed (a field outside the superpotential):
    such a term is not a product of chiral superfields and is no entry of an operator list; it is
    reported in `negative`."""
    chiral = {k: c for k, c in ops.items() if all(e >= 0 for e in k)}
    return chiral, {k: c for k, c in ops.items() if any(e < 0 for e in k)}


def operator_list(ops: Dict[Vec, F], field_cols: Sequence[int]) -> List[dict]:
    """Record form: monomial as [[field, power], ...], sorted; the empty monomial (a term
    without field content) is kept as []."""
    items = []
    for vec, c in ops.items():
        mono = [[field_cols[i], p] for i, p in enumerate(vec) if p]
        assert F(c).denominator == 1
        items.append({"monomial": mono, "multiplicity": int(c)})
    return sorted(items, key=lambda it: (it["monomial"], it["multiplicity"]))


def _merge(dst: Dict[Vec, F], src: Dict[Vec, F]) -> None:
    for k, c in src.items():
        dst[k] = c


def _unlisted(fullscalar: Poly, wvars, w, cols, relevant: Dict[Vec, F]):
    """The positive terms of the F-term-substituted scalar part at 0 < E < 6 that the inherited entry
    rule does not list: their own (E, y, flavor) has net coefficient zero in the scalar part of the
    flavor-refined reduced index and E carries no neutral term.  Taken from every field-resolved bucket,
    also one whose flavor sectors all cancel after the projection; the rules
    are built from all these buckets and agree with the inherited ones on the entry exponents."""
    below = {k: c for k, c in fullscalar.items() if 0 < k[0] < 6000}
    rules = fterm_rules(below, wvars, w)
    positive: Dict[Vec, F] = {}
    for m in sorted({k[0] for k in below}):
        _merge(positive, _operators(apply_rules(coefficient_t(below, m), rules), cols))
    chiral, negative = _split(positive)
    return operator_list({k: c for k, c in chiral.items() if k not in relevant}, cols), negative


# --------------------------------------------------------------------------- #
# C1' / C3 / C4 on the net reduced index
# --------------------------------------------------------------------------- #
def scan(terms: Iterable[FieldResolvedTerm], t_order: int) -> dict:
    """conditions.scan on the model: every bucket with E <= t_order is exact."""
    terms = list(terms)
    refined: Poly = {}
    base: Dict[Tuple[int, int], F] = {}
    for t in terms:
        _add(refined, (t.milli, t.ypow, tuple(t.markers)), F(t.coeff))
        base[(t.milli, t.ypow)] = base.get((t.milli, t.ypow), 0) + F(t.coeff)
    if terms:
        _add(refined, (0, 0, (0,) * len(terms[0].markers)), F(-1))
    base[(0, 0)] = base.get((0, 0), 0) - 1
    net: Dict[Tuple[int, int], F] = {}
    for (m, y), c in base.items():
        if c:
            for dm, dy, s in _KERNEL:
                net[(m + dm, y + dy)] = net.get((m + dm, y + dy), 0) + s * c
    flags = {"c4_vanishing": not refined, "c1prime": [], "c3_free": [], "c3_enhance": [], "noninteger": []}
    if flags["c4_vanishing"]:
        return flags
    max_milli = 1000 * t_order

    def chi(milli: int, j2: int):
        n = net.get((milli, j2), 0) - net.get((milli, j2 + 2), 0)
        return F(n) if n else None

    for j2 in range(1, (max_milli - 2000) // 1000 + 1):
        n = chi(2000 + 1000 * j2, j2)
        if n is None:
            continue
        if n.denominator != 1:
            flags["noninteger"].append((2000 + 1000 * j2, j2, str(n)))
        elif (n if j2 % 2 == 0 else -n) > 0:
            flags["c1prime"].append((j2, int(n)))
    for j2 in range(1, (max_milli - 6000) // 1000 + 1):
        n = chi(6000 + 1000 * j2, j2)
        if n is None:
            continue
        if n.denominator != 1:
            flags["noninteger"].append((6000 + 1000 * j2, j2, str(n)))
            continue
        mult = -n if j2 % 2 == 0 else n
        if mult > 0:
            flags["c3_enhance" if j2 == 1 else "c3_free"].append((j2, int(mult)))
    return flags


# --------------------------------------------------------------------------- #
# the two entry points
# --------------------------------------------------------------------------- #
@dataclass
class PostResult:
    consistency: str = "consistent"          # C1/C2 of the old Index mcode only
    verdict: str = "consistent"              # with C1'/C3/C4 routed as in the original pipeline
    decoupled: Optional[List[dict]] = None
    relevant: Optional[List[dict]] = None
    flipped: Optional[List[dict]] = None
    marginal: Optional[List[dict]] = None
    dim3: Optional[int] = None
    nonmanifest_symmetry: Optional[bool] = None
    susy_enhanced: bool = False
    unlisted: Optional[List[dict]] = None    # positive substituted terms at 0 < E < 6 that the entry rule does not list
    negative: Optional[List[dict]] = None    # positive terms with a negative power, kept out of every list above
    flags: dict = field(default_factory=dict)

    def operators(self) -> dict:
        return {k: v for k, v in (("decoupled", self.decoupled), ("flipped", self.flipped),
                                  ("relevant", self.relevant), ("marginal", self.marginal)) if v is not None}


def _setup(terms, basis, w, field_cols):
    terms = list(terms)
    n = len(terms[0].markers) if terms else 0
    cols = list(range(n)) if field_cols is None else list(field_cols)
    return terms, cols, [dict(m) for m in w]


def decouple(terms: Iterable[FieldResolvedTerm], basis: Sequence[Sequence[int]], w: Sequence[Dict[int, int]],
             t_order: int, field_cols: Optional[Sequence[int]] = None) -> PostResult:
    """The low-order pass: operators at the lowest scalar exponent when it is <= 2
    (R <= 2/3), after the F-term substitution."""
    terms, cols, w = _setup(terms, basis, w, field_cols)
    res = PostResult(decoupled=[])
    if not reduced_index(terms, None):
        return res
    reduced = reduced_index(terms, t_order)
    index2 = project_flavor(reduced, basis)
    if not index2:
        return res
    scalars = extract_scalar(reduced, _y_max(reduced))
    unrefined = extract_scalar(index2, _y_max(index2))
    if not unrefined:
        res.relevant, res.flipped = [], []
        return res
    exponents = sorted(m for m, _, _ in unrefine(unrefined) if 0 < m < 6000)
    if not exponents:
        raise PostProcessingError("decouple: no scalar exponent in (0, 6)")
    letters = {k: c for e in set(exponents) for k, c in coefficient_t(scalars, e).items()}
    rules = fterm_rules(letters, cols, w)
    if exponents[0] <= 2000:
        block = apply_rules(coefficient_t(scalars, exponents[0]), rules)
        chiral, negative = _split(_operators(block, cols))
        res.decoupled = operator_list(chiral, cols)
        res.negative = operator_list(negative, cols)
    return res


def index(terms: Iterable[FieldResolvedTerm], basis: Sequence[Sequence[int]], w: Sequence[Dict[int, int]],
          t_order: int, field_cols: Optional[Sequence[int]] = None) -> PostResult:
    """The full-order pass: C4, C1/C2, operators, C1'/C3."""
    terms, cols, w = _setup(terms, basis, w, field_cols)
    res = PostResult()
    res.flags = scan(terms, t_order)
    if res.flags["c4_vanishing"]:
        res.consistency = res.verdict = "vanishing-index"
        res.decoupled = []
        return res
    wvars = sorted({k for m in w for k in m})
    reduced2 = reduced_index(terms, t_order)
    index2 = project_flavor(reduced2, basis)
    power, fullpower = _y_max(index2), _y_max(reduced2)
    fullscalar = extract_scalar(reduced2, fullpower)
    indexscalar = extract_scalar(index2, power)
    indexspinor = {k: c for k, c in index2.items() if k[1] >= 1}
    # one entry per term of the scalar part; the old Plus order puts y-free terms first,
    # then ascending t, so entries[0] carries the lowest scalar exponent
    entries = sorted({(m, y, fl) for (m, y, fl) in indexscalar if 0 < m < 6000},
                     key=lambda e: ((0, 0) if e[1] == 0 else (1, e[1]), e[0], e[2]))
    has_six = any(m == 6000 for m, _, _ in fullscalar)

    # 1. C1/C2
    if any(m < 6000 and c < 0 for (m, _, _), c in indexscalar.items()):
        res.consistency = "inconsistent-index"
    elif indexspinor:
        floor_hit = False
        for k in range(power, 0, -1):
            ms = [m for (m, y, _) in indexspinor if y == k]
            if ms and min(ms) < 2000 + 1000 * k:
                floor_hit = True
                break
        if floor_hit:
            res.consistency = "inconsistent-index"
        else:
            for (m, y, _), c in indexspinor.items():
                k = abs(y)
                if m < 6000 + 1000 * k and (1 if c > 0 else -1) == (-1) ** (1 + k):
                    res.consistency = "inconsistent-index"
                    break

    # 2. operators
    if not entries:
        res.decoupled, res.relevant, res.flipped = [], [], []
        res.unlisted, negative = _unlisted(fullscalar, wvars, w, cols, {})
        res.negative = operator_list(negative, cols)
    else:
        letters = {k: c for m, _, _ in entries for k, c in coefficient_t(fullscalar, m).items()}
        rules = fterm_rules(letters, wvars, w)
        if entries[0][0] <= 2000:
            block = apply_rules(coefficient_t(fullscalar, entries[0][0]), rules)
            chiral, negative = _split(_operators(block, cols))
            res.decoupled = operator_list(chiral, cols)
            res.negative = operator_list(negative, cols)
        elif res.consistency != "consistent":
            res.decoupled = []
        else:
            neutral = (0,) * len(basis)
            relevant: Dict[Vec, F] = {}
            flipped: Dict[Vec, F] = {}
            for m, y, fl in entries:
                block = apply_rules(coefficient_t(fullscalar, m), rules)
                if y or fl != neutral:
                    block = {k: c for k, c in block.items() if k[1] == y and flavor_of(k[2], basis) == fl}
                ops = _operators(block, cols)
                _merge(relevant, ops)
                if m < 4000:
                    _merge(flipped, ops)
            marginal: Dict[Vec, F] = {}
            dim3 = 0
            coef_total = F(0)
            if has_six:
                six = coefficient_t(fullscalar, 6000)
                coef = w_to_one(apply_rules(six, fterm_rules(six, wvars, w)), w)
                by_fields: Dict[Vec, F] = {}
                for (_, _, v), c in coef.items():
                    key = tuple(v[f] for f in cols)
                    by_fields[key] = by_fields.get(key, 0) + c
                by_fields = {k: c for k, c in by_fields.items() if c}
                coef_total = by_fields.get((0,) * len(cols), F(0))
                if any(any(k) for k in by_fields):
                    for (_, _, v), c in coef.items():
                        key = tuple(v[f] for f in cols)
                        if any(key) and c > 0:
                            marginal[key] = marginal.get(key, 0) + c
                dim3 = int(unrefine(indexscalar).get((6000, 0, ()), 0))
            res.unlisted, negative = _unlisted(fullscalar, wvars, w, cols, relevant)
            marginal, neg6 = _split(marginal)
            _merge(negative, neg6)
            res.decoupled = []
            res.relevant = operator_list(_split(relevant)[0], cols)
            res.flipped = operator_list(_split(flipped)[0], cols)
            res.marginal = operator_list(marginal, cols)
            res.negative = operator_list(negative, cols)
            res.dim3 = dim3
            res.nonmanifest_symmetry = bool(len(basis) + coef_total < 0)

    # 3. C1'/C3 routing (C3 with j >= 1 takes precedence over C1')
    res.verdict = res.consistency
    if res.consistency == "consistent":
        if res.flags["c3_free"]:
            res.verdict = "free-sector-higher-spin-current"
        elif res.flags["c1prime"]:
            res.verdict = "free-sector"
        elif res.flags["c3_enhance"]:
            res.susy_enhanced = True
    return res
