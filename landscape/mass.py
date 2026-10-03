"""The fields a mass term makes massive, and the superpotential written in the remaining fields.

A mass term is a superpotential monomial of degree two: a*b of two fields or a^2 of one.  Its
fields have R-charges summing to 2, conjugate gauge representations and opposite flavor
charges, so their letters cancel in the flavor-refined index (the boson of one against the
fermion of the other).  The index expansion leaves them out
(`form.itotal_terms`), and the post-processing names operators by the remaining fields: its
F-term rules and its superpotential rule at t^6 read `effective_superpotential`, the
superpotential with the massive fields replaced by the solutions of their F-term equations.
The theory, its superpotential and the a-maximization are not changed.

Which fields: the mass matrix M (the degree-two monomials, generic couplings) has rank r on a
connected component of its graph; the massive set of the component is the lexicographically
first set S of r fields whose principal minor M_SS is nonsingular -- a disjoint pair a*b gives
{a, b}, a^2 gives {a}, a*b + a*c gives {a, b} (c stands for the combination that stays massless).
The F-term equations dW/dS = 0 are then solved for S by iterated substitution from S = 0; the
generic couplings are fixed rationals, so the monomials of the result are those of generic
couplings, and a monomial whose coefficient cancels is left out.  The substitution treats the
fields as commuting variables, blind to a gauge contraction that vanishes (qb·qb of one SU(2)
doublet), so a solution finite in the gauge theory can appear as a series; it is cut after
MAX_ITERATIONS rounds or MAX_MONOMIALS monomials, keeping the lowest degrees -- the monomials the
naming rules meet -- and `effective_complete` is then False.

`DROP_MASSIVE_FIELDS` is True; check scripts set it False for the comparison with the expansion
that keeps every field.
"""
from __future__ import annotations

import functools
import itertools
import random
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

DROP_MASSIVE_FIELDS = True

MAX_ITERATIONS = 4           # substitution rounds before the result is declared truncated
MAX_MONOMIALS = 400          # size of a solution, or of one substituted monomial, before truncation
MAX_COMPONENT = 12           # fields of one component of the mass matrix searched for S

Poly = Dict[Tuple[int, ...], Fraction]


def _key(w: Sequence[Dict[int, int]]) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    return tuple(tuple(sorted((int(f), int(p)) for f, p in m.items())) for m in w)


def _couplings(n_terms: int) -> List[Fraction]:
    rnd = random.Random(20260928)
    return [Fraction(rnd.randint(2, 997), rnd.randint(2, 997)) for _ in range(n_terms)]


def _det(m: List[List[Fraction]]) -> Fraction:
    a = [row[:] for row in m]
    n, det = len(a), Fraction(1)
    for c in range(n):
        p = next((r for r in range(c, n) if a[r][c] != 0), None)
        if p is None:
            return Fraction(0)
        if p != c:
            a[c], a[p] = a[p], a[c]
            det = -det
        det *= a[c][c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            if f:
                for k in range(c, n):
                    a[r][k] -= f * a[c][k]
    return det


def _rank(m: List[List[Fraction]]) -> int:
    a = [row[:] for row in m]
    rank, rows = 0, len(a)
    cols = len(a[0]) if a else 0
    for c in range(cols):
        p = next((r for r in range(rank, rows) if a[r][c] != 0), None)
        if p is None:
            continue
        a[rank], a[p] = a[p], a[rank]
        for r in range(rows):
            if r != rank and a[r][c] != 0:
                f = a[r][c] / a[rank][c]
                for k in range(c, cols):
                    a[r][k] -= f * a[rank][k]
        rank += 1
    return rank


def _mass_matrix(w_key) -> Dict[Tuple[int, int], Fraction]:
    """{(a, b): d^2 W / da db} over the degree-two monomials, a <= b, generic couplings."""
    g = _couplings(len(w_key))
    mm: Dict[Tuple[int, int], Fraction] = {}
    for c, m in zip(g, w_key):
        if sum(p for _, p in m) != 2:
            continue
        if len(m) == 1:
            (a, _), = m
            mm[(a, a)] = mm.get((a, a), 0) + 2 * c
        else:
            (a, _), (b, _) = m
            mm[(a, b)] = mm.get((a, b), 0) + c
    return mm


@functools.lru_cache(maxsize=4096)
def _massive(w_key) -> Tuple[Tuple[int, ...], bool]:
    mm = _mass_matrix(w_key)
    if not mm:
        return (), True
    adj: Dict[int, set] = {}
    for a, b in mm:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen, massive, complete = set(), [], True
    for v in sorted(adj):
        if v in seen:
            continue
        comp, stack = [], [v]
        seen.add(v)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comp.sort()
        if len(comp) > MAX_COMPONENT:
            complete = False
            continue
        entry = lambda a, b: mm.get((min(a, b), max(a, b)), Fraction(0))
        full = [[entry(a, b) for b in comp] for a in comp]
        r = _rank(full)
        for s in itertools.combinations(comp, r):
            if _det([[entry(a, b) for b in s] for a in s]) != 0:
                massive.extend(s)
                break
        else:                                          # cannot happen for a symmetric matrix of rank r
            complete = False
    return tuple(sorted(massive)), complete


def massive_fields(w: Sequence[Dict[int, int]]) -> Tuple[int, ...]:
    """The field indices a mass term of `w` makes massive (empty when the drop is off)."""
    if not DROP_MASSIVE_FIELDS:
        return ()
    return _massive(_key(w))[0]


# --------------------------------------------------------------------------- #
# polynomials over the fields: {exponent tuple: coefficient}
# --------------------------------------------------------------------------- #
def _mul(a: Poly, b: Poly) -> Poly:
    out: Poly = {}
    for ka, ca in a.items():
        for kb, cb in b.items():
            k = tuple(x + y for x, y in zip(ka, kb))
            out[k] = out.get(k, 0) + ca * cb
    return {k: c for k, c in out.items() if c}


def _add_to(out: Poly, p: Poly, s: Fraction = Fraction(1)) -> None:
    for k, c in p.items():
        v = out.get(k, 0) + s * c
        if v:
            out[k] = v
        else:
            out.pop(k, None)


def _substitute(k: Tuple[int, ...], c: Fraction, sol: Dict[int, Poly], n: int) -> Poly:
    term: Poly = {tuple(0 if f in sol else k[f] for f in range(n)): c}
    for f in sol:
        for _ in range(k[f]):
            term = _mul(term, sol[f])
            if not term:
                return term
            if len(term) > MAX_MONOMIALS:          # the lowest-degree monomials first
                term = dict(sorted(term.items(), key=lambda kc: (sum(kc[0]), kc[0]))[:MAX_MONOMIALS])
    return term


def _derivative(poly: Poly, f: int) -> Poly:
    out: Poly = {}
    for k, c in poly.items():
        if k[f]:
            kk = list(k)
            kk[f] -= 1
            _add_to(out, {tuple(kk): c * k[f]})
    return out


@functools.lru_cache(maxsize=4096)
def _effective(w_key) -> Tuple[Tuple[Tuple[Tuple[int, int], ...], ...], bool]:
    massive, complete = _massive(w_key)
    if not massive:
        return w_key, complete
    n = 1 + max(f for m in w_key for f, _ in m)
    g = _couplings(len(w_key))
    terms = [(tuple(dict(m).get(f, 0) for f in range(n)), c) for m, c in zip(w_key, g)]
    W: Poly = {}
    for k, c in terms:
        _add_to(W, {k: c})
    S = list(massive)
    unit = {s: tuple(1 if f == s else 0 for f in range(n)) for s in S}
    grads = {s: _derivative(W, s) for s in S}
    # dW/ds = sum_t M_st t + rest_s: M_SS from the linear monomials of the massive fields
    M = [[grads[s].get(unit[t], Fraction(0)) for t in S] for s in S]
    rest = {s: {k: c for k, c in grads[s].items() if k not in unit.values()} for s in S}
    inv = _inverse(M)
    sol: Dict[int, Poly] = {s: {} for s in S}
    for _ in range(MAX_ITERATIONS):
        ev = {}
        for s in S:
            p: Poly = {}
            for k, c in rest[s].items():
                _add_to(p, _substitute(k, c, sol, n))
            ev[s] = p
        new = {}
        for i, s in enumerate(S):
            p: Poly = {}
            for j, t in enumerate(S):
                if inv[i][j]:
                    _add_to(p, ev[t], -inv[i][j])
            new[s] = p
        if new == sol:
            break
        for s_ in new:
            if len(new[s_]) > MAX_MONOMIALS:
                new[s_] = dict(sorted(new[s_].items(), key=lambda kc: (sum(kc[0]), kc[0]))[:MAX_MONOMIALS])
                complete = False
        sol = new
    else:
        complete = False
    # the effective superpotential in the order of first appearance of its monomials
    order: List[Tuple[int, ...]] = []
    total: Poly = {}
    for k, c in terms:
        for kk, cc in _substitute(k, c, sol, n).items():
            if kk not in total:
                order.append(kk)
                total[kk] = Fraction(0)
            total[kk] += cc
    out = []
    for kk in order:
        if total[kk] != 0 and not any(kk[s] for s in S):
            out.append(tuple((f, e) for f, e in enumerate(kk) if e))
        elif total[kk] != 0:
            complete = False                       # a massive field left after the cap
    return tuple(out), complete


def _inverse(m: List[List[Fraction]]) -> List[List[Fraction]]:
    n = len(m)
    a = [row[:] + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(m)]
    for c in range(n):
        p = next(r for r in range(c, n) if a[r][c] != 0)
        a[c], a[p] = a[p], a[c]
        piv = a[c][c]
        a[c] = [x / piv for x in a[c]]
        for r in range(n):
            if r != c and a[r][c] != 0:
                f = a[r][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [row[n:] for row in a]


def effective_superpotential(w: Sequence[Dict[int, int]]) -> List[Dict[int, int]]:
    """`w` with its massive fields replaced by the solutions of their F-term equations, as
    monomials (exponent dictionaries) in the order of first appearance; `w` itself when no
    field is massive or the drop is off."""
    if not DROP_MASSIVE_FIELDS:
        return [dict(m) for m in w]
    return [dict(m) for m in _effective(_key(w))[0]]


def effective_complete(w: Sequence[Dict[int, int]]) -> bool:
    """False when the massive set or the substitution was truncated by a cap."""
    return _massive(_key(w))[1] and _effective(_key(w))[1]
