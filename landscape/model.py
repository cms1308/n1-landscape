"""Structured theories and records.

A theory is a list of nodes (type, rank), a list of fields — each a tuple of Dynkin
labels, one per node, the zero label for a trivial node — and superpotential terms as
exponent vectors over the fields.  Flip fields are trivial-rep fields whose provenance
records the operator they flip.  Names are display data only.

Provided here:
  * per-field group data (dimensions, Dynkin indices, spectator products);
  * the flavor lattice: the saturated integer kernel of the anomaly and neutrality
    constraints, in Hermite normal form (rows = U(1) generators, columns = fields in
    the order given);
  * the singlet multiplicity of a superpotential monomial (LiE, product group) and the
    ambiguity flag (multiplicity > 1: the exponent vector does not name the contraction);
  * the canonical form under permutations of identical nodes, permutations of fields
    and conjugation of a node, with the maps from source positions, and its hash;
  * the two index interfaces — the field-resolved expansion (per-field markers) and the
    physical index (t on the 1/1000 grid, y, flavor exponents, coefficient) — with the
    projection between them, truncation, the flavor basis of a record (single copy) and
    the identity of a fixed point (the equivalence key of the enumeration);
  * JSON serialization of theories and records.
"""
from __future__ import annotations

import decimal
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import lie

Label = Tuple[int, ...]
Term = Dict[int, int]            # field index -> power (> 0 for superpotentials)

SCHEMA_VERSION = "29.0"


# --------------------------------------------------------------------------- #
# theories
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Node:
    type: str
    rank: int

    def zero(self) -> Label:
        return (0,) * self.rank

    def key(self):
        return (self.type, self.rank)


@dataclass
class Theory:
    nodes: List[Node]
    fields: List[Tuple[Label, ...]]          # fields[f][i] = label of field f at node i
    terms: List[Term] = field(default_factory=list)
    flips: Dict[int, Term] = field(default_factory=dict)   # flip field -> flipped operator
    names: Optional[List[str]] = None       # display names, source order

    # ---- structure -------------------------------------------------------- #
    def n_fields(self) -> int:
        return len(self.fields)

    def is_trivial(self, f: int) -> bool:
        return all(not any(lab) for lab in self.fields[f])

    def dims(self, f: int) -> List[int]:
        return [dim(node, lab) for node, lab in zip(self.nodes, self.fields[f])]

    def mu(self, f: int, i: int) -> F:
        return index(self.nodes[i], self.fields[f][i])

    def spectator(self, f: int, i: int) -> int:
        d = self.dims(f)
        return math.prod(d[:i] + d[i + 1:])

    def total_dim(self, f: int) -> int:
        return math.prod(self.dims(f))

    def display_names(self) -> List[str]:
        return list(self.names) if self.names else [f"f{k + 1}" for k in range(self.n_fields())]

    # ---- serialization --------------------------------------------------- #
    def to_json(self) -> dict:
        return {
            "nodes": [[n.type, n.rank] for n in self.nodes],
            "fields": [[list(lab) for lab in fl] for fl in self.fields],
            "terms": [sorted([[f, p] for f, p in t.items()]) for t in self.terms],
            "flips": {str(f): sorted([[g, p] for g, p in op.items()]) for f, op in self.flips.items()},
            "names": self.display_names(),
        }

    @staticmethod
    def from_json(d: dict) -> "Theory":
        return Theory(
            nodes=[Node(t, int(r)) for t, r in d["nodes"]],
            fields=[tuple(tuple(int(x) for x in lab) for lab in fl) for fl in d["fields"]],
            terms=[{int(f): int(p) for f, p in t} for t in d.get("terms", [])],
            flips={int(f): {int(g): int(p) for g, p in op} for f, op in d.get("flips", {}).items()},
            names=list(d["names"]) if d.get("names") else None,
        )


def dim(node: Node, lab: Label) -> int:
    return lie.dimension(node.type, node.rank, lab)


def index(node: Node, lab: Label) -> F:
    return lie.dynkin_index(node.type, node.rank, lab)


# --------------------------------------------------------------------------- #
# integer linear algebra
# --------------------------------------------------------------------------- #
def integer_kernel(rows: Sequence[Sequence[int]], n: int) -> List[List[int]]:
    """Z-basis of {x in Z^n : M x = 0} for an integer matrix M (rows), by unimodular
    column operations: M U = [H | 0] with U unimodular; the columns of U over the zero
    block span the (saturated) kernel."""
    M = [list(map(int, r)) for r in rows]
    m = len(M)
    U = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    def col_op(j, k, a):        # column k += a * column j   (M and U)
        for r in M:
            r[k] += a * r[j]
        for r in U:
            r[k] += a * r[j]

    def col_swap(j, k):
        for r in M:
            r[j], r[k] = r[k], r[j]
        for r in U:
            r[j], r[k] = r[k], r[j]

    pivot_col = 0
    for i in range(m):
        if pivot_col >= n:
            break
        # euclid on row i over columns pivot_col..n-1
        while True:
            nz = [j for j in range(pivot_col, n) if M[i][j] != 0]
            if not nz:
                break
            jmin = min(nz, key=lambda j: abs(M[i][j]))
            if jmin != pivot_col:
                col_swap(jmin, pivot_col)
            done = True
            for j in range(pivot_col + 1, n):
                if M[i][j] != 0:
                    q = M[i][j] // M[i][pivot_col]
                    col_op(pivot_col, j, -q)
                    if M[i][j] != 0:
                        done = False
            if done:
                break
        if M[i][pivot_col] != 0:
            pivot_col += 1
    kernel = [[U[r][c] for r in range(n)] for c in range(pivot_col, n)]
    return kernel


def hnf_rows(B: Sequence[Sequence[int]]) -> List[List[int]]:
    """Row-style Hermite normal form: pivots positive, leftmost, entries above a pivot
    reduced into [0, pivot); zero rows dropped."""
    A = [list(map(int, r)) for r in B]
    if not A:
        return []
    n = len(A[0])
    r = 0
    for c in range(n):
        if r >= len(A):
            break
        while True:
            nz = [i for i in range(r, len(A)) if A[i][c] != 0]
            if not nz:
                break
            imin = min(nz, key=lambda i: abs(A[i][c]))
            A[r], A[imin] = A[imin], A[r]
            done = True
            for i in range(r + 1, len(A)):
                if A[i][c] != 0:
                    q = A[i][c] // A[r][c]
                    A[i] = [a - q * b for a, b in zip(A[i], A[r])]
                    if A[i][c] != 0:
                        done = False
            if done:
                break
        if r < len(A) and A[r][c] != 0:
            if A[r][c] < 0:
                A[r] = [-x for x in A[r]]
            for i in range(r):
                q = A[i][c] // A[r][c]
                if q:
                    A[i] = [a - q * b for a, b in zip(A[i], A[r])]
            r += 1
    return [row for row in A if any(row)]


def _solve_square(A: List[List[F]], rhs: List[List[F]]) -> List[List[F]]:
    """Solve A X = RHS for square invertible A by Gauss–Jordan over Fractions."""
    n = len(A)
    M = [list(A[i]) + list(rhs[i]) for i in range(n)]
    w = len(M[0])
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c] != 0), None)
        assert piv is not None, "singular matrix"
        M[c], M[piv] = M[piv], M[c]
        inv = 1 / M[c][c]
        M[c] = [x * inv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]


def _matmul(A, B):
    return [[sum(F(A[i][k]) * F(B[k][j]) for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def _transpose(A):
    return [list(col) for col in zip(*A)] if A else []


def solve_integer_transformation(B_old: Sequence[Sequence[int]], B_new: Sequence[Sequence[int]]) -> List[List[int]]:
    """T with B_old = T B_new (rows = generators): T = B_old B_new^T (B_new B_new^T)^{-1},
    exact; integrality asserted (the old lattice must lie in the new one)."""
    if not B_new:
        assert not B_old or all(not any(r) for r in B_old), "old basis nonzero but new lattice trivial"
        return [[] for _ in B_old]
    Bt = _transpose(B_new)
    G = _matmul(B_new, Bt)                      # k_new x k_new, invertible
    Ginv = _solve_square([[F(x) for x in r] for r in G], [[F(1 if i == j else 0) for j in range(len(G))] for i in range(len(G))])
    T = _matmul(_matmul(B_old, Bt), Ginv)
    for row in T:
        for v in row:
            assert v.denominator == 1, f"non-integral transformation {row}"
    return [[int(v) for v in row] for row in T]


def left_inverse(T: Sequence[Sequence[int]]) -> List[List[F]]:
    """L with L T = 1 (T of full column rank): L = (T^T T)^{-1} T^T."""
    if not T or not T[0]:
        return []
    Tt = _transpose(T)
    G = _matmul(Tt, T)
    Ginv = _solve_square([[F(x) for x in r] for r in G], [[F(1 if i == j else 0) for j in range(len(G))] for i in range(len(G))])
    return _matmul(Ginv, Tt)


# --------------------------------------------------------------------------- #
# flavor lattice
# --------------------------------------------------------------------------- #
def constraint_rows(th: Theory) -> List[List[int]]:
    """Integer rows: one anomaly constraint per node (mu_i(R_f) * spectator dims), one
    neutrality constraint per superpotential term."""
    rows: List[List[int]] = []
    n = th.n_fields()
    for i in range(len(th.nodes)):
        row = [th.mu(f, i) * th.spectator(f, i) for f in range(n)]
        if any(row):
            lcm = 1
            for x in row:
                lcm = lcm * x.denominator // math.gcd(lcm, x.denominator)
            rows.append([int(x * lcm) for x in row])
    for t in th.terms:
        rows.append([t.get(f, 0) for f in range(n)])
    return rows


def flavor_lattice(th: Theory) -> List[List[int]]:
    """HNF basis (rows) of the saturated integer kernel of the constraints, columns in
    the theory's field order."""
    rows = constraint_rows(th)
    n = th.n_fields()
    if not rows:
        return hnf_rows([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    return hnf_rows(integer_kernel(rows, n))


def flavor_rank(th: Theory) -> int:
    return len(flavor_lattice(th))


# --------------------------------------------------------------------------- #
# singlet multiplicity of a superpotential monomial (LiE, product group)
# --------------------------------------------------------------------------- #
_SINGLET_CACHE: Dict[tuple, int] = {}


def singlet_multiplicity(th: Theory, term: Term) -> int:
    """Number of gauge singlets in the tensor product over the fields of the symmetric
    powers Sym^{p}(R_f) of the product-group representations of the term."""
    involved = [(i, node) for i, node in enumerate(th.nodes)
                if any(any(th.fields[f][i]) for f in term)]
    if not involved:
        return 1
    key = (tuple((node.type, node.rank) for _, node in involved),
           tuple(sorted((tuple(th.fields[f][i] for i, _ in involved), p) for f, p in term.items()
                        if any(any(th.fields[f][i]) for i, _ in involved))))
    if key in _SINGLET_CACHE:
        return _SINGLET_CACHE[key]
    group = lie.lie_group([node.key() for _, node in involved])
    rank = sum(node.rank for _, node in involved)
    factors = []
    for labs, p in key[1]:
        lab = lie.lie_label(labs)
        factors.append(f"sym_tensor({p},{lab},{group})" if p > 1 else f"1X{lab}")
    acc = factors[0]
    for fct in factors[1:]:
        acc = f"tensor({acc},{fct},{group})"
    (val,) = lie.lie_values([acc], group)
    poly = lie.parse_poly(val)
    mult = poly.get(tuple(0 for _ in range(rank)), 0)
    _SINGLET_CACHE[key] = mult
    return mult


def ambiguous_terms(th: Theory) -> List[int]:
    return [k for k, t in enumerate(th.terms) if singlet_multiplicity(th, t) != 1]


# --------------------------------------------------------------------------- #
# canonical form
# --------------------------------------------------------------------------- #
def _relabel(th: Theory, node_perm: Sequence[int], conj: Sequence[bool]) -> Theory:
    """Nodes reordered (new position k holds old node node_perm[k]) and conjugated."""
    nodes = [th.nodes[j] for j in node_perm]
    fields = []
    for fl in th.fields:
        labs = []
        for k, j in enumerate(node_perm):
            lab = fl[j]
            if conj[k]:
                lab = lie.conjugate(nodes[k].type, nodes[k].rank, lab)
            labs.append(tuple(lab))
        fields.append(tuple(labs))
    return Theory(nodes, fields, [dict(t) for t in th.terms], {f: dict(o) for f, o in th.flips.items()}, th.names)


def _encode(th: Theory, order: Sequence[int]) -> tuple:
    """Encoding of the theory with fields renumbered by order (new index k <- old order[k])."""
    pos = {old: new for new, old in enumerate(order)}
    fields = tuple(th.fields[old] for old in order)
    terms = tuple(sorted(tuple(sorted((pos[f], p) for f, p in t.items())) for t in th.terms))
    return (tuple(n.key() for n in th.nodes), fields, terms)


def _refine_partition(th: Theory, cells: List[List[int]]) -> List[List[int]]:
    """Equitable refinement of an ordered partition of the fields: a cell splits by the
    signature of its members — for every term containing the field, its power there and
    the multiset of (cell index, power) of the term's other fields — sub-cells ordered by
    signature.  Deterministic, so the cell order is canonical up to ties inside cells."""
    while True:
        cell_of = {f: k for k, cell in enumerate(cells) for f in cell}
        new_cells: List[List[int]] = []
        split = False
        for cell in cells:
            if len(cell) == 1:
                new_cells.append(cell)
                continue
            sig: Dict[int, tuple] = {}
            for f in cell:
                items = []
                for t in th.terms:
                    if f in t:
                        items.append((t[f], tuple(sorted((cell_of[g], p) for g, p in t.items() if g != f))))
                sig[f] = tuple(sorted(items))
            groups: Dict[tuple, List[int]] = {}
            for f in cell:
                groups.setdefault(sig[f], []).append(f)
            if len(groups) > 1:
                split = True
            for key in sorted(groups):
                new_cells.append(sorted(groups[key]))
        cells = new_cells
        if not split:
            return cells


class CanonicalSearchLimit(RuntimeError):
    pass


def _canonical_order(th: Theory, limit: int = 200000) -> Tuple[List[int], tuple]:
    """Lexicographically minimal encoding over the field orderings compatible with the
    refined partition, by individualization and refinement with backtracking; returns
    (order, encoding).  Cells that no term touches are interchangeable and are not
    branched on."""
    n = th.n_fields()
    init: Dict[tuple, List[int]] = {}
    for f in range(n):
        init.setdefault(th.fields[f], []).append(f)
    cells0 = [init[k] for k in sorted(init)]
    touched = {f for t in th.terms for f in t}
    best: list = [None, None]
    leaves = [0]

    def rec(cells: List[List[int]]):
        cells = _refine_partition(th, cells)
        for k, cell in enumerate(cells):
            if len(cell) > 1 and any(f in touched for f in cell):
                for f in cell:
                    rest = [g for g in cell if g != f]
                    rec(cells[:k] + [[f], rest] + cells[k + 1:])
                return
        leaves[0] += 1
        if leaves[0] > limit:
            raise CanonicalSearchLimit(f"more than {limit} leaves")
        order = [f for cell in cells for f in cell]
        enc = _encode(th, order)
        if best[1] is None or enc < best[1]:
            best[0], best[1] = order, enc

    rec(cells0)
    return best[0], best[1]


@dataclass
class CanonicalForm:
    theory: Theory
    node_perm: List[int]         # canonical node k = source node node_perm[k]
    conj: List[bool]
    field_perm: List[int]        # canonical field k = source field field_perm[k]
    encoding: tuple

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.encoding, sort_keys=True).encode()).hexdigest()


def canonical_form(th: Theory) -> CanonicalForm:
    """Minimal encoding over permutations of identical nodes, conjugations of nodes and
    field orderings."""
    classes: Dict[tuple, List[int]] = {}
    for i, node in enumerate(th.nodes):
        classes.setdefault(node.key(), []).append(i)
    class_keys = sorted(classes)
    best = None
    perms_per_class = [list(itertools.permutations(classes[k])) for k in class_keys]
    for combo in itertools.product(*perms_per_class):
        node_perm = [i for group in combo for i in group]
        for conj in itertools.product([False, True], repeat=len(th.nodes)):
            rel = _relabel(th, node_perm, conj)
            order, enc = _canonical_order(rel)
            cand = (enc, node_perm, list(conj), order)
            if best is None or cand[0] < best[0]:
                best = cand
    enc, node_perm, conj, order = best
    rel = _relabel(th, node_perm, conj)
    pos = {old: new for new, old in enumerate(order)}
    canon = Theory(
        nodes=rel.nodes,
        fields=[rel.fields[old] for old in order],
        terms=[{pos[f]: p for f, p in t.items()} for t in rel.terms],
        flips={pos[f]: {pos[g]: p for g, p in op.items()} for f, op in rel.flips.items()},
        names=[rel.display_names()[old] for old in order],
    )
    return CanonicalForm(canon, node_perm, conj, order, enc)


# --------------------------------------------------------------------------- #
# index interfaces
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldResolvedTerm:
    coeff: F
    milli: int                    # 1000 * (t exponent)
    ypow: int
    markers: Tuple[int, ...]      # exponent of each field's symbol (negative = fermion)


@dataclass(frozen=True)
class PhysicalTerm:
    coeff: F
    milli: int
    ypow: int
    flavor: Tuple[int, ...]       # exponents of the U(1) fugacities in the basis used


def project(terms: Iterable[FieldResolvedTerm], basis: Sequence[Sequence[int]]) -> List[PhysicalTerm]:
    """Physical index from the field-resolved expansion: flavor_a = sum_f B[a][f] n_f;
    terms with equal (t, y, flavor) merge — the markers are what keeps colliding
    charges apart before this projection."""
    acc: Dict[tuple, F] = {}
    for t in terms:
        fl = tuple(sum(b * nf for b, nf in zip(row, t.markers)) for row in basis)
        key = (t.milli, t.ypow, fl)
        acc[key] = acc.get(key, F(0)) + t.coeff
    return sorted((PhysicalTerm(c, k[0], k[1], k[2]) for k, c in acc.items() if c),
                  key=lambda p: (p.milli, p.flavor, p.ypow))


def truncate(terms: Iterable, t_order: int) -> list:
    """Keep the terms with t exponent <= t_order (inclusive; milli units)."""
    return [t for t in terms if t.milli <= 1000 * t_order]


def index_array(terms: Iterable[PhysicalTerm]) -> list:
    return [[t.milli, t.ypow, list(t.flavor), str(t.coeff)] for t in
            sorted(terms, key=lambda p: (p.milli, p.flavor, p.ypow))]


CENTRAL_DIGITS = 25


def canonical_flavor_basis(th: Theory, canonical: Optional[CanonicalForm] = None, ambiguous: bool = False) -> List[List[int]]:
    """The flavor basis of a record (rows = U(1) generators, columns = fields in the SOURCE
    order): the Hermite normal form of the lattice in the canonical field order, its
    columns carried back through field_perm; the source-order lattice for a
    theory without a canonical identity (ambiguous superpotential terms).  The single
    copy of this construction; amax.flavor_basis and
    index.canonical_basis delegate here."""
    if ambiguous:
        return flavor_lattice(th)
    if canonical is None:
        canonical = canonical_form(th)
    B = flavor_lattice(canonical.theory)
    out = [[0] * th.n_fields() for _ in B]
    for k, src in enumerate(canonical.field_perm):
        for a, row in enumerate(B):
            out[a][src] = row[k]
    return out


def unrefined_index(terms: Iterable, t_order: int) -> List[Tuple[int, int, str]]:
    """[(milli, y, coefficient)] of the unrefined index below t^t_order -- the inherited
    strict truncation E < t_order: the reduced-index strings the old codes compared hold
    exactly this content (a stored index is complete at t^t_order, a converted old
    record's is not).  `terms` are PhysicalTerms or stored index rows
    [milli, y, flavor, coefficient]."""
    lim = 1000 * t_order
    acc: Dict[Tuple[int, int], F] = {}
    for t in terms:
        milli, ypow, coeff = (t.milli, t.ypow, t.coeff) if isinstance(t, PhysicalTerm) else (t[0], t[1], F(t[3]))
        if milli < lim:
            acc[(milli, ypow)] = acc.get((milli, ypow), F(0)) + coeff
    return [(m, y, str(c)) for (m, y), c in sorted(acc.items()) if c]


def central_string(value: Optional[str], digits: int = CENTRAL_DIGITS) -> Optional[str]:
    """A central charge (exact rational 'p/q' or a decimal string) rounded to `digits`
    significant digits (half even), as a plain decimal string; None stays None.  Equal
    strings are an exact, transitive relation; a change in any later digit can flip the
    rounding of the last kept one, so 'agreement to `digits` digits' here means equal
    roundings, not a tolerance."""
    if value is None:
        return None
    ctx = decimal.Context(prec=digits, rounding=decimal.ROUND_HALF_EVEN)
    if "/" in value:
        num, den = value.split("/")
        d = ctx.divide(decimal.Decimal(num), decimal.Decimal(den))
    else:
        d = ctx.plus(decimal.Decimal(value))
    return format(d.normalize(ctx), "f")


def identity_of_fixed_point(terms: Iterable, t_order: int, a: Optional[str], c: Optional[str]) -> Optional[str]:
    """The stored identity of a fixed point: sha256 of t_order, the unrefined index below
    t^t_order and the central charges a, c rounded half-even to CENTRAL_DIGITS significant
    digits (central_string) -- an exact, transitive key for a database.  The equivalence of
    two fixed points is that of arXiv:2408.02953 sec. 3 (equal unrefined index) with the
    central charges compared within a relative tolerance of 10^-CENTRAL_DIGITS
    (enumerate.equivalent).  The rounding is neither coarser nor finer than the tolerance
    (the two disagree at rounding boundaries in both directions), so equal identities are
    neither sufficient nor necessary for the enumeration's equivalence; on the checked
    records the two gave the same marks.  Complete retrieval of the candidates of a record
    therefore goes through the unrefined-index key (enumerate.fixed_point_key) and the
    predicate, as enumerate.mark_duplicates does; an identity lookup is an optimization only
    with that fallback.  Independent of the flavor basis and of the field order.  None
    without central charges."""
    if a is None or c is None:
        return None
    payload = json.dumps({"t_order": t_order, "unrefined": unrefined_index(terms, t_order),
                          "a": central_string(a), "c": central_string(c)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def record_identity(rec: dict) -> Optional[str]:
    """identity_of_fixed_point from a record dict; None without an index."""
    if rec.get("index") is None:
        return None
    return identity_of_fixed_point(rec["index"]["terms"], rec["index"]["t_order"], rec["charges"]["a"], rec["charges"]["c"])


def transform_flavor(exps: Sequence[int], T: Sequence[Sequence[int]]) -> Tuple[int, ...]:
    """Old exponents from new ones: e_old = T e_new (always integral)."""
    return tuple(sum(T[a][b] * exps[b] for b in range(len(exps))) for a in range(len(T)))


def inverse_transform_flavor(e_old: Sequence[int], L: Sequence[Sequence[F]]) -> Tuple[int, ...]:
    """New exponents from old ones through the left inverse L of T (L T = 1); an
    operator's charges are integral in the saturated basis, so the result must be."""
    if not L:
        return ()
    vals = [sum(L[i][j] * e_old[j] for j in range(len(e_old))) for i in range(len(L))]
    assert all(v.denominator == 1 for v in vals), f"non-integral new exponents {vals} for {e_old}"
    return tuple(int(v) for v in vals)


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
def make_record(th: Theory, *, charges: dict, flavor_basis: List[List[int]], index_terms: Optional[List[PhysicalTerm]],
                t_order: Optional[int], operators: dict, verdict: str, provenance: dict,
                canonical: Optional[CanonicalForm] = None, ambiguous: Optional[List[int]] = None,
                identity: Optional[str] = None) -> dict:
    rec = {
        "schema_version": SCHEMA_VERSION,
        "theory": th.to_json(),
        "charges": charges,
        "flavor": {"basis": flavor_basis, "rank": len(flavor_basis)},
        "index": None if index_terms is None else {"t_order": t_order, "terms": index_array(index_terms)},
        "operators": operators,
        "verdict": verdict,
        "canonical": None,
        "identity": identity,
        "provenance": provenance,
    }
    if canonical is not None:
        rec["canonical"] = {
            "hash": None if ambiguous else canonical.hash(),
            "ambiguous_terms": list(ambiguous or []),
            "node_perm": canonical.node_perm,
            "conj": canonical.conj,
            "field_perm": canonical.field_perm,
        }
    return rec
