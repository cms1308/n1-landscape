"""The series whose plethystic exponential the expansion engine computes: the letters of a theory and its expansion orders.

The series is the explicit itotal of the original landscape code (which wrote it as a FORM program), as data for the
native engine (`Series`, `itotal_terms`):

  * one character slot per (node, Dynkin label, Adams index k), shared by all fields with that label at that node; a field
    in R_1 x ... x R_k carries the same Adams index j at every node it charges, its conjugate fermion the conjugate labels;
  * one positional marker exponent per field (-j on the fermion letter); no flavor fugacities -- the flavor exponents of a
    term are B . markers (model.project);
  * one vector multiplet per node, (-t^3 y - t^3/y + 2 t^6) x adjoint character;
  * inherited from the original code: the exponent encoding t^p -> t^{int(500p)} s^{d1} r^{d2} with base-5000 digits
    (`encode`, the arithmetic of `single` at the default Decimal context), the truncation of the integer t-power at
    500 t_order, the expansion orders (`get_order`), the `max_order > MAX_ORDER` stop (40 in the original code, 100 here;
    None lifts it) and the descendant factor sum_{a,b <= vec_order} (t^3 y)^a (t^3/y)^b.

The j-range of a field is its own get_order.  The fields a mass term makes massive have no letters (their letters cancel
in the flavor-refined index; `mass`), and the expansion order is that of the remaining fields.  A remaining field whose
boson or fermion letter falls at t^0 on the 1/1000 grid of the post-processing (3R or 3(2 - R) below 0.0005, R = 0 and 2
included) stops the expansion: the truncation does not bound such a letter, and its operators are read at t^0; so does
an expansion order above MAX_ORDER (at any truncation; a field of R-charge too close to 0 or 2).  An
expansion the engine computes is cut when it has formed more than WORK_BOUND product terms (a count independent of the
machine and its load); EXPANSION_TIMEOUT_S is a safety net.  These are the expansions a build leaves uncomputed by design.
A truncation order may be a fraction, a multiple of 1/500 (LOW_TRUNCATIONS, the decoupling pass of a theory whose order at
t^3 exceeds LOW_TRUNCATION_ORDER).
"""
from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

from . import lie, mass
from .convert import render_fraction
from .model import Theory

MAX_ORDER = 100                 # the stop of the expansion order (40 in the original code; None lifts it, for a later run)
LOW_TRUNCATION_ORDER = 100      # a theory whose expansion order at t^3 exceeds it runs the decoupling pass on LOW_TRUNCATIONS first
LOW_TRUNCATIONS = tuple(Fraction(k, 500) for k in (4, 8, 16, 32, 64, 125, 250, 500, 1000))   # 0.008 ... 2, multiples of 1/500
WORK_BOUND = 10**10             # the work bound of one expansion: the product terms the engine forms (landscape_native.last_work)
EXPANSION_TIMEOUT_S = 86400     # the deadline of one expansion: a safety net, far above the work bound on a single core


def charge_string(r) -> str:
    """The string the exponent arithmetic starts from: a decimal string as recorded, or
    the printed form of an exact rational, render_fraction (`p/q` strings included)."""
    if isinstance(r, Fraction):
        return render_fraction(r)
    s = str(r).strip()
    return render_fraction(Fraction(s)) if "/" in s else s


def _round_half_up(val) -> int:
    return int(Decimal(str(val)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def encode(weight: Decimal) -> Tuple[int, int, int]:
    """(t, s, r) powers of a letter of t-weight `weight`."""
    p = weight * 500
    return int(p), int(p % 1 * 5000), _round_half_up((p % 1 * 5000) % 1 * 5000)


def get_order(t_order: int, r_list: Sequence) -> int:
    """Order of the plethystic expansion for the given R-charges."""
    if not r_list:
        return 0
    l = []
    for i in r_list:
        i_f = float(i)
        if i_f != 0:
            l.append(t_order / (3 * i_f))
        if i_f != 2:
            l.append(t_order / (6 - 3 * i_f))
    return math.ceil(max(l))


def letter_milli(weight: Decimal) -> int:
    """The milli exponent of a letter t^weight on the grid of the post-processing: round half up of 1000 weight, from the
    encoding (the rule of the native rows)."""
    t, d1, d2 = encode(weight)
    return (25_000_000 * t + 5_000 * d1 + d2 + 6_250_000) // 12_500_000


def _orders(th: Theory, r_str: Sequence[str], t_order, cap: bool = True) -> Optional[Tuple[set, List[int], int, int]]:
    """(massive fields, expansion order per field, vector order, max order), or None where the
    expansion stops: a remaining field whose boson or fermion letter falls at t^0 on the grid
    (letter_milli 0; R = 0 and 2 included), which the truncation does not bound, or an order above
    MAX_ORDER where it is set (and `cap`).  The fields a mass term makes massive (mass.massive_fields)
    have no letters and no order."""
    massive = set(mass.massive_fields(th.terms))
    if any(letter_milli(3 * Decimal(r)) == 0 or letter_milli(6 - 3 * Decimal(r)) == 0
           for f, r in enumerate(r_str) if f not in massive):
        return None
    orders = [0 if f in massive else get_order(t_order, [r]) for f, r in enumerate(r_str)]
    vec_order = get_order(t_order, [1])
    max_order = max(orders + [vec_order])
    if cap and MAX_ORDER is not None and max_order > MAX_ORDER:
        return None
    return massive, orders, vec_order, max_order


# --------------------------------------------------------------------------- #
# the series as data: the input of the native expansion engine
# --------------------------------------------------------------------------- #
class Series:
    """The explicit itotal as data, the input of the expansion engine.

    A monomial is an exponent vector [t, s, r, y, f_1..f_n, c_1..c_m]: the t-power (units of
    1/500), the s and r digits, the y power, one marker exponent per field, one exponent per
    character slot -- the slot j of `slots` is (node, Dynkin label, Adams index k), psi^k of the
    character of that label at that node, with its exponent the power of that character.  `terms`
    are (numerator, denominator, exponents) with the t-power at most `t_limit` (500 t_order),
    equal monomials combined.  The expansion is the exponential of the series truncated at
    `t_limit` and at `max_order` powers; a theory whose expansion stops (`_orders`) has no Series.
    `t_order` may be a fraction, a multiple of 1/500 (a lower truncation of the decoupling pass)."""

    def __init__(self, t_order: int, t_limit: int, max_order: int, n_fields: int, n_nodes: int, slots, terms):
        self.t_order, self.t_limit, self.max_order = t_order, t_limit, max_order
        self.n_fields, self.n_nodes = n_fields, n_nodes
        self.slots = slots                       # [(node, label tuple, k), ...]
        self.terms = terms                       # [(num, den, [t, s, r, y, f..., c...]), ...]


def expansion_stops(th: Theory, charges: Sequence, t_order) -> bool:
    """Whether the expansion of order t_order stops before the engine (a remaining field whose letter falls at t^0 on
    the grid, or an order above MAX_ORDER where it is set): itotal_terms gives None."""
    return _orders(th, [charge_string(r) for r in charges], t_order) is None


def low_truncations(th: Theory, charges: Sequence) -> Tuple[Fraction, ...]:
    """LOW_TRUNCATIONS where the expansion order at t^3 exceeds LOW_TRUNCATION_ORDER, else none: the decoupling pass
    then reads the lower truncations first, each deciding a flip only (a term below T is complete in the expansion
    truncated at T).  A truncation at or below the lowest letter of the remaining fields is left out: nothing lies
    below it, and its series is empty.  The order at t^3 is read without MAX_ORDER, which then stops a lower truncation
    whose own order exceeds it."""
    r_str = [charge_string(r) for r in charges]
    stop = _orders(th, r_str, 3, cap=False)
    if stop is None or stop[3] <= LOW_TRUNCATION_ORDER:
        return ()
    lowest = min((min(3 * Decimal(r), 6 - 3 * Decimal(r)) for f, r in enumerate(r_str) if f not in stop[0]), default=Decimal(3))
    return tuple(k for k in LOW_TRUNCATIONS if k > Fraction(str(lowest)))


def itotal_terms(th: Theory, charges: Sequence, t_order) -> Optional[Series]:
    """The Series of `th`, or None where the expansion stops (`_orders`)."""
    r_str = [charge_string(r) for r in charges]
    assert len(r_str) == th.n_fields(), "one R-charge per field"
    stop = _orders(th, r_str, t_order)
    if stop is None:
        return None
    massive, orders, vec_order, max_order = stop
    t_limit = Fraction(t_order) * 500
    assert t_limit.denominator == 1, "a truncation order is a multiple of 1/500"
    t_limit = int(t_limit)
    n = th.n_fields()
    slots: List[Tuple[int, Tuple[int, ...], int]] = []
    slot_index: dict = {}

    def slot(node: int, label, k: int) -> int:
        key = (node, tuple(int(x) for x in label), k)
        if key not in slot_index:
            slot_index[key] = len(slots)
            slots.append(key)
        return slot_index[key]

    raw: List[Tuple[Fraction, tuple]] = []      # (coefficient, exponent key) before combining
    jterms = {j: [(1500 * (a + b) * j, (a - b) * j) for a in range(vec_order + 1) for b in range(vec_order + 1)]
              for j in range(1, max_order + 1)}

    def add(coeff: Fraction, j: int, tsr: Tuple[int, int, int], field: Optional[int], sign: int, char_slots: List[int]):
        for tj, yj in jterms[j]:
            t = tsr[0] * j + tj
            if t > t_limit:
                continue
            f_exps = [0] * n
            if field is not None:
                f_exps[field] = sign * j
            c_exps = {}
            for c in char_slots:
                c_exps[c] = c_exps.get(c, 0) + 1
            raw.append((coeff, (t, tsr[1] * j, tsr[2] * j, yj, tuple(f_exps), tuple(sorted(c_exps.items())))))

    for f, r in enumerate(r_str):
        if f in massive:
            continue
        r_val = Decimal(r)
        boson, fermion = encode(3 * r_val), encode(6 - 3 * r_val)
        for j in range(1, orders[f] + 1):
            bos_slots, fer_slots = [], []
            for i, node in enumerate(th.nodes):
                lab = th.fields[f][i]
                if any(lab):
                    bos_slots.append(slot(i, lab, j))
                    fer_slots.append(slot(i, lie.conjugate(node.type, node.rank, lab), j))
            add(Fraction(1, j), j, boson, f, 1, bos_slots)
            add(Fraction(-1, j), j, fermion, f, -1, fer_slots)
    for i, node in enumerate(th.nodes):
        adj = lie.highest_root(node.type, node.rank)
        for j in range(1, vec_order + 1):
            sl = [slot(i, adj, j)]
            # (-t^{1500 j} y^j - t^{1500 j} y^{-j} + 2 t^{3000 j}) adj(j) / j, with the descendant factor
            for tj, yj in jterms[j]:
                for (tpow, ypow, c) in ((1500 * j, j, Fraction(-1, j)), (1500 * j, -j, Fraction(-1, j)), (3000 * j, 0, Fraction(2, j))):
                    t = tpow + tj
                    if t > t_limit:
                        continue
                    raw.append((c, (t, 0, 0, ypow + yj, tuple([0] * n), ((sl[0], 1),))))
    combined: dict = {}
    order_keys: List[tuple] = []
    for c, key in raw:
        if key not in combined:
            combined[key] = Fraction(0)
            order_keys.append(key)
        combined[key] += c
    m = len(slots)
    terms = []
    for key in order_keys:
        c = combined[key]
        if c == 0:
            continue
        t, sp, rp, y, f_exps, c_items = key
        c_exps = [0] * m
        for idx, e in c_items:
            c_exps[idx] = e
        terms.append((c.numerator, c.denominator, [t, sp, rp, y, *f_exps, *c_exps]))
    return Series(t_order, t_limit, max_order, n, len(th.nodes), slots, terms)


def expand_series_reference(series: Series):
    """The exponential of the series in exact Python arithmetic, truncated at t_limit and at
    max_order powers -- the reference of the native engine on small programs: {exponent
    tuple: Fraction}, the constant term included."""
    from collections import defaultdict
    t_limit = series.t_limit
    itotal = [(Fraction(n, d), tuple(e)) for n, d, e in series.terms]
    itotal.sort(key=lambda x: x[1][0])
    result: dict = defaultdict(Fraction)
    zero = tuple([0] * (4 + series.n_fields + len(series.slots)))
    result[zero] += 1
    power: dict = {}
    for c, e in itotal:
        power[e] = power.get(e, Fraction(0)) + c
    for k in range(1, series.max_order + 1):
        if k > 1:
            nxt: dict = defaultdict(Fraction)
            for e_p, c_p in power.items():
                for c_i, e_i in itotal:
                    if e_i[0] > t_limit - e_p[0]:
                        break
                    nxt[tuple(a + b for a, b in zip(e_p, e_i))] += c_p * c_i / k
            power = {e: c for e, c in nxt.items() if c}
            if not power:
                break
        for e, c in power.items():
            result[e] += c
    return {e: c for e, c in result.items() if c}
