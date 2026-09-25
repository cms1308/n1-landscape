"""FORM program of the superconformal index of a theory.

The program is that of the original landscape code (with an explicit itotal), written
for the model:

  * one character function per (node, Dynkin label), named by singlet.char_symbol and
    shared by all fields with that label at that node; a field in R_1 x ... x R_k
    carries the same Adams index j at every node it charges, its conjugate fermion the
    conjugate labels;
  * one positional marker symbol per field, f1..fn (f^-1 on the fermion letter); no
    flavor fugacities — the flavor exponents of a term are B . markers (model.project);
  * one vector multiplet per node, (-t^3 y - t^3/y + 2 t^6) x adjoint character;
  * inherited from the original code: the exponent encoding t^p -> t^{int(500p)} s^{d1} r^{d2}
    with base-5000 digits (`encode`, the arithmetic of `single` at the default Decimal
    context), the truncation `t(: 500 t_order)`, the expansion orders (`get_order`), the
    `max_order > 40` stop and the descendant factor sum_{a,b <= vec_order} (t^3 y)^a (t^3/y)^b;
  * the exponential as the loop z -> 1 + z itotal / i, i = 2..max_order, with a degree
    bound (`scheme="bounded"`, the default): a term whose t-power lies in block k
    (500 k <= power <= 500 k + 499) is multiplied only by itotal<m>, m = t_order - k, the
    part of itotal with t-power at most 500 m -- every product that survives the
    truncation is generated, the rest is not.  The original loop, every term multiplied
    by the whole itotal and the products above the truncation discarded by FORM, is
    `scheme="horner"`; both give the same polynomial.
  * the rational coefficients as FORM's own numbers during the expansion, and the
    PolyRatFun `d` -- the output representation `d(n,m)` the parser reads -- declared
    only before the final `result` (`polyratfun="late"`, the default): every coefficient
    is a plain rational, and carrying it through the polynomial-rational-function
    machinery of a PolyRatFun costs about a third of the wall of a heavy program.  The
    original program declares the PolyRatFun from the start (`polyratfun="early"`, kept
    as the comparison baseline); both give the same polynomial, printed in a slightly
    different term order.

The j-range of a field is its own get_order (the original code used the maximum over the
species); letters beyond the truncation are dropped by FORM, so the output through
t^{t_order} is the same.

Scratch files: FORM's TempDir is a per-process
directory under the runner's work directory (`-t`), never the current directory, and it
is emptied after a timeout (FORM removes its own files on a normal exit; a killed process
leaves its sort files behind).
"""
from __future__ import annotations

import math
import os
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import lie
from .convert import render_fraction
from .model import Theory
from .singlet import char_symbol

MAX_ORDER = 40
FORM_TIMEOUT_S = 600
TFORM_THRESHOLD_BYTES = 2000


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


def marker(f: int) -> str:
    return f"f{f + 1}"


def _power(sym: str, e: int) -> str:
    return f"{sym}^{e}" if e >= 0 else f"{sym}^({e})"


def _letter(tsr: Tuple[int, int, int], j: int, mark: Optional[str], sign: int) -> str:
    factors = [_power(b, e * j) for b, e in zip(("t", "s", "r"), tsr) if e]
    if mark is not None:
        factors.append(_power(mark, sign * j))
    return "*".join(factors) if factors else "1"


SCHEMES = ("bounded", "horner")
POLYRATFUN = ("late", "early")


def program(th: Theory, charges: Sequence, t_order: int, scheme: str = "bounded", polyratfun: str = "late") -> Optional[str]:
    """FORM source for the index of `th` through t^t_order, or None when the expansion
    order exceeds MAX_ORDER (an R-charge too close to 0 or 2).  `scheme` selects the
    loop of the exponential: "bounded" (degree-bounded multiplication, the default) or
    "horner" (the original loop, kept as the comparison baseline); `polyratfun` where the
    PolyRatFun d is declared: "late" (before the final result only, the default) or
    "early" (from the start, the original program, the comparison baseline); see the
    module docstring."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; one of {SCHEMES}")
    if polyratfun not in POLYRATFUN:
        raise ValueError(f"unknown polyratfun {polyratfun!r}; one of {POLYRATFUN}")
    r_str = [charge_string(r) for r in charges]
    assert len(r_str) == th.n_fields(), "one R-charge per field"
    orders = [get_order(t_order, [r]) for r in r_str]
    vec_order = get_order(t_order, [1])
    max_order = max(orders + [vec_order])
    if max_order > MAX_ORDER:
        return None

    def J(j):
        return "(" + "+".join(
            f"t^{1500 * (a + b) * j}*y^{(a - b) * j}" if a != b else f"t^{1500 * (a + b) * j}"
            for a in range(vec_order + 1) for b in range(vec_order + 1)) + ")"

    functions: List[str] = []

    def chars(f: int, j: int, conj: bool) -> str:
        out = ""
        for i, node in enumerate(th.nodes):
            lab = th.fields[f][i]
            if any(lab):
                if conj:
                    lab = lie.conjugate(node.type, node.rank, lab)
                name = char_symbol(i, lab)
                if name not in functions:
                    functions.append(name)
                out += f"*{name}({j})"
        return out

    terms = []
    for f, r in enumerate(r_str):
        r_val = Decimal(r)
        boson, fermion = encode(3 * r_val), encode(6 - 3 * r_val)
        for j in range(1, orders[f] + 1):
            terms.append(f"{J(j)}*(({_letter(boson, j, marker(f), 1)}){chars(f, j, False)}"
                         f"-({_letter(fermion, j, marker(f), -1)}){chars(f, j, True)})/{j}")
    for i, node in enumerate(th.nodes):
        adj = char_symbol(i, lie.highest_root(node.type, node.rank))
        if adj not in functions:
            functions.append(adj)
        for j in range(1, vec_order + 1):
            terms.append(f"{J(j)}*(-t^{1500 * j}*y^{j}-t^{1500 * j}*y^(-{j})+2*t^{3000 * j})*{adj}({j})/{j}")

    markers = "".join(f",{marker(f)}" for f in range(th.n_fields()))
    declare_d = "CF d;\nPolyratfun d;\n"
    head = f"""#: maxtermsize 600000
Off statistics;
S y, z, r, s, t(: {t_order * 500}){markers};

CF {",".join(functions)};
{declare_d if polyratfun == "early" else ""}
L itotal = {"+".join(terms)};
.sort
"""
    if scheme == "horner":
        loop = f"""
L I = z;
id z = z * itotal;
#do i=2, {max_order}
  id z = 1 + z * itotal / `i';
  .sort:step `i';
#enddo
.sort
"""
    else:
        # itotal<m> = the terms of itotal with t-power at most 500 m, m = 0..t_order (itotal<t_order>
        # is itotal itself); a term in block k of the exponential is multiplied by itotal<t_order - k>.
        # The auxiliary expressions are hidden: available on the right-hand side, not processed.
        loop = f"""
#do m=0,{t_order}
L itotal`m' = itotal;
#enddo
.sort
#do m=0,{t_order}
if ( expression(itotal`m') && (count(t,1) > {{500*`m'}}) ) discard;
#enddo
.sort

Hide itotal;
#do m=0,{t_order}
Hide itotal`m';
#enddo
L I = z;
id z = z * itotal;
#do i=2, {max_order}
  #do k={t_order},0,-1
    if ( (count(t,1) >= {{500*`k'}}) && (count(t,1) <= {{500*`k'+499}}) ) id z = 1 + z * itotal{{{t_order}-`k'}} / `i';
  #enddo
  .sort:step `i';
#enddo
.sort
"""
    return head + loop + (declare_d if polyratfun == "late" else "") + """
L result = (1 + I);
.sort
Print result;
.end
"""


# --------------------------------------------------------------------------- #
# the series as data: the input of the native expansion engine (step 39f)
# --------------------------------------------------------------------------- #
class Series:
    """The explicit itotal of `program` as data, for an expansion engine other than FORM.

    A monomial is an exponent vector [t, s, r, y, f_1..f_n, c_1..c_m]: the t-power (units of
    1/500), the s and r digits, the y power, one marker exponent per field, one exponent per
    character slot -- the slot j of `slots` is (node, Dynkin label, Adams index k), the symbol
    C<node>L<label>(k) of the program, with its exponent the power of that symbol.  `terms` are
    (numerator, denominator, exponents) with the t-power at most `t_limit` (the truncation
    `t(:500 t_order)`), equal monomials combined, in the order the program writes them.  The
    exponential of the series truncated at `t_limit` and at `max_order` powers is FORM's
    `result`; a theory whose expansion order exceeds MAX_ORDER has no Series (`program` stops)."""

    def __init__(self, t_order: int, t_limit: int, max_order: int, n_fields: int, n_nodes: int, slots, terms):
        self.t_order, self.t_limit, self.max_order = t_order, t_limit, max_order
        self.n_fields, self.n_nodes = n_fields, n_nodes
        self.slots = slots                       # [(node, label tuple, k), ...]
        self.terms = terms                       # [(num, den, [t, s, r, y, f..., c...]), ...]


def itotal_terms(th: Theory, charges: Sequence, t_order: int) -> Optional[Series]:
    """The Series of `th` (the same letters, descendant factor, orders and truncation as
    `program`), or None where `program` stops (max_order > MAX_ORDER)."""
    r_str = [charge_string(r) for r in charges]
    assert len(r_str) == th.n_fields(), "one R-charge per field"
    orders = [get_order(t_order, [r]) for r in r_str]
    vec_order = get_order(t_order, [1])
    max_order = max(orders + [vec_order])
    if max_order > MAX_ORDER:
        return None
    t_limit = 500 * t_order
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


class FormRunner:
    """Runs FORM programs in a work directory, with the inherited output cleaning and an
    automatic TFORM policy: an expansion runs under `tform -w<workers>` when the most recent
    lower-order output of this runner exceeded `threshold` bytes, sequentially otherwise
    (workers = 0: always sequential)."""

    def __init__(self, workdir: str | Path, tform_workers: int = 4,
                 threshold: int = TFORM_THRESHOLD_BYTES, timeout: float = FORM_TIMEOUT_S):
        self._dir = Path(workdir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._workers = tform_workers
        self._threshold = threshold
        self._timeout = timeout
        self._last: Optional[Tuple[int, int]] = None      # (t_order, output bytes)
        self.last_runner = None
        self.last_cleanup = 0                             # files removed after the last timeout

    def tempdir(self) -> Path:
        """FORM's TempDir for this process: <workdir>/tmp<pid>, created on demand."""
        tmp = self._dir / f"tmp{os.getpid()}"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp

    def clear_tempdir(self) -> int:
        """Remove every file FORM left in this process's TempDir; returns the count."""
        tmp = self._dir / f"tmp{os.getpid()}"
        n = 0
        if tmp.is_dir():
            for f in tmp.iterdir():
                if f.is_file():
                    f.unlink()
                    n += 1
        return n

    def run(self, source: str, t_order: int) -> Optional[str]:
        """Cleaned FORM output ('+'-separated terms), or None on a FORM timeout."""
        heavy = (self._workers > 0 and self._last is not None
                 and self._last[0] < t_order and self._last[1] > self._threshold)
        frm = self._dir / f"index{os.getpid()}.frm"
        frm.write_text(source)
        tmp = self.tempdir()
        cmd = ["tform", f"-w{self._workers}", "-t", str(tmp), "-q", str(frm)] if heavy else ["form", "-t", str(tmp), "-q", str(frm)]
        self.last_runner = cmd[0]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=self._timeout)
        except subprocess.TimeoutExpired:
            self.last_cleanup = self.clear_tempdir()
            return None
        finally:
            frm.unlink(missing_ok=True)
        out = (res.stdout.strip().replace("result", "").replace(" ", "").replace("=", "")
               .replace("\n", "").replace("z", "1").replace("\\", ""))[:-1]
        self._last = (t_order, len(out))
        return out
