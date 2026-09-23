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
    `max_order > 40` stop, the descendant factor sum_{a,b <= vec_order} (t^3 y)^a (t^3/y)^b
    and the Horner loop.

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


def program(th: Theory, charges: Sequence, t_order: int) -> Optional[str]:
    """FORM source for the index of `th` through t^t_order, or None when the expansion
    order exceeds MAX_ORDER (an R-charge too close to 0 or 2)."""
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
    return f"""#: maxtermsize 600000
Off statistics;
S y, z, r, s, t(: {t_order * 500}){markers};

CF d, {",".join(functions)};
Polyratfun d;

L itotal = {"+".join(terms)};
.sort

L I = z;
id z = z * itotal;
#do i=2, {max_order}
  id z = 1 + z * itotal / `i';
  .sort:step `i';
#enddo
.sort

L result = (1 + I);
.sort
Print result;
.end
"""


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
