"""Gauge-singlet projection of character products for a product of simple groups, and the call of the
native expansion engine.

`expand_series_native` hands a `form.Series` to the extension (`landscape_native.expand_series`): the
irreducible mode -- one highest weight per node in place of the character slots of a monomial, a
product with a letter decomposed at once by the Brauer-Klimyk rule, the rows the entries whose highest
weights are all zero -- or, for the checks, the slot engine, which asks `ProductProjector.multiplicity`
for a character product it cannot project itself.

A character product carries, per node, the Adams multiplicity key of every label present; its singlet
multiplicity is the product over the nodes of the singlet multiplicity of that node's character product
(the singlet of R_1 x ... x R_k of a product group is the product of the per-node singlets).  One label is
one store lookup.  Several labels are split into two parts at label granularity, balanced by the product
of the term counts of their stored decompositions; each part is decomposed through the node's own
tensor-step cache (one label needs no tensor step, a part of several labels a chain of tensor steps
starting from its first decomposition), and the two parts are contracted by the pairing of dual
representations, sum over lambda of a_lambda b_{lambda*} with lambda* the conjugate highest weight -- the
singlet multiplicity of the product of two virtual characters, so the complete product is never
tensor-decomposed (scheme "split", the default).  The left-to-right chain through the complete product
(scheme "chain") is kept as the comparison baseline.  The tensor steps run in the native character engine
(`store.run_lie`, LiE's output format), with the inherited [53:] banner slice, 'line'-triggered maxobjects
retry and validity gate before caching.
"""
from __future__ import annotations

import functools
import math
import os
import subprocess
import threading
import time
from typing import Dict, Optional, Sequence, Tuple

import landscape_native as _native

from . import lie
from .store import LabelStore, run_lie, singlet_coefficient, label_key

Label = Tuple[int, ...]
KeyVec = Tuple[int, ...]
NodeChars = Tuple[Tuple[Label, KeyVec], ...]      # sorted by label
Chars = Tuple[NodeChars, ...]                     # one entry per node


def _switch(var: str, default: str = "1") -> bool:
    return os.environ.get(var, default).strip().lower() not in ("0", "off", "no", "")


# The native post-processing pass (LANDSCAPE_NATIVE_POST, on by default): the engine keeps its rows in the
# extension, and post.index / post.decouple / post.prefilter and model.project read the reduced index, its
# flavor projection, the net index and the physical index from one native pass over them; the Python
# functions are the reference and the fallback (a plain list of FieldResolvedTerms, or an overflow).
NATIVE_POST = _switch("LANDSCAPE_NATIVE_POST")
# The flavor projection above t^6 (LANDSCAPE_NATIVE_FLAVOR, on by default): the rows above t^6 are
# flavor-refined, which only the native post pass consumes, so it needs NATIVE_POST.  The 64-bit coefficient
# path (LANDSCAPE_NATIVE_COEF64) and the exact-milli truncation (LANDSCAPE_NATIVE_EXACT): off by default.
NATIVE_FLAVOR = NATIVE_POST and _switch("LANDSCAPE_NATIVE_FLAVOR")
NATIVE_COEF64 = _switch("LANDSCAPE_NATIVE_COEF64", "0")
NATIVE_EXACT = _switch("LANDSCAPE_NATIVE_EXACT", "0")
# The full-order expansion computed flavor-refined throughout and spliced with the order-6 expansion's
# rows through t^6, the same terms (on by default; LANDSCAPE_NATIVE_SPLICE=0 computes the full expansion).
NATIVE_SPLICE = NATIVE_FLAVOR and _switch("LANDSCAPE_NATIVE_SPLICE")


class NativeOverflow(ArithmeticError):
    """A flavor-refined row of the expansion beyond 128 bits; the caller asks for the expansion field-resolved, whose rows
    then come with arbitrary-precision coefficients."""


class NativeCapacity(NativeOverflow):
    """The slot engine's expansion exceeds its monomial cap (LANDSCAPE_NATIVE_MAX_TERMS, default 8,000,000 per power;
    engine="slot" only -- the irreducible mode spills a power beyond its budget to disk)."""


class WorkBound(RuntimeError):
    """The expansion was cut by its work bound: the product terms the irreducible mode formed exceeded `work_bound`
    (`terms`, the count at the cut)."""

    def __init__(self, terms: int):
        super().__init__(f"work-bound: {terms} product terms formed")
        self.terms = terms


def last_work() -> int:
    """The product terms the last expansion of the irreducible mode in this process formed (up to the cut)."""
    return int(_native.last_work())


def expand_series_native(series, projector: "ProductProjector", timeout: Optional[float] = None,
                         budget_s: Optional[float] = None, monomials: bool = False, native_rows: bool = False,
                         basis=None, exact: bool = False, coef64: bool = False, flavor_only: bool = False,
                         engine: str = "irrep", work_bound: Optional[int] = None):
    """The native expansion of a `form.Series`: the rows of the field-resolved expansion, a list of (numerator,
    denominator, milli, y, markers) or a FieldResolvedRows object when native_rows is set (with `basis`, the rows above
    t^6 flavor-refined by it, the object only).  `engine`: "irrep", the irreducible mode (the package's engine), or
    "slot", the slot engine -- the checks' reference, which with monomials=True gives the polynomial itself
    [(numerator, denominator, exponents)] and asks `projector.multiplicity` where the extension cannot project a
    character product.  `exact` and `coef64` are refinements of the engine, off by default.  A row beyond 128 bits of a
    field-resolved expansion comes in the list with arbitrary-precision coefficients; NativeOverflow when a row beyond
    128 bits is flavor-refined (the caller asks for the expansion field-resolved), NativeCapacity at the slot engine's
    cap; subprocess.TimeoutExpired past the timeout; WorkBound when the irreducible mode forms more than `work_bound`
    product terms."""
    try:
        return _native.expand_series(series.terms, series.t_limit, series.max_order, series.n_fields,
                                     [(node, list(label), k) for node, label, k in series.slots], series.n_nodes,
                                     math.ceil(series.t_order), lambda chars: projector.multiplicity(chars, budget_s), timeout, monomials,
                                     native_rows, basis, exact, coef64,
                                     [(s.lie_group, s.type, s.rank) for s in projector._stores],
                                     flavor_only=flavor_only, engine=engine, work_bound=work_bound,
                                     milli_limit=None if series.t_order == int(series.t_order) else int(1000 * series.t_order))
    except ValueError as e:
        if str(e).startswith("work-bound: "):
            raise WorkBound(int(str(e).split()[1])) from e
        if str(e).startswith("capacity"):
            raise NativeCapacity(str(e)) from e
        if "overflow" in str(e):
            raise NativeOverflow(str(e)) from e
        raise


SCHEMES = ("split", "chain")


class ProductProjector:
    """Gauge-singlet projection of character products for a product of simple groups: one
    LabelStore per node (character decompositions and tensor-step cache), the singlet
    multiplicity = product over nodes; per node the balanced split and the pairing of
    dual representations (scheme "split"), or the chain through the complete product
    (scheme "chain", the comparison baseline)."""

    def __init__(self, stores: Sequence[LabelStore], lie_timeout: float = 180.0,
                 lie_runner=None, timeout_error=subprocess.TimeoutExpired, scheme: str = "split"):
        if scheme not in SCHEMES:
            raise ValueError(f"unknown projection scheme {scheme!r}; one of {SCHEMES}")
        self._stores = list(stores)
        self._scheme = scheme
        self._conj = [functools.partial(lie.conjugate, s.type, s.rank) for s in self._stores]
        self._lie_timeout = lie_timeout
        self._lie_runner = lie_runner if lie_runner is not None else run_lie
        self._timeout_error = timeout_error
        self._step_memo: Dict[str, str] = {}
        self._sig_memo: Dict[tuple, int] = {}
        self._poly_memo: Dict[str, Dict[Label, int]] = {}
        self._lock = threading.Lock()
        self.lie_calls = 0

    @property
    def scheme(self) -> str:
        return self._scheme

    @property
    def n_nodes(self) -> int:
        return len(self._stores)

    def _tensor_step(self, i: int, products: str, decomp: str, deadline: float) -> str:
        store = self._stores[i]
        key = store.cache_key(products, decomp)
        with self._lock:
            cached = self._step_memo.get(key)
        if cached is not None:
            return cached
        cached = store.cache_get(key)
        if cached is not None:
            with self._lock:
                self._step_memo[key] = cached
            return cached
        group = store.lie_group
        out = ""
        for preamble in ("maxnodes 9999999 \n", "maxobjects 9999999 \n maxnodes 9999999 \n"):
            remaining = deadline - time.time()
            if remaining <= 0:
                raise self._timeout_error("lie chain wall-clock cutoff", self._lie_timeout)
            lcode = f"{preamble}res=tensor({products},{decomp},{group});\nprint(res);"
            try:
                with self._lock:
                    self.lie_calls += 1
                out = self._lie_runner(lcode, min(self._lie_timeout, remaining))[53:].strip()
                out = out.replace("\n", "").replace(" ", "")
            except subprocess.TimeoutExpired:
                raise self._timeout_error("character engine timeout", self._lie_timeout)
            if "line" not in out:
                break
        if out and "X" in out and "line" not in out:
            with self._lock:
                self._step_memo[key] = out
            store.cache_put(key, out)
        return out

    def poly(self, decomp: str) -> Dict[Label, int]:
        """A LiE virtual-character string as {highest weight: multiplicity}, parsed once
        per string."""
        with self._lock:
            p = self._poly_memo.get(decomp)
        if p is None:
            p = lie.parse_poly(decomp)
            with self._lock:
                self._poly_memo[decomp] = p
        return p

    def pair(self, i: int, a: str, b: str) -> int:
        """Singlet multiplicity of the product of two virtual characters of node i:
        sum over lambda of a_lambda b_{lambda*}, lambda* the conjugate highest weight
        (the trivial representation occurs in lambda x mu iff mu = lambda*)."""
        A, B = self.poly(a), self.poly(b)
        if len(A) > len(B):
            A, B = B, A
        conj = self._conj[i]
        return sum(c * B.get(conj(lam), 0) for lam, c in A.items())

    def split(self, i: int, chars_i: NodeChars) -> Tuple[NodeChars, NodeChars]:
        """The node's labels (in their sorted order) in two parts, balanced by the product
        of the term counts of their stored decompositions: over every bipartition with
        the first label in the first part, the smallest larger product wins, the first
        such bipartition on a tie."""
        store = self._stores[i]
        weights = [store.decomp(label, key_vec).count("X") for label, key_vec in chars_i]
        n = len(chars_i)
        best_mask, best_score = None, None
        for mask in range(1, 1 << (n - 1)):          # bit j set: label j + 1 in the second part
            w1 = w2 = 1
            for j in range(n):
                if j and (mask >> (j - 1)) & 1:
                    w2 *= weights[j]
                else:
                    w1 *= weights[j]
            score = max(w1, w2)
            if best_score is None or score < best_score:
                best_mask, best_score = mask, score
        first = tuple(c for j, c in enumerate(chars_i) if not (j and (best_mask >> (j - 1)) & 1))
        second = tuple(c for j, c in enumerate(chars_i) if j and (best_mask >> (j - 1)) & 1)
        return first, second

    def partial_product(self, i: int, part: NodeChars, deadline: float) -> str:
        """LiE string of the product of a part's decompositions: the decomposition itself
        for one label, otherwise a chain of tensor steps through the cache starting from
        the first label's decomposition."""
        store = self._stores[i]
        products = store.decomp(*part[0])
        for label, key_vec in part[1:]:
            products = self._tensor_step(i, products, store.decomp(label, key_vec), deadline)
        return products

    def node_multiplicity(self, i: int, chars_i: NodeChars, budget_s: Optional[float] = None) -> int:
        """Singlet multiplicity of one node's character product."""
        if not chars_i:
            return 1
        store = self._stores[i]
        if len(chars_i) == 1:
            label, key_vec = chars_i[0]
            return singlet_coefficient(store.decomp(label, key_vec))
        deadline = time.time() + (budget_s if budget_s else 10 * self._lie_timeout)
        if self._scheme == "chain":
            products = "1X" + label_key([0] * store.rank)
            for label, key_vec in chars_i:
                products = self._tensor_step(i, products, store.decomp(label, key_vec), deadline)
            return singlet_coefficient(products)
        first, second = self.split(i, chars_i)
        return self.pair(i, self.partial_product(i, first, deadline), self.partial_product(i, second, deadline))

    def multiplicity(self, chars: Chars, budget_s: Optional[float] = None) -> int:
        """Product over the nodes; budget_s bounds one node's chain, as MATCH_TIMEOUT
        bounded one projection of the old pipeline."""
        if not any(chars):
            return 1
        with self._lock:
            memo = self._sig_memo.get(chars)
        if memo is not None:
            return memo
        result = 1
        for i, chars_i in enumerate(chars):
            if chars_i:
                result *= self.node_multiplicity(i, chars_i, budget_s)
                if result == 0:
                    break
        with self._lock:
            self._sig_memo[chars] = result
        return result
