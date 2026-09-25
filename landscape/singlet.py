"""FORM-output parser and gauge-singlet projection for a product of simple groups.

The FORM program (form.py) writes one character function per
(node, Dynkin label), named by `char_symbol`: `C0L1x0(k)` is psi^k of the C2
representation [1,0] at node 0.  Fields with the same label at a node share the symbol,
as the species symbols of the old program were shared (`q(k)` for every fundamental);
a field in a product representation contributes the same Adams index k at every node
it charges, because chi_{R_1 x R_2}(x^k) = psi^k(chi_{R_1}) psi^k(chi_{R_2}).

A parsed term carries, per node, the Adams multiplicity key of every label present;
the singlet multiplicity of the term is the product over the nodes of the singlet
multiplicity of that node's character product (the singlet of R_1 x ... x R_k of a
product group is the product of the per-node singlets).  One label is one store
lookup.  Several labels are split into two parts at label granularity, balanced by
the product of the term counts of their stored decompositions; each part is
decomposed through the node's own tensor-step cache (one label needs no LiE call, a
part of several labels a chain of tensor steps starting from its first decomposition),
and the two parts are contracted by the pairing of dual representations, sum over
lambda of a_lambda b_{lambda*} with lambda* the conjugate highest weight -- the
singlet multiplicity of the product of two virtual characters, so the complete
product is never tensor-decomposed (scheme "split", the default).  The left-to-right
chain through the complete product (scheme "chain") is kept as the comparison
baseline.  The old species-keyed FORM outputs are read back through
`chars_from_species` (species merged into their label) for the single-node
regressions.

Exact semantics inherited from the original code: the t-exponent decoding
t^{d0} s^{d1} r^{d2} -> d0/500 + d1/2.5e6 + d2/1.25e10 quantized to 0.001 (ROUND_HALF_UP);
Adams key entry k-1 = multiplicity of Adams_k, zero-padded to total degree; the
singlet of a single decomposition read from the first term of a LiE character string;
the LiE tensor invocation, [53:] banner slice, 'line'-triggered maxobjects retry and
the validity gate before caching.
"""
from __future__ import annotations

import functools
import os
import re
import subprocess
import threading
import time
from decimal import Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from . import lie
from .store import LabelStore, run_lie, singlet_coefficient, label_key

Label = Tuple[int, ...]
KeyVec = Tuple[int, ...]
NodeChars = Tuple[Tuple[Label, KeyVec], ...]      # sorted by label
Chars = Tuple[NodeChars, ...]                     # one entry per node

_SYMBOL_RE = re.compile(r"^C(\d+)L(\d+(?:x\d+)*)$")
_MILLI_Q = Decimal("0.001")


def char_symbol(node: int, label: Sequence[int]) -> str:
    """FORM function name of psi^k(chi_label) at a node: C<node>L<l1>x<l2>x..."""
    return f"C{node}L" + "x".join(str(int(x)) for x in label)


def parse_symbol(name: str) -> Optional[Tuple[int, Label]]:
    m = _SYMBOL_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), tuple(int(x) for x in m.group(2).split("x"))


def key_vec_of(adams: Dict[int, int]) -> KeyVec:
    """{k: m_k} -> [m_1, ..., m_N] with N = sum k m_k."""
    adams = {k: m for k, m in adams.items() if m}
    if not adams:
        return ()
    length = sum(k * m for k, m in adams.items())
    vec = [0] * length
    for k, m in adams.items():
        vec[k - 1] = m
    return tuple(vec)


def adams_of(key_vec: Sequence[int]) -> Dict[int, int]:
    return {k + 1: m for k, m in enumerate(key_vec) if m}


def node_chars(per_label: Dict[Label, Dict[int, int]]) -> NodeChars:
    return tuple(sorted((lab, key_vec_of(ad)) for lab, ad in per_label.items() if any(ad.values())))


class Term(NamedTuple):
    """One '+'-separated FORM output term, exactly decoded."""
    coeff: Fraction
    milli: int                              # physical t-exponent * 1000 (rounded)
    ypow: int
    fug: Tuple[Tuple[str, int], ...]        # sorted ((symbol, exponent), ...): field markers, flavor fugacities
    chars: Chars


def _milli_exponent(tpow: int, spow: int, rpow: int) -> int:
    with localcontext() as ctx:
        ctx.prec = 50
        p = (Decimal(tpow) / Decimal(500) + Decimal(spow) / Decimal(2500000)
             + Decimal(rpow) / Decimal(12500000000))
        return int(p.quantize(_MILLI_Q, rounding=ROUND_HALF_UP) * 1000)


def parse_term(term: str, n_nodes: int) -> Term:
    """Parse one FORM output term (no spaces, '*'-separated factors)."""
    coeff = Fraction(1)
    tpow = spow = rpow = ypow = 0
    fug: Dict[str, int] = {}
    per_node: List[Dict[Label, Dict[int, int]]] = [dict() for _ in range(n_nodes)]
    for factor in term.split("*"):
        if factor.startswith("d(") and factor.endswith(")"):
            n_str, m_str = factor[2:-1].split(",")
            coeff *= Fraction(int(n_str), int(m_str))
            continue
        base, caret, exp_str = factor.partition("^")
        exp = int(exp_str.strip("()")) if caret else 1
        if "(" in base:
            name, _, arg = base[:-1].partition("(")
            sym = parse_symbol(name)
            if sym is None:
                raise ValueError(f"unknown character function {factor!r}")
            node, label = sym
            if node >= n_nodes:
                raise ValueError(f"node {node} of {factor!r} beyond {n_nodes} nodes")
            per = per_node[node].setdefault(label, {})
            per[int(arg)] = per.get(int(arg), 0) + exp
        elif base == "t":
            tpow += exp
        elif base == "s":
            spow += exp
        elif base == "r":
            rpow += exp
        elif base == "y":
            ypow += exp
        elif base.isdigit() or (base.startswith("-") and base[1:].isdigit()):
            coeff *= Fraction(int(base)) ** exp
        elif base:
            fug[base] = fug.get(base, 0) + exp
        else:
            raise ValueError(f"empty factor in term {term!r}")
    return Term(coeff=coeff, milli=_milli_exponent(tpow, spow, rpow), ypow=ypow,
                fug=tuple(sorted((k, v) for k, v in fug.items() if v != 0)),
                chars=tuple(node_chars(p) for p in per_node))


try:                                            # the optional native extension (landscape_native, Rust)
    import landscape_native as _native
except ImportError:                             # the pure-Python path
    _native = None

def _switch(var: str, default: str = "1") -> bool:
    return os.environ.get(var, default).strip().lower() not in ("0", "off", "no", "")


# The native parser is used when the extension imports and LANDSCAPE_NATIVE is not 0/off/no;
# the combined native pass (parse, singlet lookups and the field-resolved aggregation in one
# call) when in addition LANDSCAPE_NATIVE_EXPAND is not off.  Both attributes can be set at
# runtime (the interleaved comparisons of the paths).
NATIVE = _native is not None and _switch("LANDSCAPE_NATIVE")
NATIVE_EXPAND = NATIVE and hasattr(_native, "expand") and _switch("LANDSCAPE_NATIVE_EXPAND")
# The native expansion engine in place of FORM (LANDSCAPE_NATIVE_FORM; on by default since its
# gate passed: the polynomial equal to FORM's on the replay set, the corpus and the sample, the
# expansion stages 5.4x faster): the exponential of the explicit itotal computed by the
# extension, FORM the reference and the fallback (an overflow of the 128-bit coefficients, a
# missing extension or the switch off run FORM as before).
NATIVE_FORM = NATIVE and hasattr(_native, "expand_series") and _switch("LANDSCAPE_NATIVE_FORM")
# The native post-processing pass (LANDSCAPE_NATIVE_POST; on by default when the extension provides
# FieldResolvedRows): the engine and the combined pass keep their rows in the extension, and
# post.index / post.decouple / post.prefilter and model.project read the reduced index, its flavor
# projection, the net index and the physical index from one native pass over them; the Python
# functions are the reference and the fallback (a plain list of FieldResolvedTerms, or an overflow).
NATIVE_POST = NATIVE and hasattr(_native, "FieldResolvedRows") and _switch("LANDSCAPE_NATIVE_POST")
# The refinements of the expansion engine (step 44), each gated separately and on by default only
# where adopted: the flavor projection above t^6 (LANDSCAPE_NATIVE_FLAVOR, adopted -- on by default;
# the rows above t^6 are then flavor-refined, which only the native post pass consumes, so it needs
# NATIVE_POST), the 64-bit coefficient path (LANDSCAPE_NATIVE_COEF64, declined: 4.5 % on the engine
# stage, off by default) and the exact-milli truncation (LANDSCAPE_NATIVE_EXACT, declined: no gain, off
# by default).
NATIVE_FLAVOR = NATIVE_POST and NATIVE_FORM and _switch("LANDSCAPE_NATIVE_FLAVOR")
NATIVE_COEF64 = NATIVE_FORM and _switch("LANDSCAPE_NATIVE_COEF64", "0")
NATIVE_EXACT = NATIVE_FORM and _switch("LANDSCAPE_NATIVE_EXACT", "0")


def native_available() -> bool:
    return _native is not None


def native_expand_available() -> bool:
    return _native is not None and hasattr(_native, "expand")


def native_form_available() -> bool:
    return _native is not None and hasattr(_native, "expand_series")


def native_post_available() -> bool:
    return _native is not None and hasattr(_native, "FieldResolvedRows")


class NativeOverflow(ArithmeticError):
    """The series engine's 128-bit coefficients overflowed; the caller falls back to FORM."""


class NativeCapacity(NativeOverflow):
    """The series engine's expansion exceeds its monomial cap (LANDSCAPE_NATIVE_MAX_TERMS, default
    8,000,000 per power and in total: a theory of 25 fields at expansion order 38 reached 10 M
    monomials at k = 5 and 20 GB at k = 6); the caller falls back to FORM, which sorts on disk."""


def expand_series_native(series, projector: "ProductProjector", timeout: Optional[float] = None,
                         budget_s: Optional[float] = None, monomials: bool = False, native_rows: bool = False,
                         basis=None, exact: bool = False, coef64: bool = False):
    """The native expansion of a `form.Series`: the rows of the field-resolved expansion
    (as expand_native; a FieldResolvedRows object when native_rows is set -- with `basis`, the
    rows above t^6 flavor-refined by it, the object only), or with monomials=True the
    polynomial itself [(numerator, denominator, exponents)] for the comparison with FORM's
    output.  `exact` and `coef64` are the engine refinements of step 44.  NativeOverflow on an
    overflow; subprocess.TimeoutExpired past the timeout."""
    try:
        return _native.expand_series(series.terms, series.t_limit, series.max_order, series.n_fields,
                                     [(node, list(label), k) for node, label, k in series.slots], series.n_nodes,
                                     series.t_order, lambda chars: projector.multiplicity(chars, budget_s), timeout, monomials,
                                     native_rows, basis, exact, coef64)
    except ValueError as e:
        if str(e).startswith("capacity"):
            raise NativeCapacity(str(e)) from e
        if "overflow" in str(e):
            raise NativeOverflow(str(e)) from e
        raise


def expand_native(text: str, n_nodes: int, n_fields: int, t_order: int, projector: "ProductProjector",
                  budget_s: Optional[float] = None, native_rows: bool = False):
    """The combined native pass: [(numerator, denominator, milli, ypow, markers)] of the
    field-resolved expansion through t^t_order (a FieldResolvedRows object when native_rows is
    set), the singlet multiplicity of every distinct character product from
    `projector.multiplicity` (its memo, the stores and LiE as usual; products occurring only in
    terms above the truncation are not looked up)."""
    return _native.expand(text, n_nodes, n_fields, t_order, lambda chars: projector.multiplicity(chars, budget_s), native_rows)


def parse_terms(text: str, n_nodes: int) -> List[Term]:
    """Every '+'-separated term of a FORM output, parsed: the native parser when the
    extension is importable and NATIVE is set, `parse_term` term by term otherwise; the
    same Term objects (coefficient a Fraction, the tuples as parse_term builds them) either
    way."""
    if NATIVE and _native is not None:
        try:
            return _native.parse_terms(text, n_nodes, Term, Fraction)
        except ValueError as e:
            if "overflow" not in str(e):
                raise
            # a coefficient beyond 128 bits (expansion orders above about 20; found on the
            # June-2026 Sp2nf5 landscape at order 25): the Python parser, arbitrary precision
    return [parse_term(t, n_nodes) for t in text.split("+") if t]


def parse_form_file(path: Path, n_nodes: int) -> List[Term]:
    return parse_terms(Path(path).read_text(), n_nodes)


def chars_from_species(old_chars: Sequence[Tuple[str, Sequence[int]]], n_nodes: int, node: int,
                       label_of: Dict[str, Sequence[int]]) -> Chars:
    """Old single-group chars ((species, key_vec), ...) -> product chars with the species
    merged into their Dynkin label at `node` (Adams multiplicities added)."""
    per: Dict[Label, Dict[int, int]] = {}
    for species, kv in old_chars:
        lab = tuple(int(x) for x in label_of[species])
        ad = per.setdefault(lab, {})
        for k, m in adams_of(kv).items():
            ad[k] = ad.get(k, 0) + m
    out: List[NodeChars] = [() for _ in range(n_nodes)]
    out[node] = node_chars(per)
    return tuple(out)


SCHEMES = ("split", "chain")


class ProductProjector:
    """Gauge-singlet projection of parsed terms for a product of simple groups: one
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
                raise self._timeout_error("lie subprocess timeout", self._lie_timeout)
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


def project_terms(terms: Sequence[Term], projector: ProductProjector, core: int = 1,
                  match_timeout: Optional[float] = None) -> List[Tuple[Term, int]]:
    """(term, singlet multiplicity) for every parsed term; chains on a thread pool."""
    unique = {t.chars for t in terms if any(t.chars)}
    multi = [c for c in unique if any(len(ci) >= 2 for ci in c)]
    if multi and core > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=core) as pool:
            for fut in [pool.submit(projector.multiplicity, c, match_timeout) for c in multi]:
                fut.result()
    return [(t, projector.multiplicity(t.chars, match_timeout)) for t in terms]


def process_form_output(form_path: Path, projector: ProductProjector, core: int = 1,
                        match_timeout: Optional[float] = None) -> List[Tuple[Term, int]]:
    return project_terms(parse_form_file(form_path, projector.n_nodes), projector, core, match_timeout)
