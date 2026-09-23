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
product group is the product of the per-node singlets), each computed as one store lookup for a single label, otherwise a
chain of LiE tensor steps through the node's own tensor-step cache.  The old
species-keyed FORM outputs are read back through `chars_from_species` (species merged
into their label) for the single-node regressions.

Exact semantics inherited from the original code: the t-exponent decoding
t^{d0} s^{d1} r^{d2} -> d0/500 + d1/2.5e6 + d2/1.25e10 quantized to 0.001 (ROUND_HALF_UP);
Adams key entry k-1 = multiplicity of Adams_k, zero-padded to total degree; the
singlet read from the first term of a LiE character string; the LiE tensor invocation,
[53:] banner slice, 'line'-triggered maxobjects retry and the validity gate before
caching.
"""
from __future__ import annotations

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


def parse_form_file(path: Path, n_nodes: int) -> List[Term]:
    text = Path(path).read_text()
    return [parse_term(t, n_nodes) for t in text.split("+") if t]


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


class ProductProjector:
    """Gauge-singlet projection of parsed terms for a product of simple groups: one
    LabelStore per node (character decompositions and tensor-step cache), the singlet
    multiplicity = product over nodes."""

    def __init__(self, stores: Sequence[LabelStore], lie_timeout: float = 180.0,
                 lie_runner=None, timeout_error=subprocess.TimeoutExpired):
        self._stores = list(stores)
        self._lie_timeout = lie_timeout
        self._lie_runner = lie_runner if lie_runner is not None else run_lie
        self._timeout_error = timeout_error
        self._step_memo: Dict[str, str] = {}
        self._sig_memo: Dict[tuple, int] = {}
        self._lock = threading.Lock()
        self.lie_calls = 0

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

    def node_multiplicity(self, i: int, chars_i: NodeChars, budget_s: Optional[float] = None) -> int:
        """Singlet multiplicity of one node's character product."""
        if not chars_i:
            return 1
        store = self._stores[i]
        if len(chars_i) == 1:
            label, key_vec = chars_i[0]
            return singlet_coefficient(store.decomp(label, key_vec))
        deadline = time.time() + (budget_s if budget_s else 10 * self._lie_timeout)
        products = "1X" + label_key([0] * store.rank)
        for label, key_vec in chars_i:
            products = self._tensor_step(i, products, store.decomp(label, key_vec), deadline)
        return singlet_coefficient(products)

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
