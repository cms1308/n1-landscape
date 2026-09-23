"""Lie data from highest weights.

Everything a node of a product gauge group needs is computed from its Cartan matrix in
LiE's convention, so that the pipeline no longer carries per-species tables:

  * Cartan matrices of the A–G types in LiE's numbering and convention
    A[i][j] = <alpha_i, alpha_j^vee> = 2 (alpha_i, alpha_j) / (alpha_j, alpha_j);
  * the root system by reflection closure, the Weyl dimension formula, the Dynkin index
    mu(R) = dim R (lambda, lambda + 2 rho) / (2 dim G) with long roots of length^2 2
    (mu(fund SU(N)) = 1/2, mu(adj) = h^vee);
  * weight systems (dominant weights with multiplicities from LiE's dom_char, orbits by
    reflection) and the quadratic/cubic weight tensors used by the anomaly validators;
  * the conjugation automorphism of the Dynkin diagram;
  * the routing of the low-rank isomorphic types A1 -> C1, B2 -> C2, D3 -> A3;
  * a LiE runner for dim, dom_char, tensor and sym_tensor (composite groups included).

Weights are tuples of Dynkin labels (coordinates in the fundamental-weight basis).
The E and F types are admitted by the same code but not verified by the package's regression checks.
"""
from __future__ import annotations

import functools
import itertools
import re
import subprocess
from fractions import Fraction as F
from typing import Dict, Iterable, List, Sequence, Tuple

Label = Tuple[int, ...]

# --------------------------------------------------------------------------- #
# Cartan matrices (LiE numbering and convention)
# --------------------------------------------------------------------------- #
def _chain(n: int) -> List[List[int]]:
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = 2
        if i + 1 < n:
            A[i][i + 1] = A[i + 1][i] = -1
    return A


@functools.lru_cache(maxsize=None)
def cartan_matrix(t: str, n: int) -> Tuple[Tuple[int, ...], ...]:
    """A[i][j] = 2 (alpha_i, alpha_j) / (alpha_j, alpha_j), LiE's numbering."""
    if t == "A":
        A = _chain(n)
    elif t == "B":
        if n < 2:
            raise ValueError("B_n needs n >= 2 (B1 is A1)")
        A = _chain(n)
        A[n - 2][n - 1] = -2          # alpha_n short: 2(a_{n-1},a_n)/(a_n,a_n) = -2
    elif t == "C":
        A = _chain(n)
        if n >= 2:
            A[n - 1][n - 2] = -2      # alpha_n long, alpha_{n-1} short
    elif t == "D":
        if n < 3:
            raise ValueError("D_n needs n >= 3")
        A = _chain(n - 1)
        A = [row + [0] for row in A] + [[0] * n]
        A[n - 1][n - 1] = 2
        A[n - 2][n - 1] = A[n - 1][n - 2] = 0
        A[n - 3][n - 1] = A[n - 1][n - 3] = -1   # nodes n-1 and n both attach to n-2
    elif t == "E":
        if n not in (6, 7, 8):
            raise ValueError("E_n needs n in 6..8")
        # Bourbaki: chain 1-3-4-5-...-n, node 2 attached to 4
        A = [[0] * n for _ in range(n)]
        for i in range(n):
            A[i][i] = 2
        edges = [(0, 2), (1, 3), (2, 3)] + [(k, k + 1) for k in range(3, n - 1)]
        for i, j in edges:
            A[i][j] = A[j][i] = -1
    elif t == "F":
        if n != 4:
            raise ValueError("F_n needs n = 4")
        A = _chain(4)
        A[1][2] = -2                 # alpha_3, alpha_4 short
    elif t == "G":
        if n != 2:
            raise ValueError("G_n needs n = 2")
        A = [[2, -1], [-3, 2]]       # alpha_1 short
    else:
        raise ValueError(f"unknown type {t!r}")
    return tuple(tuple(row) for row in A)


@functools.lru_cache(maxsize=None)
def root_data(t: str, n: int):
    """Symmetrized data: d_j = (alpha_j, alpha_j)/2 (long roots -> 1), Gram matrices."""
    A = cartan_matrix(t, n)
    d = [None] * n
    d[0] = F(1)
    # propagate along the diagram: A[i][j] d_j = A[j][i] d_i
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(n):
                if i != j and A[i][j] != 0:
                    if d[i] is not None and d[j] is None:
                        d[j] = d[i] * F(A[j][i], A[i][j])
                        changed = True
    m = max(d)
    d = [x / m for x in d]                       # long roots: d = 1, length^2 2
    S = [[F(A[i][j]) * d[j] for j in range(n)] for i in range(n)]   # (alpha_i, alpha_j)
    import sympy
    Am = sympy.Matrix(A)
    Ainv = Am.inv()
    Sm = sympy.Matrix(S)
    Gw = Ainv * Sm * Ainv.T                     # (omega_i, omega_j)
    G = [[F(int(Gw[i, j].p), int(Gw[i, j].q)) for j in range(n)] for i in range(n)]
    Ainv_f = [[F(int(Ainv[i, j].p), int(Ainv[i, j].q)) for j in range(n)] for i in range(n)]
    return tuple(d), tuple(tuple(r) for r in S), tuple(tuple(r) for r in G), tuple(tuple(r) for r in Ainv_f)


def inner(t: str, n: int, w: Sequence, v: Sequence) -> F:
    """(w, v) for weights in Dynkin-label coordinates."""
    _, _, G, _ = root_data(t, n)
    return sum(F(w[i]) * G[i][j] * F(v[j]) for i in range(n) for j in range(n))


def reflect(A, w: Label, i: int) -> Label:
    """Simple reflection s_i(w) = w - w_i alpha_i, alpha_i = row i of A."""
    wi = w[i]
    if wi == 0:
        return w
    return tuple(w[k] - wi * A[i][k] for k in range(len(w)))


def weyl_orbit(t: str, n: int, w: Label) -> List[Label]:
    A = cartan_matrix(t, n)
    seen = {tuple(w)}
    frontier = [tuple(w)]
    while frontier:
        nxt = []
        for x in frontier:
            for i in range(n):
                y = reflect(A, x, i)
                if y not in seen:
                    seen.add(y)
                    nxt.append(y)
        frontier = nxt
    return sorted(seen)


@functools.lru_cache(maxsize=None)
def positive_roots(t: str, n: int) -> Tuple[Label, ...]:
    A = cartan_matrix(t, n)
    _, _, _, Ainv = root_data(t, n)
    roots = set()
    for i in range(n):
        roots.update(weyl_orbit(t, n, tuple(A[i])))
    pos = []
    for r in roots:
        # alpha-coordinates c = r . A^{-1}
        c = [sum(F(r[i]) * Ainv[i][j] for i in range(n)) for j in range(n)]
        if all(x >= 0 for x in c) and any(x > 0 for x in c):
            pos.append(r)
    return tuple(sorted(pos))


def dim_group(t: str, n: int) -> int:
    return n + 2 * len(positive_roots(t, n))


def dual_coxeter(t: str, n: int) -> F:
    """h^vee = mu(adjoint); adjoint = highest root."""
    return dynkin_index(t, n, highest_root(t, n))


def highest_root(t: str, n: int) -> Label:
    _, _, _, Ainv = root_data(t, n)
    best, besth = None, -1
    for r in positive_roots(t, n):
        h = sum(sum(F(r[i]) * Ainv[i][j] for i in range(n)) for j in range(n))
        if h > besth:
            best, besth = r, h
    return best


def dimension(t: str, n: int, lam: Sequence[int]) -> int:
    """Weyl dimension formula."""
    rho = (1,) * n
    lr = tuple(int(x) + 1 for x in lam)
    num, den = F(1), F(1)
    for a in positive_roots(t, n):
        num *= inner(t, n, lr, a)
        den *= inner(t, n, rho, a)
    val = num / den
    assert val.denominator == 1, (t, n, lam, val)
    return int(val)


def casimir(t: str, n: int, lam: Sequence[int]) -> F:
    """(lambda, lambda + 2 rho) / 2, long roots of length^2 2 (C2(adj) = h^vee)."""
    rho = (1,) * n
    l2 = tuple(int(x) + 2 for x in lam)
    return inner(t, n, lam, l2) / 2


def dynkin_index(t: str, n: int, lam: Sequence[int]) -> F:
    """mu(R) = dim R C2(R) / dim G; mu(fund SU(N)) = 1/2, mu(adj) = h^vee."""
    return F(dimension(t, n, lam)) * casimir(t, n, lam) / dim_group(t, n)


# --------------------------------------------------------------------------- #
# conjugation and routing
# --------------------------------------------------------------------------- #
def conjugate(t: str, n: int, lam: Sequence[int]) -> Label:
    """Dynkin-diagram automorphism giving the dual representation (LiE numbering)."""
    lam = tuple(int(x) for x in lam)
    if t == "A":
        return lam[::-1]
    if t == "D" and n % 2 == 1:
        return lam[:-2] + (lam[-1], lam[-2])
    if t == "E" and n == 6:
        p = {0: 5, 1: 1, 2: 4, 3: 3, 4: 2, 5: 0}
        return tuple(lam[p[i]] for i in range(6))
    return lam


ROUTING = {("A", 1): ("C", 1), ("B", 2): ("C", 2), ("D", 3): ("A", 3)}


def route(t: str, n: int, lam: Sequence[int]) -> Tuple[str, int, Label]:
    """Canonical algebra of a node: A1 -> C1 ([n] -> [n]), B2 -> C2 ([a,b] -> [b,a]),
    D3 -> A3 ([a1,a2,a3] -> [a2,a1,a3]); other types unchanged."""
    lam = tuple(int(x) for x in lam)
    if (t, n) == ("A", 1):
        return "C", 1, lam
    if (t, n) == ("B", 2):
        return "C", 2, (lam[1], lam[0])
    if (t, n) == ("D", 3):
        return "A", 3, (lam[1], lam[0], lam[2])
    return t, n, lam


# --------------------------------------------------------------------------- #
# LiE runner
# --------------------------------------------------------------------------- #
LIE_BIN = "lie"
_POLY_TERM = re.compile(r"([+-]?\d+)X\[([-\d,]*)\]")


def lie_run(code: str, timeout: float = 600) -> str:
    """Run LiE statements; returns the raw stdout after the banner. LiE stops at the
    first error, which is raised."""
    p = subprocess.run(f"{LIE_BIN}", shell=True, input=code + "\n", capture_output=True,
                       text=True, timeout=timeout)
    out = p.stdout
    if "line 1 of file stdin" in out or "(in " in out and "at line" in out:
        raise RuntimeError(f"LiE error for {code!r}: {out.strip()[-400:]}")
    return out


def lie_values(exprs: Sequence[str], group: str, timeout: float = 600) -> List[str]:
    """Evaluate several expressions in one LiE process, separated by a sentinel."""
    code = ";".join(f'print({e});print("@@")' for e in exprs)
    out = lie_run(code, timeout)
    parts = out.replace("\n", "").replace(" ", "").split("@@")
    vals = parts[:len(exprs)]
    # the banner precedes the first value: keep the tail matching a value
    vals[0] = re.sub(r"^.*?(?=[+-]?\d|\[)", "", vals[0], count=1)
    if len(vals) != len(exprs):
        raise RuntimeError(f"LiE returned {len(vals)} values for {len(exprs)} expressions")
    return vals


def parse_poly(text: str) -> Dict[Label, int]:
    """'2X[0,0] +1X[1,1]' -> {(0,0): 2, (1,1): 1}."""
    text = text.replace(" ", "").replace("\n", "").replace("+-", "-")   # LiE writes a negative multiplicity after '+' in virtual characters
    if text in ("0", ""):
        return {}
    out: Dict[Label, int] = {}
    pos = 0
    for m in _POLY_TERM.finditer(text):
        if m.start() != pos:
            raise ValueError(f"unparsed LiE output {text!r} at {pos}")
        pos = m.end()
        lab = tuple(int(x) for x in m.group(2).split(",")) if m.group(2) else ()
        out[lab] = out.get(lab, 0) + int(m.group(1))
    if pos != len(text):
        raise ValueError(f"unparsed LiE output {text!r} tail")
    return {k: v for k, v in out.items() if v}


def lie_group_name(t: str, n: int) -> str:
    """LiE's name of a simple group; LiE has no C1/B1, which are A1 with the same label."""
    if (t, n) in (("C", 1), ("B", 1)):
        return "A1"
    return f"{t}{n}"


def lie_group(nodes: Sequence[Tuple[str, int]]) -> str:
    return "".join(lie_group_name(t, n) for t, n in nodes)


def lie_label(labels: Sequence[Sequence[int]]) -> str:
    return "[" + ",".join(str(int(x)) for lab in labels for x in lab) + "]"


def lie_dim(t: str, n: int, lam: Sequence[int]) -> int:
    (v,) = lie_values([f"dim({lie_label([lam])},{lie_group_name(t, n)})"], f"{t}{n}")
    return int(v)


def lie_dom_char(t: str, n: int, lam: Sequence[int]) -> Dict[Label, int]:
    (v,) = lie_values([f"dom_char({lie_label([lam])},{lie_group_name(t, n)})"], f"{t}{n}")
    return parse_poly(v)


def weight_system(t: str, n: int, lam: Sequence[int]) -> Dict[Label, int]:
    """All weights with multiplicities: LiE's dominant character expanded by Weyl orbits."""
    out: Dict[Label, int] = {}
    for dom, m in lie_dom_char(t, n, lam).items():
        for w in weyl_orbit(t, n, dom):
            out[w] = out.get(w, 0) + m
    return out


def quadratic_tensor(ws: Dict[Label, int], n: int) -> List[List[int]]:
    """M_ij = sum_w m_w w_i w_j (coroot coordinates h: sum m_w (w.h)^2 = h M h)."""
    M = [[0] * n for _ in range(n)]
    for w, m in ws.items():
        for i in range(n):
            if w[i]:
                for j in range(n):
                    M[i][j] += m * w[i] * w[j]
    return M


def cubic_tensor(ws: Dict[Label, int], n: int) -> Dict[Tuple[int, int, int], int]:
    """C_ijk = sum_w m_w w_i w_j w_k for i <= j <= k (the cubic anomaly form)."""
    C: Dict[Tuple[int, int, int], int] = {}
    for w, m in ws.items():
        for i, j, k in itertools.combinations_with_replacement(range(n), 3):
            v = m * w[i] * w[j] * w[k]
            if v:
                C[(i, j, k)] = C.get((i, j, k), 0) + v
    return {k: v for k, v in C.items() if v}


def coroot_gram(t: str, n: int) -> List[List[F]]:
    """(alpha_i^vee, alpha_j^vee) = S_ij / (d_i d_j)."""
    d, S, _, _ = root_data(t, n)
    return [[S[i][j] / (d[i] * d[j]) for j in range(n)] for i in range(n)]


def witten_parity(nodes_data: Iterable[Tuple[F, int]]) -> int:
    """Parity of sum_f 2 mu(R_f) * (product of the other nodes' dimensions) for one
    C-type node; the pairs are (mu, spectator dimension) per field. 1 = anomalous."""
    total = F(0)
    for mu, spect in nodes_data:
        total += 2 * mu * spect
    assert total.denominator == 1, total
    return int(total) % 2
