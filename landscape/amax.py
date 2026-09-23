"""a-maximization on the model.

For a theory of the model — nodes (type, rank), fields with one Dynkin label per node,
superpotential terms as exponent vectors — `solve` returns the verdict class, the
R-charges, a and c, in this order of tests:

  1. two validators on every node after the routing of the low-rank isomorphic types
     (lie.route: A1 -> C1, B2 -> C2, D3 -> A3): the Witten anomaly of a C-type node
     (sum_f 2 mu(R_f,i) prod_{j != i} dim R_f,j odd) and the cubic gauge anomaly (the
     symmetric tensor sum_f prod_{j != i} dim R_f,j sum_w m_w w_a w_b w_c of the weight
     systems, which vanishes identically except for A_n, n >= 2);
  2. the linear constraints, exactly over the rationals: one ABJ constraint per node,
     h^vee_i + sum_f mu_i(R_f) prod_{j != i} dim R_f,j (r_f - 1) = 0, and R(W) = 2 per
     superpotential term; an inconsistent system is `no-r-symmetry`;
  3. the exactly flat directions of the trial a = 3/32 (3 Tr R^3 - Tr R) on the solution
     space (the directional derivative vanishes identically); any is
     `r-charge-undetermined`;
  4. the local maximum with a negative-definite Hessian: Newton iteration at 60 digits,
     polish and certificate at 80 digits, exact rational detection (the conclusive
     early stop restricted to an exactly certified singular negative-semidefinite
     stationary point); none is
     `no-local-maximum`, and `info["conclusive"]` says whether that absence is certified;
  5. `non-positive-r-charge` when some R <= 0 or R >= 2, else `consistent`.

The strict local maximum is unique (the Hessian is affine in the mixing parameters, so
the region where it is negative definite is convex and a is strictly concave there); it
is therefore invariant under every permutation of fields that preserves the trial
function and the constraints, and no identification of symmetric fields is imposed.

Numbers: a value is an exact Fraction when the maximum is rational (or when the
constraints alone fix it), otherwise an mpf at 80 digits; `charges_json` writes "p/q" or
30 significant digits (rounded on 60 digits, half-up).
"""
from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction as F
from typing import Dict, List, Optional, Sequence, Tuple

import mpmath as mp

from . import lie
from .model import Theory, canonical_flavor_basis

VERDICTS = ("consistent", "gauge-anomaly", "witten-anomaly", "no-r-symmetry", "r-charge-undetermined",
            "no-local-maximum", "non-positive-r-charge")

# verdict strings of the legacy codes and of their older versions -> class at the
# a-maximization stage (verdicts assigned after FindCharges map to `consistent`)
_AFTER_FINDCHARGES = ("consistent", "consistent (duplicated)", "inconsistent",
                      "inconsistent (negative central charges)", "too small Rcharge")


def verdict_class(old: str) -> Optional[str]:
    s = " ".join(str(old).split())
    if s in _AFTER_FINDCHARGES:
        return "consistent"
    if s == "non-positive R-charges":
        return "non-positive-r-charge"
    if s == "rcharge undetermined":
        return "r-charge-undetermined"
    if s.endswith("too many superpotentials") or s.endswith("bad superpotentials"):
        return "no-r-symmetry"
    if s.endswith("no negative Hessian when maximizing a"):
        return "no-local-maximum"
    return None          # 'imaginary central charges' and anything unknown


# --------------------------------------------------------------------------- #
# group data per (type, rank, label)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def _dim(t: str, n: int, lab: tuple) -> int:
    return lie.dimension(t, n, lab)


@functools.lru_cache(maxsize=None)
def _mu(t: str, n: int, lab: tuple) -> F:
    return lie.dynkin_index(t, n, lab) if any(lab) else F(0)


@functools.lru_cache(maxsize=None)
def _cubic(t: str, n: int, lab: tuple) -> tuple:
    """Cubic weight tensor of a representation in the routed algebra, as sorted items."""
    if not any(lab):
        return ()
    return tuple(sorted(lie.cubic_tensor(lie.weight_system(t, n, lab), n).items()))


def field_dims(th: Theory) -> List[List[int]]:
    return [[_dim(nd.type, nd.rank, tuple(lab)) for nd, lab in zip(th.nodes, fl)] for fl in th.fields]


def _prod(xs) -> int:
    out = 1
    for x in xs:
        out *= x
    return out


# --------------------------------------------------------------------------- #
# validators
# --------------------------------------------------------------------------- #
def witten_anomalous(th: Theory) -> List[Optional[bool]]:
    """Per node: True/False for a C-type node after routing, None for the others."""
    dims = field_dims(th)
    out: List[Optional[bool]] = []
    for i, nd in enumerate(th.nodes):
        t, n, _ = lie.route(nd.type, nd.rank, nd.zero())
        if t != "C":
            out.append(None)
            continue
        data = []
        for f, fl in enumerate(th.fields):
            if any(fl[i]):
                data.append((_mu(nd.type, nd.rank, tuple(fl[i])), _prod(dims[f][:i] + dims[f][i + 1:])))
        out.append(bool(lie.witten_parity(data)))
    return out


def cubic_anomaly(th: Theory) -> List[Dict[Tuple[int, int, int], int]]:
    """Per node: the nonzero components of sum_f (spectator dimension) x cubic tensor of
    R_f,i, in the routed algebra; empty = no cubic gauge anomaly."""
    dims = field_dims(th)
    out = []
    for i, nd in enumerate(th.nodes):
        tot: Dict[Tuple[int, int, int], int] = {}
        for f, fl in enumerate(th.fields):
            if not any(fl[i]):
                continue
            t, n, lab = lie.route(nd.type, nd.rank, fl[i])
            spect = _prod(dims[f][:i] + dims[f][i + 1:])
            for key, v in _cubic(t, n, lab):
                tot[key] = tot.get(key, 0) + spect * v
        out.append({k: v for k, v in tot.items() if v})
    return out


# --------------------------------------------------------------------------- #
# exact linear algebra over Fractions
# --------------------------------------------------------------------------- #
def constraint_system(th: Theory) -> Tuple[List[List[F]], List[F]]:
    """Rows and right-hand sides of the linear constraints on the R-charges."""
    dims = field_dims(th)
    nf = th.n_fields()
    rows, rhs = [], []
    for i, nd in enumerate(th.nodes):
        row = [_mu(nd.type, nd.rank, tuple(th.fields[f][i])) * _prod(dims[f][:i] + dims[f][i + 1:]) for f in range(nf)]
        rows.append(row)
        rhs.append(sum(row) - lie.dual_coxeter(nd.type, nd.rank))
    for t in th.terms:
        rows.append([F(t.get(f, 0)) for f in range(nf)])
        rhs.append(F(2))
    return rows, rhs


def solve_affine(rows: Sequence[Sequence[F]], rhs: Sequence[F], n: int):
    """Solution set {c + A x} of rows . r = rhs by Gauss-Jordan elimination: returns
    (c, A, free) with A[f][k] the coefficient of the free variable r_{free[k]} in r_f, or
    None when the system is inconsistent."""
    M = [[F(x) for x in row] + [F(b)] for row, b in zip(rows, rhs)]
    pivots: List[int] = []
    r = 0
    for col in range(n):
        piv = next((i for i in range(r, len(M)) if M[i][col] != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = 1 / M[r][col]
        M[r] = [x * inv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][col] != 0:
                k = M[i][col]
                M[i] = [x - k * y for x, y in zip(M[i], M[r])]
        pivots.append(col)
        r += 1
    if any(row[n] != 0 for row in M[r:]):
        return None
    free = [j for j in range(n) if j not in pivots]
    c = [F(0)] * n
    A = [[F(0)] * len(free) for _ in range(n)]
    for k, j in enumerate(free):
        A[j][k] = F(1)
    for i, col in enumerate(pivots):
        c[col] = M[i][n]
        for k, j in enumerate(free):
            A[col][k] = -M[i][j]
    return c, A, free


def _nullspace(rows: Sequence[Sequence[F]], n: int) -> List[List[F]]:
    sol = solve_affine(rows, [F(0)] * len(rows), n)
    _, A, free = sol
    return [[A[f][k] for f in range(n)] for k in range(len(free))]


def flat_directions(A: Sequence[Sequence[F]], cvec: Sequence[F], dims: Sequence[F]) -> List[List[F]]:
    """Basis of {v : d/dv a(c + A x) = 0 identically in x}.  With l_f = c_f - 1 + A_f.x the
    derivative is 3/32 sum_f d_f (A v)_f (9 l_f^2 - 1); its coefficients as a polynomial
    in x are linear in v: sum_f d_f A_fi A_fj A_fk, sum_f d_f A_fi A_fj (c_f - 1) and
    sum_f d_f A_fi (9 (c_f - 1)^2 - 1)."""
    nf, nx = len(A), len(A[0]) if A else 0
    if nx == 0:
        return []
    rows = []
    for j in range(nx):
        for k in range(j, nx):
            rows.append([sum(dims[f] * A[f][i] * A[f][j] * A[f][k] for f in range(nf)) for i in range(nx)])
        rows.append([sum(dims[f] * A[f][i] * A[f][j] * (cvec[f] - 1) for f in range(nf)) for i in range(nx)])
    rows.append([sum(dims[f] * A[f][i] * (9 * (cvec[f] - 1) ** 2 - 1) for f in range(nf)) for i in range(nx)])
    return _nullspace(rows, nx)


# --------------------------------------------------------------------------- #
# trial a and its maximization
# --------------------------------------------------------------------------- #
def _central(rvals, dims, dim_adj):
    """(a, c) from R-charges rvals (Fractions or mpf) and dimensions."""
    if rvals and not isinstance(rvals[0], F):
        rrr = mp.mpf(dim_adj.numerator) / dim_adj.denominator
        r = rrr
        for x, d in zip(rvals, dims):
            dd = mp.mpf(d.numerator) / d.denominator
            rrr += dd * (x - 1) ** 3
            r += dd * (x - 1)
        return mp.mpf(3) * (3 * rrr - r) / 32, (9 * rrr - 5 * r) / 32
    rrr = dim_adj + sum(d * (x - 1) ** 3 for x, d in zip(rvals, dims))
    r = dim_adj + sum(d * (x - 1) for x, d in zip(rvals, dims))
    return F(3) * (3 * rrr - r) / 32, (9 * rrr - 5 * r) / 32


def _is_negdef(H):
    try:
        mp.cholesky(-H)
        return True
    except (ValueError, ZeroDivisionError):
        return False


def _negdef_margin(H):
    """The largest eigenvalue of the symmetric H relative to its norm: negative
    definite means margin < 0.  A margin within 1e-30 of zero at 80 digits is not
    accepted as a strict maximum (a numerical threshold: algebraicity alone gives no
    lower bound on a nonzero eigenvalue); whether such a point excludes a maximum is
    decided separately and exactly (_exact_stationary_class)."""
    ev = mp.eigsy(H)[0]
    lam_max = max(ev)
    scale = max(mp.mpf(1), max(abs(e) for e in ev))
    return lam_max / scale


def _maximize(A, cvec, dims, dps, dps_final, max_iter, seed):
    """Local maximum of a(x) = 3/32 (3 Tr R^3 - Tr R) with R = cvec + A x.
    Returns (x as mp.matrix at dps_final, info) or (None, info)."""
    nf, nx_ = len(A), len(A[0])
    info = {"n_free": nx_, "starts": 0, "iterations": 0, "degenerate": False, "conclusive": False}

    def setup():
        Am = mp.matrix([[mp.mpf(a.numerator) / a.denominator for a in row] for row in A])
        cm = mp.matrix([mp.mpf(c.numerator) / c.denominator for c in cvec])
        dm = [mp.mpf(d.numerator) / d.denominator for d in dims]
        return Am, cm, dm

    def make_funcs(Am, cm, dm):
        k = mp.mpf(3) / 32

        def R_of(x):
            return cm + Am * x

        def a_of(x):
            r = R_of(x)
            return k * sum(dm[i] * (3 * (r[i] - 1) ** 3 - (r[i] - 1)) for i in range(nf))

        def grad(x):
            r = R_of(x)
            wgt = [dm[i] * (9 * (r[i] - 1) ** 2 - 1) for i in range(nf)]
            return mp.matrix([k * sum(wgt[i] * Am[i, j] for i in range(nf)) for j in range(nx_)])

        def hess(x):
            r = R_of(x)
            wgt = [dm[i] * 18 * (r[i] - 1) for i in range(nf)]
            H = mp.matrix(nx_, nx_)
            for j in range(nx_):
                for l in range(j, nx_):
                    s = k * sum(wgt[i] * Am[i, j] * Am[i, l] for i in range(nf))
                    H[j, l] = s
                    H[l, j] = s
            return H
        return a_of, grad, hess

    prng = random.Random(seed)
    # First start: the point where every R-charge is as close as possible to the
    # free-field value 2/3 (weighted least squares, exact) -- all R_f below 1 there
    # in practice, where the Hessian is negative definite; then all free variables
    # at 1/2; then seeded random points.
    starts = [_ls_start(A, cvec, dims, F(2, 3)), [F(1, 2)] * nx_]
    for _ in range(20):
        starts.append([F(prng.randint(20, 120), 100) for _ in range(nx_)])

    found = None
    with mp.workdps(dps):
        Am, cm, dm = setup()
        a_of, grad, hess = make_funcs(Am, cm, dm)
        I = mp.eye(nx_)
        tol_step = mp.mpf(10) ** (-(dps - 8))
        basin = mp.mpf(10) ** (-(dps // 6))  # inside the quadratic basin: full Newton steps
        for x0 in starts:
            info["starts"] += 1
            x = mp.matrix([mp.mpf(v.numerator) / v.denominator for v in x0])
            ok = False
            for it in range(max_iter):
                info["iterations"] += 1
                g = grad(x)
                H = hess(x)
                negdef = _is_negdef(H)
                try:
                    if negdef:
                        d = mp.lu_solve(H, -g)
                    else:
                        lam = max(mp.eigsy(H)[0]) + 1
                        d = mp.lu_solve(H - lam * I, -g)
                except ZeroDivisionError:  # numerically singular: abandon this start
                    break
                dn = mp.norm(d)
                if negdef and dn < tol_step:
                    ok = True
                    break
                if not negdef and mp.norm(g) < tol_step:
                    # a stationary point whose Hessian is not negative definite.  A strict
                    # maximum is excluded only if this Hessian is negative SEMIdefinite:
                    # with H(p) < 0 at a strict maximum p and H(q) <= 0 here, the affine
                    # Hessian is negative definite on the open segment, a is strictly
                    # concave along it, and its derivative cannot vanish at both ends.
                    # A positive eigenvalue resolved above the rounding of this precision
                    # -- however small against the other eigenvalues -- is a minimum or a
                    # saddle and excludes nothing: only this start is abandoned.  An
                    # unresolved sign ends the search only with an exact certificate (a
                    # rational stationary point, exact gradient, Hessian exactly negative
                    # semidefinite AND singular); otherwise the start is abandoned as well.
                    ev = mp.eigsy(H)[0]
                    scale = max(mp.mpf(1), max(abs(e) for e in ev))
                    if max(ev) <= scale * mp.mpf(10) ** (-(dps - 15)) and \
                            _exact_stationary_class(x, A, cvec, dims) == "singular-semidefinite":
                        info["degenerate"] = True
                        info["conclusive"] = True
                        return None, info
                    key = "saddles" if max(ev) > scale * mp.mpf(10) ** (-(dps - 15)) else "unresolved"
                    info[key] = info.get(key, 0) + 1
                    break
                if negdef and dn < basin:
                    x = x + d
                    continue
                if not negdef and dn > 1:  # outside the concave region: bounded steps
                    d = d / dn
                a0 = a_of(x)
                t = mp.mpf(1)
                for _ in range(60):
                    xn = x + t * d
                    if a_of(xn) > a0:
                        break
                    t /= 2
                x = xn
                if mp.norm(x) > 1000:  # the cubic is unbounded: this start runs away
                    break
            if ok and _is_negdef(hess(x)) and mp.norm(grad(x)) < mp.mpf(10) ** (-(dps - 12)):
                found = x
                break
    if found is None:
        return None, info
    with mp.workdps(dps_final):
        Am, cm, dm = setup()
        a_of, grad, hess = make_funcs(Am, cm, dm)
        x = mp.matrix([mp.mpf(found[i]) for i in range(nx_)])
        for _ in range(4):
            try:
                x = x + mp.lu_solve(hess(x), -grad(x))
            except ZeroDivisionError:  # singular at 80 digits: degenerate maximum
                break
        margin = _negdef_margin(hess(x))
        info["gnorm"] = mp.nstr(mp.norm(grad(x)), 5)
        info["hessian_margin"] = mp.nstr(margin, 5)
        if margin > -mp.mpf("1e-30"):
            # below the numerical margin: decided exactly where possible.  An exactly
            # stationary rational point with an exactly negative-definite Hessian IS the
            # strict maximum, however badly scaled; an exactly singular semidefinite one
            # is the conclusive absence; anything else stays uncertified.
            cls = _exact_stationary_class(x, A, cvec, dims)
            if cls == "negative-definite":
                info["negdef"] = True
                info["exact_maximum"] = True
                return x, info
            info["degenerate"] = True
            info["negdef"] = False
            info["conclusive"] = cls == "singular-semidefinite"
            return None, info
        info["negdef"] = True
        return x, info


def _ls_start(A, cvec, dims, r0):
    """x minimizing sum_f dim_f (c_f + A_f x - r0)^2, exactly (normal equations
    over Fractions; A has full column rank since every free variable is the
    R-charge of a field)."""
    nf, nx_ = len(A), len(A[0])
    M = [[sum(dims[i] * A[i][j] * A[i][l] for i in range(nf)) for l in range(nx_)] for j in range(nx_)]
    b = [sum(dims[i] * A[i][j] * (r0 - cvec[i]) for i in range(nf)) for j in range(nx_)]
    # Gaussian elimination
    aug = [row[:] + [b[j]] for j, row in enumerate(M)]
    for i in range(nx_):
        piv = next((r for r in range(i, nx_) if aug[r][i] != 0), None)
        if piv is None:
            return [F(1, 2)] * nx_
        aug[i], aug[piv] = aug[piv], aug[i]
        for r in range(nx_):
            if r != i and aug[r][i] != 0:
                f = aug[r][i] / aug[i][i]
                aug[r] = [aug[r][k] - f * aug[i][k] for k in range(nx_ + 1)]
    return [aug[i][nx_] / aug[i][i] for i in range(nx_)]


def _exact_negdef(H):
    """Sylvester's criterion for -H with exact Fractions."""
    n = len(H)
    M = [[-H[i][j] for j in range(n)] for i in range(n)]
    for k in range(1, n + 1):
        sub = [row[:k] for row in M[:k]]
        # determinant by Gaussian elimination
        det = F(1)
        for i in range(k):
            piv = next((r for r in range(i, k) if sub[r][i] != 0), None)
            if piv is None:
                return False
            if piv != i:
                sub[i], sub[piv] = sub[piv], sub[i]
                det = -det
            det *= sub[i][i]
            for r in range(i + 1, k):
                f = sub[r][i] / sub[i][i]
                if f:
                    sub[r] = [sub[r][j] - f * sub[i][j] for j in range(k)]
        if det <= 0:
            return False
    return True


def _exact_negsemidef(H):
    """-H positive semidefinite, exactly: symmetric elimination over Fractions (a zero
    diagonal entry needs a zero row; a negative one fails)."""
    M = [[-x for x in row] for row in H]
    idx = list(range(len(M)))
    while idx:
        k = idx[0]
        d = M[k][k]
        if d < 0:
            return False
        if d == 0:
            if any(M[k][j] != 0 for j in idx):
                return False
            idx = idx[1:]
            continue
        for i in idx[1:]:
            f = M[i][k] / d
            if f:
                for j in idx[1:]:
                    M[i][j] -= f * M[k][j]
        idx = idx[1:]
    return True


def _exact_stationary_class(x, A, cvec, dims, denom_bound=10 ** 9, tol=mp.mpf("1e-40")):
    """Exact classification of a numerically stationary point: None unless x is within
    tol of a rational point at which the gradient vanishes EXACTLY; then
    "negative-definite" (that point is the strict maximum), "singular-semidefinite"
    (Hessian exactly negative semidefinite with zero determinant: the only conclusive
    certificate that no strict local maximum exists -- a distinct one is excluded by
    concavity along the segment, the point itself by its singular Hessian) or "other"."""
    cand = []
    for i in range(len(x)):
        fr = F(Decimal(mp.nstr(x[i], 50, min_fixed=-200, max_fixed=200))).limit_denominator(denom_bound)
        if abs(x[i] - mp.mpf(fr.numerator) / fr.denominator) > tol:
            return None
        cand.append(fr)
    nf, nx_ = len(A), len(A[0])
    r = [cvec[i] + sum(A[i][j] * cand[j] for j in range(nx_)) for i in range(nf)]
    if any(sum(dims[i] * (9 * (r[i] - 1) ** 2 - 1) * A[i][j] for i in range(nf)) != 0 for j in range(nx_)):
        return None
    H = [[sum(dims[i] * 18 * (r[i] - 1) * A[i][j] * A[i][l] for i in range(nf)) for l in range(nx_)] for j in range(nx_)]
    if _exact_negdef(H):
        return "negative-definite"
    return "singular-semidefinite" if _exact_negsemidef(H) else "other"


def _rational_point(xstar, A, cvec, dims, denom_bound=10 ** 9, tol=mp.mpf("1e-45")):
    """If xstar is within tol of a rational point that is EXACTLY a critical point
    with a negative-definite Hessian, return it (list of Fractions), else None."""
    cand = []
    for i in range(len(xstar)):
        fr = F(Decimal(mp.nstr(xstar[i], 55, min_fixed=-200, max_fixed=200))).limit_denominator(denom_bound)
        if abs(xstar[i] - mp.mpf(fr.numerator) / fr.denominator) > tol:
            return None
        cand.append(fr)
    nf, nx_ = len(A), len(A[0])
    r = [cvec[i] + sum(A[i][j] * cand[j] for j in range(nx_)) for i in range(nf)]
    for j in range(nx_):
        g = sum(dims[i] * (9 * (r[i] - 1) ** 2 - 1) * A[i][j] for i in range(nf))
        if g != 0:
            return None
    H = [[sum(dims[i] * 18 * (r[i] - 1) * A[i][j] * A[i][l] for i in range(nf)) for l in range(nx_)] for j in range(nx_)]
    if not _exact_negdef(H):
        return None
    return cand


# --------------------------------------------------------------------------- #
# number printing: 30 significant digits; an exact rational with a terminating decimal
# expansion as its shortest exact decimal
# --------------------------------------------------------------------------- #
def _round30(s):
    try:
        d = Decimal(s)
    except Exception:
        return s
    if d.is_zero():
        return "0"
    with localcontext() as ctx:
        ctx.prec = 30
        ctx.rounding = ROUND_HALF_UP
        d = +d
    return format(d, 'f')


def _terminating(fr: F) -> bool:
    den = fr.denominator
    for q in (2, 5):
        while den % q == 0:
            den //= q
    return den == 1


def _frac_digits(fr: F) -> str:
    if _terminating(fr):
        with localcontext() as ctx:
            ctx.prec = 400
            s = format(Decimal(fr.numerator) / Decimal(fr.denominator), 'f')
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    with localcontext() as ctx:
        ctx.prec = 60
        return format(Decimal(fr.numerator) / Decimal(fr.denominator), 'f')


def _rational_near(v, tol=mp.mpf("1e-50"), denom_bound=10 ** 12):
    fr = F(Decimal(mp.nstr(v, 55, min_fixed=-200, max_fixed=200))).limit_denominator(denom_bound)
    if _terminating(fr) and abs(v - mp.mpf(fr.numerator) / fr.denominator) < tol:
        return fr
    return None


def decimal_string(v) -> str:
    """30 significant digits, or the shortest exact decimal of a terminating rational."""
    if isinstance(v, F):
        return _round30(_frac_digits(v))
    with mp.workdps(80):
        fr = _rational_near(v)
        if fr is not None:
            return _round30(_frac_digits(fr))
        return _round30(mp.nstr(v, 60, min_fixed=-200, max_fixed=200))


def value_string(v) -> str:
    """Record form (D8): "p/q" for an exact rational, 30 significant digits otherwise."""
    if isinstance(v, F):
        return f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)
    return decimal_string(v)


# --------------------------------------------------------------------------- #
# result
# --------------------------------------------------------------------------- #
@dataclass
class AmaxResult:
    verdict: str
    R: Optional[list] = None            # per field: Fraction (exact) or mpf (80 digits)
    a: object = None
    c: object = None
    rational: bool = False
    witten: List[Optional[bool]] = field(default_factory=list)
    cubic: List[dict] = field(default_factory=list)
    flat: List[List[F]] = field(default_factory=list)     # flat directions, as vectors of field R-charges
    info: dict = field(default_factory=dict)

    def charges_json(self) -> dict:
        if self.R is None:
            return {"R": None, "a": None, "c": None, "rational": False}
        return {"R": [value_string(r) for r in self.R], "a": value_string(self.a), "c": value_string(self.c),
                "rational": self.rational}


def solve(th: Theory, dps: int = 60, dps_final: int = 80, max_iter: int = 200, seed: int = 20260914) -> AmaxResult:
    witten = witten_anomalous(th)
    cubic = cubic_anomaly(th)
    res = AmaxResult("consistent", witten=witten, cubic=cubic, info={"n_free": 0, "iterations": 0, "starts": 0})
    if any(cubic):
        res.verdict = "gauge-anomaly"
        return res
    if any(w for w in witten):
        res.verdict = "witten-anomaly"
        return res
    nf = th.n_fields()
    rows, rhs = constraint_system(th)
    sol = solve_affine(rows, rhs, nf)
    if sol is None:
        res.verdict = "no-r-symmetry"
        return res
    cvec, A, free = sol
    dims = [F(_prod(d)) for d in field_dims(th)]
    dim_adj = F(sum(lie.dim_group(nd.type, nd.rank) for nd in th.nodes))
    res.info["n_free"] = len(free)
    if free:
        flat = flat_directions(A, cvec, dims)
        if flat:
            res.verdict = "r-charge-undetermined"
            res.flat = [[sum(A[f][k] * v[k] for k in range(len(free))) for f in range(nf)] for v in flat]
            return res
        xstar, minfo = _maximize(A, cvec, dims, dps, dps_final, max_iter, seed)
        res.info.update(minfo)
        if xstar is None:
            res.verdict = "no-local-maximum"
            return res
        with mp.workdps(dps_final):
            rat = _rational_point(xstar, A, cvec, dims)
            if rat is not None:
                res.rational = True
                rvals = [cvec[f] + sum(A[f][k] * rat[k] for k in range(len(free))) for f in range(nf)]
                res.R = rvals
            else:
                rvals = [mp.mpf(cvec[f].numerator) / cvec[f].denominator
                         + sum(mp.mpf(A[f][k].numerator) / A[f][k].denominator * xstar[k] for k in range(len(free)))
                         for f in range(nf)]
                # an R-charge fixed by the constraints alone is exact by construction
                res.R = [cvec[f] if not any(A[f]) else rvals[f] for f in range(nf)]
            res.a, res.c = _central(rvals, dims, dim_adj)
    else:
        res.rational = True
        res.R = list(cvec)
        res.a, res.c = _central(res.R, dims, dim_adj)
    lo, hi = r_range(res.R, dps_final)
    if lo <= 0 or hi >= 2:
        res.verdict = "non-positive-r-charge"
    return res


def r_range(R: Sequence, dps: int = 80):
    """(min, max) of the R-charges; exact Fractions when all are, else mpf (an exact 0 or 2
    converts exactly, so the comparisons with 0 and 2 are not affected)."""
    if all(isinstance(r, F) for r in R):
        return min(R), max(R)
    with mp.workdps(dps):
        vals = [mp.mpf(r.numerator) / r.denominator if isinstance(r, F) else r for r in R]
        return min(vals), max(vals)


def flavor_basis(th: Theory, ambiguous: bool = False) -> List[List[int]]:
    """The flavor lattice (saturated integer kernel, Hermite normal form) in the canonical
    field order, columns carried back to the source order; source order for a theory
    without a canonical identity.  The single copy is model.canonical_flavor_basis."""
    return canonical_flavor_basis(th, ambiguous=ambiguous)
