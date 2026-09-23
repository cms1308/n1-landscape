"""Converters from the legacy formats into the model, with the maps each conversion
retains (name map, position map, flavor-basis transformation) and the classifier of the
semantic transformations between the old conventions and the model.

Sources:
  * records of the original single-group landscape code: species names of one simple
    group, `q1*qb1*phi1` strings, InputForm index strings, per-field `global` vectors,
    charges as strings (30 significant digits or the shortest exact decimal) with an
    optional `rational` dictionary;
  * the 2023 product-group corpus (`quiver.py`): representation pairs by position,
    fields `O1, O2, ...`, machine-precision floats, InputForm index strings;
  * the 2026 `quiver.py` input format: named fields with a multiplicity (only
    multiplicity 1 is supported).

Legacy index fields: `fullindex` = the flavor-refined reduced index, `index` = the
unrefined reduced index, `shortindex` = the unrefined reduced index with the decoupled
sector removed (single-group records only; not converted).  The model stores the full
refined index (constant term 1), recovered exactly from the reduced one by series
division through the truncation order.

The species -> Dynkin-label registry of the legacy formats is `data/species_labels.json`."""
from __future__ import annotations

import ast
import json
import re
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction as F
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import lie, notation
from .model import (Theory, Node, PhysicalTerm, flavor_lattice, solve_integer_transformation, constraint_rows,
                    inverse_transform_flavor, transform_flavor, index_array, make_record, left_inverse)

DATA_DIR = Path(__file__).resolve().parent / "data"

NAME_ORDER = ['X', 'M', 'q', 'qb', 'phi', 'S', 'Sb', 'A', 'Ab', 'U', 'Ub', 'V', 'Vb', 'W', 'Wb']
TRIVIAL = {"X", "M", "1", "O"}
_FIELD = re.compile(r"^([A-Za-z]+)(\d+)$")

# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
_REGISTRY: Dict[str, Dict[str, tuple]] = {}


def registry() -> Dict[str, Dict[str, tuple]]:
    if not _REGISTRY:
        d = json.loads((DATA_DIR / "species_labels.json").read_text())
        for key, lab in d.items():
            g, sp = key.split("/")
            _REGISTRY.setdefault(g, {})[sp] = tuple(int(x) for x in lab.strip("[]").split(","))
    return _REGISTRY


def species_label(node: Node, species: str) -> tuple:
    if species in TRIVIAL:
        return node.zero()
    g = f"{node.type}{node.rank}"
    try:
        return registry()[g][species]
    except KeyError:
        raise KeyError(f"species {species!r} not registered for {g}")


# --------------------------------------------------------------------------- #
# strings
# --------------------------------------------------------------------------- #
_MONO_TOK = re.compile(r"\s*(?:(\d+)|([A-Za-z]+\d*|-)|(\^)|(\*)|(/)|(\()|(\)))")


def parse_monomial(text: str, pos: Dict[str, int]) -> Tuple[Dict[int, int], F]:
    """Monomial string -> ({field index: power}, coefficient).  Accepts Mathematica
    InputForm ('2*q1*q2^2'), the 2023 corpus's F-term-substituted forms ('O1^(-2)',
    '(O1*O2)/O3^2', '(O1*O2)/O3^2*O12') and the package notation ('x4^-5' style powers).
    Division applies to the next item only, as in Mathematica's InputForm."""
    toks = []
    i = 0
    s = text.strip()
    while i < len(s):
        m = _MONO_TOK.match(s, i)
        if not m or m.end() == i:
            raise ValueError(f"bad monomial {text!r} at {i}")
        toks.append(next((k, v) for k, v in enumerate(m.groups()) if v is not None))
        i = m.end()
    out: Dict[int, int] = {}
    coeff = F(1)
    p = 0

    def exponent():
        nonlocal p
        if p < len(toks) and toks[p][0] == 2:           # '^'
            p += 1
            opened = False
            if p < len(toks) and toks[p][0] == 5:      # '^(' ... ')'
                opened = True; p += 1
            sign = 1
            if p < len(toks) and toks[p][0] == 1 and toks[p][1] == "-":
                sign = -1; p += 1
            assert p < len(toks) and toks[p][0] == 0, f"bad exponent in {text!r}"
            e = int(toks[p][1]) * sign; p += 1
            if opened:
                assert p < len(toks) and toks[p][0] == 6, f"unbalanced exponent parentheses in {text!r}"
                p += 1
            return e
        return 1

    def item(sign):
        nonlocal p, coeff
        if toks[p][0] == 5:                             # '(' product ')'
            p += 1
            product(sign)
            assert toks[p][0] == 6, f"unbalanced parentheses in {text!r}"
            p += 1
            if p < len(toks) and toks[p][0] == 2:
                e = exponent()                          # (...)^n unsupported beyond 1
                assert e == 1, f"power of a group in {text!r}"
        elif toks[p][0] == 0:                           # number
            c = F(int(toks[p][1])); p += 1
            coeff = coeff * c if sign > 0 else coeff / c
        elif toks[p][0] == 1:                           # name
            name = toks[p][1]; p += 1
            e = exponent()
            if name not in pos:
                raise KeyError(f"unknown field {name!r} in {text!r}")
            out[pos[name]] = out.get(pos[name], 0) + sign * e
        else:
            raise ValueError(f"unexpected token in {text!r}")

    def product(sign):
        nonlocal p
        item(sign)
        while p < len(toks) and toks[p][0] in (3, 4):
            op = toks[p][0]; p += 1
            item(sign if op == 3 else -sign)

    # a leading '-' sign is not produced by the sources; '-' inside exponents handled above
    product(1)
    assert p == len(toks), f"trailing tokens in {text!r}"
    return {f: e for f, e in out.items() if e != 0}, coeff


def monomial_string(term: Dict[int, int], names: Sequence[str], coeff: int = 1) -> str:
    """Mathematica InputForm order (notation.mma_key): case-insensitive, lowercase first."""
    from .notation import mma_key
    body = "*".join(f"{names[f]}^{p}" if p != 1 else names[f]
                    for f, p in sorted(term.items(), key=lambda kv: mma_key(names[kv[0]])))
    if coeff == 1:
        return body
    return f"{coeff}*{body}"


_FLOAT_EXP = re.compile(r"t\^(\d+\.\d+)")


def normalize_exponents(text: str) -> Tuple[str, bool]:
    """Round floating-point t exponents of the 2023 corpus (t^8.969999999999999) to the
    1/1000 grid, half up; returns (text, changed)."""
    from decimal import ROUND_HALF_UP
    def fix(m):
        d = Decimal(m.group(1)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        return "t^" + format(d.normalize(), "f")
    new = _FLOAT_EXP.sub(fix, text)
    return new, new != text


def parse_index(text: str, rank_old: int) -> List[PhysicalTerm]:
    """InputForm or package-notation string -> terms with flavor exponents in the old basis g_1..g_k."""
    out = []
    text, _ = normalize_exponents(text)
    for (t_milli, factors), coeff in notation.parse(text).items():
        fl = [0] * rank_old
        y = 0
        for sym, p in factors:
            if sym[0] == 0:
                assert 1 <= sym[1] <= rank_old, f"fugacity g{sym[1]} beyond rank {rank_old}"
                fl[sym[1] - 1] += p
            elif sym[0] == 1:
                y += p
            else:
                raise ValueError(f"field symbol in index string: {sym}")
        out.append(PhysicalTerm(F(coeff), t_milli, y, tuple(fl)))
    return out


def index_string(terms: Iterable[PhysicalTerm]) -> str:
    """Package notation of a physical index (fugacities x_i in the basis of the terms)."""
    poly: notation.Poly = {}
    for t in terms:
        factors = [((0, i + 1), e) for i, e in enumerate(t.flavor) if e] + ([((1,), t.ypow)] if t.ypow else [])
        key = (t.milli, tuple(sorted(factors)))
        poly[key] = poly.get(key, F(0)) + t.coeff
    return notation.format_poly({k: v for k, v in poly.items() if v})


def index_equal(a: Iterable[PhysicalTerm], b: Iterable[PhysicalTerm]) -> bool:
    def norm(ts):
        acc: Dict[tuple, F] = {}
        for t in ts:
            acc[(t.milli, t.ypow, tuple(t.flavor))] = acc.get((t.milli, t.ypow, tuple(t.flavor)), F(0)) + t.coeff
        return {k: v for k, v in acc.items() if v}
    return norm(a) == norm(b)


# --------------------------------------------------------------------------- #
# reduced <-> full index (exact series division through the truncation order)
# --------------------------------------------------------------------------- #
def reduce_index(full: Iterable[PhysicalTerm], t_order: int) -> List[PhysicalTerm]:
    """(1 - t^3 y)(1 - t^3/y)(I - 1), truncated at t^t_order."""
    kernel = ((0, 0, 1), (3000, 1, -1), (3000, -1, -1), (6000, 0, 1))
    acc: Dict[tuple, F] = {}
    for t in full:
        if t.milli == 0 and t.ypow == 0 and not any(t.flavor):
            c = t.coeff - 1
            if c == 0:
                continue
            t = PhysicalTerm(c, 0, 0, t.flavor)
        for dm, dy, s in kernel:
            key = (t.milli + dm, t.ypow + dy, tuple(t.flavor))
            if key[0] <= 1000 * t_order:
                acc[key] = acc.get(key, F(0)) + s * t.coeff
    return [PhysicalTerm(c, k[0], k[1], k[2]) for k, c in acc.items() if c]


def full_index(reduced: Iterable[PhysicalTerm], t_order: int, rank: int) -> List[PhysicalTerm]:
    """I = 1 + I_red * sum_{m,n>=0} t^{3(m+n)} y^{m-n}, truncated at t^t_order."""
    acc: Dict[tuple, F] = {(0, 0, (0,) * rank): F(1)}
    lim = 1000 * t_order
    for t in reduced:
        m = 0
        while t.milli + 3000 * m <= lim:
            n = 0
            while t.milli + 3000 * (m + n) <= lim:
                key = (t.milli + 3000 * (m + n), t.ypow + m - n, tuple(t.flavor))
                acc[key] = acc.get(key, F(0)) + t.coeff
                n += 1
            m += 1
    return [PhysicalTerm(c, k[0], k[1], k[2]) for k, c in acc.items() if c]


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
def parse_number(s) -> Tuple[str, Optional[F]]:
    """A charge value from a source: returns (stored string, exact Fraction or None)."""
    if isinstance(s, (int, float)):
        d = Decimal(repr(s))
        return format(d, "f") if isinstance(s, float) else str(s), None
    s = str(s).strip()
    if "/" in s or re.fullmatch(r"-?\d+", s):
        fr = F(s)
        return f"{fr.numerator}/{fr.denominator}" if fr.denominator != 1 else str(fr.numerator), fr
    return s, None


def render_fraction(fr: F, digits: int = 30) -> str:
    """Shortest exact decimal when the expansion terminates, else `digits` significant."""
    q = fr.denominator
    while q % 2 == 0:
        q //= 2
    while q % 5 == 0:
        q //= 5
    if q == 1:
        d = Decimal(fr.numerator) / Decimal(fr.denominator)
        s = format(d.normalize(), "f")
        return s if "." in s else s + (".0" if False else "")
    with localcontext() as ctx:
        ctx.prec = digits
        ctx.rounding = ROUND_HALF_EVEN
        return format(Decimal(fr.numerator) / Decimal(fr.denominator), "f")


def numbers_agree(source: str, stored: str, exact: Optional[F]) -> str:
    """'exact' | 'print-convention' | 'unattributed'."""
    if source == stored:
        return "exact"
    try:
        a = Decimal(source)
        b = Decimal(exact.numerator) / Decimal(exact.denominator) if exact is not None else Decimal(stored)
    except Exception:
        return "unattributed"
    with localcontext() as ctx:
        ctx.prec = 40
        if a == b:
            return "print-convention"
        # source has fewer digits: compare at the source's precision
        sig = len(a.as_tuple().digits)
        if a == b.quantize(a) if sig <= 32 else False:
            return "print-convention"
        if abs(a - b) <= Decimal(10) ** (a.as_tuple().exponent) / 2:
            return "print-convention"
    return "unattributed"


# --------------------------------------------------------------------------- #
# module records
# --------------------------------------------------------------------------- #
NONFIELD = {"w", "a", "c", "consistency", "global", "rational", "decoupled", "relevant", "fliped",
            "marginal", "dim3", "non-manifestsymmetry", "fullindex", "index", "shortindex", "nw", "n"}


def module_field_names(rec: dict) -> List[str]:
    if "global" in rec:
        return list(rec["global"].keys())
    n = rec["nw"][0] if "nw" in rec else rec["n"]
    names = []
    for sp, cnt in zip(NAME_ORDER, n):
        names += [f"{sp}{j + 1}" for j in range(cnt)]
    return names


def module_theory(rec: dict, node: Node) -> Theory:
    names = module_field_names(rec)
    fields = []
    for nm in names:
        m = _FIELD.match(nm)
        sp = m.group(1)
        fields.append((species_label(node, sp),))
    pos = {nm: k for k, nm in enumerate(names)}
    th = Theory([node], fields, [], names=names)
    _set_terms(th, rec.get("w", []), pos)
    return th


def infer_flips(th: Theory) -> Dict[int, Dict[int, int]]:
    flips = {}
    for f in range(th.n_fields()):
        if not th.is_trivial(f):
            continue
        hits = [t for t in th.terms if t.get(f) == 1]
        if len(hits) == 1 and len(hits[0]) > 1:
            flips[f] = {g: p for g, p in hits[0].items() if g != f}
    return flips


# --------------------------------------------------------------------------- #
# the 2023 product-group corpus (quiver.py)
# --------------------------------------------------------------------------- #
CORPUS_SEEDS = {
    "SU2SU2bi4": ([Node("A", 1), Node("A", 1)], [["q", "q"]] * 4),
    "SU2SU2N=3": ([Node("A", 1), Node("A", 1)], [["q", "1"], ["q", "q"], ["phi", "1"], ["q", "phi"]]),
    "SU2SU4A2A5": ([Node("A", 1), Node("A", 3)], [["q", "qb"], ["qb", "q"], ["phi", "1"], ["1", "phi"], ["1", "q"], ["1", "qb"]]),
    "Sp1Sp2A2A3": ([Node("C", 1), Node("C", 2)], [["q", "q"], ["q", "q"], ["phi", "1"], ["1", "phi"], ["1", "q"], ["1", "q"]]),
    "SU4SU4linear": ([Node("A", 3), Node("A", 3)], [["phi", "1"], ["1", "phi"], ["q", "qb"], ["qb", "q"], ["q", "1"], ["qb", "1"], ["1", "q"], ["1", "qb"]]),
}


def corpus_theory(rec_or_nw, seed: str) -> Theory:
    nodes, seed_reps = CORPUS_SEEDS[seed]
    if isinstance(rec_or_nw, dict):
        rec = rec_or_nw
        reps = rec.get("n")
        w = rec.get("w", [])
        if reps is None:
            k = max([len(seed_reps)] + [int(_FIELD.match(nm).group(2)) for nm in rec if _FIELD.match(nm) and nm.startswith("O")]
                    + [int(m) for t in w for m in re.findall(r"O(\d+)", t)])
            reps = seed_reps + [["1"] * len(nodes)] * (k - len(seed_reps))
    else:
        reps, w = rec_or_nw
        k = max([len(reps)] + [int(m) for t in w for m in re.findall(r"O(\d+)", t)])
        reps = list(reps) + [["1"] * len(nodes)] * (k - len(reps))
    names = [f"O{i + 1}" for i in range(len(reps))]
    fields = [tuple(species_label(node, sp) for node, sp in zip(nodes, rp)) for rp in reps]
    pos = {nm: k for k, nm in enumerate(names)}
    th = Theory(list(nodes), fields, [], names=names)
    _set_terms(th, w, pos)
    return th


def _set_terms(th: Theory, w: Sequence[str], pos: Dict[str, int]) -> None:
    """Superpotential strings -> terms; a string with a non-positive exponent (an
    F-term-substituted expression the 2023 corpus wrote as a superpotential term, a
    defect of that code's post-processing) is kept
    verbatim in th.legacy_terms and excluded from the theory."""
    th.terms = []
    th.legacy_terms = []
    for s in w:
        mono, _ = parse_monomial(s, pos)
        if all(p > 0 for p in mono.values()):
            th.terms.append(mono)
        else:
            th.legacy_terms.append(s)
    th.flips = infer_flips(th)


def quiver2026_theory(nw, nodes: Sequence[Node]) -> Theory:
    """[[name, [rep per node], multiplicity], ...], [w ...]  (multiplicity 1 only)."""
    spec, w = nw
    names, fields = [], []
    for name, reps, mult in spec:
        if mult != 1:
            raise ValueError(f"multiplicity {mult} for {name}: only 1 is supported")
        names.append(name)
        fields.append(tuple(species_label(node, sp) for node, sp in zip(nodes, reps)))
    pos = {nm: k for k, nm in enumerate(names)}
    th = Theory(list(nodes), fields, [], names=names)
    _set_terms(th, w, pos)
    return th


# --------------------------------------------------------------------------- #
# record conversion with the round-trip classifier
# --------------------------------------------------------------------------- #
APPROVED = ("exact", "print-convention", "notation", "flavor-basis", "class-name-map", "legacy-defect")


def convert_record(rec: dict, th: Theory, source: dict, t_order: int = 9) -> Tuple[dict, dict]:
    """Model record + round-trip report {key: class}."""
    names = th.display_names()
    pos = {nm: k for k, nm in enumerate(names)}
    report: Dict[str, str] = {}
    # ---- superpotential ----
    legacy_terms = list(getattr(th, "legacy_terms", []))
    defects: List[str] = []
    if legacy_terms:
        defects.append("superpotential term with a non-positive exponent (old F-term substitution)")
    rendered = iter(monomial_string(t, names) for t in th.terms)
    for src in rec.get("w", []):
        if src in legacy_terms:
            report[f"w:{src}"] = "legacy-defect"
            continue
        ren = next(rendered)
        report[f"w:{src}"] = "exact" if src == ren else ("notation" if parse_monomial(src, pos)[0] == parse_monomial(ren, pos)[0] else "unattributed")
    # ---- charges ----
    rational = rec.get("rational") or {}
    R, exact_flags = [], []
    for nm in names:
        if nm in rec:
            s, fr = parse_number(rec[nm])
            if fr is None and nm in rational:
                fr = F(str(rational[nm]))
                s = f"{fr.numerator}/{fr.denominator}"
            R.append(s)
            exact_flags.append(fr)
            report[f"R:{nm}"] = numbers_agree(str(rec[nm]) if not isinstance(rec[nm], float) else format(Decimal(repr(rec[nm])), "f"), render_fraction(fr) if fr is not None else s, fr)
        else:
            R.append(None)
            exact_flags.append(None)
    a = c = None
    for key in ("a", "c"):
        if key in rec:
            s, fr = parse_number(rec[key])
            if fr is None and key in rational:
                fr = F(str(rational[key]))
                s = f"{fr.numerator}/{fr.denominator}"
            if key == "a":
                a = s
            else:
                c = s
            report[f"{key}"] = numbers_agree(str(rec[key]) if not isinstance(rec[key], float) else format(Decimal(repr(rec[key])), "f"), render_fraction(fr) if fr is not None else s, fr)
    charges = {"R": R, "a": a, "c": c, "rational": bool(rational)}
    # ---- flavor lattice and transformation ----
    B_new = flavor_lattice(th)
    T = None
    basis_defect = False
    if "global" in rec:
        k_old = len(next(iter(rec["global"].values()))) if rec["global"] else 0
        B_old = [[int(rec["global"][nm][a_]) for nm in names] for a_ in range(k_old)]
        B_old = [row for row in B_old if any(row)]
        rows = constraint_rows(th)
        violated = [(r_i, a_) for r_i, row in enumerate(rows) for a_, b in enumerate(B_old)
                    if sum(x * y for x, y in zip(row, b)) != 0]
        if violated or len(B_old) != len(B_new):
            # the source's U(1) basis does not satisfy the anomaly/neutrality constraints
            # of its own theory (a defect of the 2023 corpus): no transformation exists, the
            # index stays unconverted
            basis_defect = True
            defects.append("source flavor basis violates the constraints or has the wrong rank; index not convertible")
            report["flavor:basis"] = "legacy-defect"
        else:
            T = solve_integer_transformation(B_old, B_new)
            report["flavor:basis"] = "exact"
    # ---- index ----
    index_terms = None
    if "fullindex" in rec and basis_defect:
        report["fullindex"] = "legacy-defect"
    if "fullindex" in rec and T is not None:
        _, rounded = normalize_exponents(rec["fullindex"])
        red_old = parse_index(rec["fullindex"], len(T))
        L = left_inverse(T)
        red_new = [PhysicalTerm(t.coeff, t.milli, t.ypow, inverse_transform_flavor(t.flavor, L)) for t in red_old]
        index_terms = full_index(red_new, t_order, len(B_new))
        # round trip: full -> reduced -> old basis -> compare with the source polynomial
        back = [PhysicalTerm(t.coeff, t.milli, t.ypow, transform_flavor(t.flavor, T)) for t in reduce_index(index_terms, t_order)]
        report["fullindex"] = ("print-convention" if rounded else "notation") if index_equal(back, red_old) else "unattributed"
        if "index" in rec:
            unref = parse_index(rec["index"], 0)
            mine = [PhysicalTerm(t.coeff, t.milli, t.ypow, ()) for t in back]
            report["index"] = "notation" if index_equal(mine, unref) else "unattributed"
    # ---- operators ----
    ops = {}
    for src_key, dst in (("decoupled", "decoupled"), ("fliped", "flipped"), ("relevant", "relevant"), ("marginal", "marginal")):
        if src_key in rec:
            items = []
            for s in rec[src_key]:
                mono, coeff = parse_monomial(s, pos)
                items.append({"monomial": sorted([[f, p] for f, p in mono.items()]), "multiplicity": int(coeff)})
            ops[dst] = items
            back = [monomial_string({f: p for f, p in it["monomial"]}, names, it["multiplicity"]) for it in items]
            negative = any(p <= 0 for it in items for _, p in it["monomial"])
            if negative:
                defects.append(f"operator list {src_key} with a non-positive exponent (old F-term substitution)")
            if back == list(rec[src_key]):
                report[f"ops:{src_key}"] = "exact"
            elif all(parse_monomial(x, pos) == parse_monomial(y, pos) for x, y in zip(back, rec[src_key])):
                report[f"ops:{src_key}"] = "legacy-defect" if negative else "notation"
            else:
                report[f"ops:{src_key}"] = "unattributed"
    record = make_record(th, charges=charges, flavor_basis=B_new, index_terms=index_terms, t_order=t_order if index_terms is not None else None,
                         operators=ops, verdict=str(rec.get("consistency", "")),
                         provenance=dict(source, name_map=names, basis_transformation=T, legacy_terms=legacy_terms, legacy_defects=sorted(set(defects)),
                                         legacy_index_strings=({k: rec[k] for k in ("fullindex", "index") if k in rec} if basis_defect else None),
                                         legacy_index_fields={"fullindex": "flavor-refined reduced index", "index": "unrefined reduced index"}))
    return record, report


def classify(report: Dict[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for v in report.values():
        counts[v] = counts.get(v, 0) + 1
    return counts
