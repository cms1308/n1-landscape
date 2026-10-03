"""Records: one theory through a-maximization (amax.py), the index (index.py) and the
post-processing (post.py) to a record of the schema `schema/record.schema.json`, written
as JSON lines.

The per-theory sequence follows the original landscape code: charges -> expansion at the
low order (for a theory whose expansion order at t^3 exceeds form.LOW_TRUNCATION_ORDER, first
the lower truncations form.LOW_TRUNCATIONS, each deciding a flip only; a term below a
truncation is complete there) -> operators at R <= 2/3 -> when there are any, one of them is flipped (a
trivial field X with the term X.O added, the flip recorded in `theory.flips`) and the
sequence restarts -> otherwise the early-rejection stage (`prefilter`: post.prefilter, the
C1/C2 conditions on the exact part of a low-order expansion -- the decoupling pass's order-3
expansion first, then an expansion of order 6; a violation found there is a violation of the
full-order pass, so the theory is recorded inconsistent-index from that order, post.index at
that order giving its index, identity and analysis, `index.t_order` that order and
`provenance.prefilter_order` recording it, and the full-order expansion is skipped; a hit whose
post.index raises or does not return inconsistent-index falls back to the full order,
`provenance.prefilter_fallback` recording why; a low-order expansion past the deadline leaves the
decision to the full order; a low-order expansion cut by the work bound ends the build
index-not-computed, the full order forming every product term the lower one forms) ->
otherwise the expansion at the full order
(no descent to a lower order when it is not returned) -> conditions and operators.  A theory
whose full-order expansion stops (an order above form.MAX_ORDER, or a remaining field whose letter
falls at t^0 on the grid) takes the C1/C2 test on the decoupling pass's order-3 expansion (already
computed) and is otherwise recorded index-not-computed, without the order-6 expansion: it can be
neither consistent nor expanded to the next level.  The
operator flipped is the first entry of the decoupled list (sorted by monomial).

Verdicts: those of amax.VERDICTS, those of post.INDEX_VERDICTS, and
  negative-central-charge, hofman-maldacena-violation   (checked in this order before the
                                                         index verdict)
  post-processing-error a branch the inherited post-processing leaves ill-defined; the
                        message is in provenance
  index-not-computed    no expansion is returned at the low order or at the requested order;
                        provenance.not_computed = {"cause", "order"}: "expansion-order" (an order
                        above form.MAX_ORDER, or a remaining field whose letter falls at t^0 on the
                        1/1000 grid -- an R-charge too close to 0 or 2), "work-bound" (the engine formed more than
                        form.WORK_BOUND product terms; "bound" that bound -- not the count at the cut, which
                        depends on the engine's threads) or "deadline"
                        (the expansion past form.EXPANSION_TIMEOUT_S, a safety net), and the order
                        of that expansion (a fraction for a lower truncation).
A theory with an ambiguous superpotential term (singlet multiplicity above one) is written
with `canonical.hash = null` and its ambiguous terms listed.

`identity`: the key of the fixed point -- t_order, the unrefined index below t^t_order and
the central charges rounded to 25 significant digits (model.identity_of_fixed_point) --
for an unflagged record with an index and central charges; null for a flagged theory (the
enumeration recomputes the key for it, enumerate.identity) and null without an index or
central charges.  The enumeration's equivalence is the tolerance predicate of
enumerate.equivalent, with which the stored key coincides on every checked record."""
from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import mpmath as mp

from . import amax, form, post, singlet
from .model import Theory, ambiguous_terms, canonical_form, identity_of_fixed_point, make_record, project

RECORD_VERSION = "36.0"
SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
MAX_FLIPS = 32


def flip(th: Theory, op: Dict[int, int], prefix: str = "X") -> Theory:
    """th with a trivial field X and the superpotential term X.O; op = {field: power}.
    The display name is prefix + a running number (X for a decoupled operator; M for the
    flip deformations of the enumeration)."""
    x = th.n_fields()
    names = None
    if th.names:
        k = 1 + sum(1 for nm in th.names if nm.startswith(prefix) and nm[len(prefix):].isdigit())
        names = list(th.names) + [f"{prefix}{k}"]
    return Theory(nodes=list(th.nodes), fields=list(th.fields) + [tuple(n.zero() for n in th.nodes)],
                  terms=[dict(t) for t in th.terms] + [{x: 1, **{int(f): int(p) for f, p in op.items()}}],
                  flips={**{f: dict(o) for f, o in th.flips.items()}, x: {int(f): int(p) for f, p in op.items()}},
                  names=names)


def _ratio_verdict(a, c) -> Optional[str]:
    a, c = mp.mpf(a.numerator) / a.denominator if isinstance(a, F) else a, mp.mpf(c.numerator) / c.denominator if isinstance(c, F) else c
    if a <= 0 or c <= 0:
        return "negative-central-charge"
    if a / c <= mp.mpf(1) / 2 or a / c >= mp.mpf(3) / 2:
        return "hofman-maldacena-violation"
    return None


def build(th: Theory, engine, t_order: int = 9, low_order: int = 3, provenance: Optional[dict] = None,
          prefilter: Optional[Sequence[int]] = (3, 6)) -> dict:
    """The record of `th` (after the flips of its decoupled operators).  `prefilter`: the orders
    below t_order at which the early-rejection stage evaluates C1/C2 (an order equal to
    low_order reuses the decoupling pass's expansion); None disables the stage."""
    prov = dict(provenance or {})
    prov.setdefault("record_version", RECORD_VERSION)
    flipped_ops: List[dict] = []
    for _ in range(MAX_FLIPS + 1):
        amb = ambiguous_terms(th)
        canonical = canonical_form(th)
        res = amax.solve(th)
        prov_here = dict(prov, amax={k: (v if isinstance(v, (int, bool, str, type(None))) else str(v)) for k, v in res.info.items()},
                         witten=res.witten, flips_added=flipped_ops, t_order_requested=t_order)
        basis = amax.flavor_basis(th, bool(amb))

        def rec(verdict, terms=None, order=None, ops=None, analysis=None):
            phys = None
            identity = None
            if terms is not None:
                phys = project(terms, basis)
                if not amb and res.a is not None:      # null for a flagged theory
                    cj = res.charges_json()
                    identity = identity_of_fixed_point(phys, order, cj["a"], cj["c"])
            r = make_record(th, charges=res.charges_json() if res.R is not None else {"R": [None] * th.n_fields(), "a": None, "c": None, "rational": False},
                            flavor_basis=basis, index_terms=phys, t_order=order, operators=ops or {}, verdict=verdict,
                            provenance=prov_here, canonical=canonical, ambiguous=amb, identity=identity)
            r["schema_version"] = RECORD_VERSION
            if analysis is not None:
                r["analysis"] = analysis
            return r

        if res.verdict != "consistent":
            return rec(res.verdict)
        charges = res.charges_json()["R"]

        def not_computed(k, cause=None):
            # the cause the engine gives (index.IndexEngine.stop); without it, an expansion the series stops never reached
            # the engine and any other was past its deadline
            stop = dict(getattr(engine, "stop", None) or {}) if cause is None else {"cause": cause}
            if not stop.get("cause"):
                stop["cause"] = "expansion-order" if form.expansion_stops(th, charges, k) else "deadline"
            stop.pop("terms", None)                 # the count at a work-bound cut depends on the threads; the bound is kept
            prov_here["not_computed"] = {"order": k, **stop}
            return rec("index-not-computed")
        dec = None
        for k in form.low_truncations(th, charges):     # the decoupling pass on lower truncations first: a flip only
            low_k = engine.expansion(th, charges, k, basis=basis)
            if low_k is None:
                return not_computed(float(k))           # the low-order expansion forms every product term of this one
            try:
                dec_k = post.decouple(low_k, basis, th.terms, k)
            except post.PostProcessingError:
                continue
            if dec_k.decoupled:
                dec = dec_k
                break
        if dec is None:
            low = engine.expansion(th, charges, low_order, basis=basis)
            if low is None:
                return not_computed(low_order)
            try:
                dec = post.decouple(low, basis, th.terms, low_order)
            except post.PostProcessingError as e:
                prov_here["post_processing_error"] = str(e)
                return rec("post-processing-error")
        if dec.decoupled:
            op = {f: p for f, p in dec.decoupled[0]["monomial"]}
            flipped_ops = flipped_ops + [{"operator": dec.decoupled[0]["monomial"], "field": th.n_fields()}]
            th = flip(th, op)
            continue
        stops = form.expansion_stops(th, charges, t_order)   # neither consistent nor expanded further: the order-3 test only
        order, terms, out = t_order, None, None     # no descent to lower orders
        below = None                                # the order-6 expansion, for the splice (singlet.NATIVE_SPLICE)
        for k in (prefilter or ()):                 # early rejection on the exact part of a low order
            if not low_order <= k < t_order or (stops and k != low_order):
                continue
            low_k = low if k == low_order else engine.expansion(th, charges, k, basis=basis)
            if low_k is None:
                if (getattr(engine, "stop", None) or {}).get("cause") == "work-bound":
                    return not_computed(k)          # the full order forms every product term of this one
                break                               # past the deadline: the full order decides
            if k == 6:
                below = low_k
            if not post.prefilter(low_k, basis, k):
                continue
            try:
                out_k = post.index(low_k, basis, th.terms, k)
            except post.PostProcessingError as e:
                prov_here["prefilter_fallback"] = f"order {k}: {e}"
                break
            if out_k.verdict != "inconsistent-index":
                prov_here["prefilter_fallback"] = f"order {k}: {out_k.verdict}"
                break
            order, terms, out = k, low_k, out_k
            prov_here["prefilter_order"] = k
            break
        if terms is None and stops:
            return not_computed(t_order, "expansion-order")
        if terms is None:
            splice = {"below": below} if (below is not None and singlet.NATIVE_SPLICE) else {}
            terms = engine.expansion(th, charges, order, basis=basis, **splice)
            if terms is None:
                return not_computed(order)
            try:
                try:
                    out = post.index(terms, basis, th.terms, order)
                except post.FlavorRowsOverflow:
                    # a sum beyond 128 bits over the flavor-refined rows: the expansion field-resolved throughout,
                    # which the Python functions read in exact arithmetic
                    terms = engine.expansion(th, charges, order)
                    if terms is None:
                        return not_computed(order)
                    out = post.index(terms, basis, th.terms, order)
            except post.PostProcessingError as e:
                prov_here["post_processing_error"] = str(e)
                return rec("post-processing-error", terms, order)
        verdict = _ratio_verdict(res.a, res.c) or out.verdict
        analysis = {"consistency": out.consistency, "dim3": out.dim3, "nonmanifest_symmetry": out.nonmanifest_symmetry,
                    "extra_supercurrents": out.extra_supercurrents, "unlisted_positive_terms": out.unlisted,
                    "negative_power_terms": out.negative,
                    "flags": {k: (v if isinstance(v, bool) else [list(x) for x in v]) for k, v in out.flags.items()}}
        return rec(verdict, terms, order, out.operators(), analysis)
    raise RuntimeError(f"more than {MAX_FLIPS} flips")


# --------------------------------------------------------------------------- #
# JSON lines and the schema
# --------------------------------------------------------------------------- #
def write_jsonl(path, records: Iterable[dict], append: bool = True) -> int:
    n = 0
    with open(path, "a" if append else "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path) -> List[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def validator():
    import jsonschema
    from referencing import Registry, Resource
    tschema = json.loads((SCHEMA_DIR / "theory.schema.json").read_text())
    rschema = json.loads((SCHEMA_DIR / "record.schema.json").read_text())
    registry = Registry().with_resources([(tschema["$id"], Resource.from_contents(tschema)),
                                          (rschema["$id"], Resource.from_contents(rschema))])
    return jsonschema.Draft202012Validator(rschema, registry=registry)


def schema_errors(record: dict, _v=[]) -> List[str]:
    if not _v:
        _v.append(validator())
    return [e.message[:200] for e in _v[0].iter_errors(record)]
