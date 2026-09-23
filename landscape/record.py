"""Records: one theory through a-maximization (amax.py), the index (index.py) and the
post-processing (post.py) to a record of the schema `schema/record.schema.json`, written
as JSON lines.

The per-theory sequence follows the original landscape code: charges -> expansion at the
low order -> operators at R <= 2/3 -> when there are any, one of them is flipped (a
trivial field X with the term X.O added, the flip recorded in `theory.flips`) and the
sequence restarts -> otherwise the expansion at the full order (no descent to a lower
order when it is not returned) -> conditions and operators.  The operator flipped is the
first entry of the decoupled list (sorted by monomial).

Verdicts: those of amax.VERDICTS, those of post.INDEX_VERDICTS, and
  negative-central-charge, hofman-maldacena-violation   (checked in this order before the
                                                         index verdict)
  post-processing-error a branch the inherited post-processing leaves ill-defined; the
                        message is in provenance
  index-not-computed    no expansion is returned at the low order or at the requested order
                        (form.MAX_ORDER exceeded or a timeout).
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
from typing import Dict, Iterable, List, Optional

import mpmath as mp

from . import amax, post
from .model import Theory, ambiguous_terms, canonical_form, identity_of_fixed_point, make_record, project

RECORD_VERSION = "35.0"
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


def build(th: Theory, engine, t_order: int = 9, low_order: int = 3, provenance: Optional[dict] = None) -> dict:
    """The record of `th` (after the flips of its decoupled operators)."""
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
        low = engine.expansion(th, charges, low_order)
        if low is None:
            return rec("index-not-computed")
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
        order = t_order                  # no descent to lower orders
        terms = engine.expansion(th, charges, order)
        if terms is None:
            return rec("index-not-computed")
        try:
            out = post.index(terms, basis, th.terms, order)
        except post.PostProcessingError as e:
            prov_here["post_processing_error"] = str(e)
            return rec("post-processing-error", terms, order)
        verdict = _ratio_verdict(res.a, res.c) or out.verdict
        analysis = {"consistency": out.consistency, "dim3": out.dim3, "nonmanifest_symmetry": out.nonmanifest_symmetry,
                    "susy_enhanced": out.susy_enhanced, "unlisted_positive_terms": out.unlisted,
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
