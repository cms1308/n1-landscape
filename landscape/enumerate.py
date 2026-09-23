"""The next level of the landscape: the deformations of an accepted record -- one theory
per relevant operator (the operator added to the superpotential) and one per flipped
operator (a trivial field M with the term M.O) -- with the deduplication of the inputs
and the duplicate rule for fixed points.

Conventions, inherited from the original landscape code (arXiv:2408.02953):

  * the deformations of a record are taken from `operators.relevant` and
    `operators.flipped` only (the unlisted positive terms and the entries with a negative
    power are never enumerated);
  * a record without a flavor symmetry (flavor rank 0) contributes no relevant deformation,
    only its flips;
  * the inputs of a level are deduplicated by the canonical form of their exponent data.
    A theory with an ambiguous superpotential term (singlet multiplicity above one) is
    flagged in its record and carries no canonical hash there; the enumeration nevertheless
    merges the inputs whose exponent data coincide up to relabeling -- the exponent data
    are all it generates -- and expands the flagged theory;
  * two fixed points are equivalent when they have the same unrefined index
    (arXiv:2408.02953 sec. 3, "Statistics on the Landscape": "We say that two fixed points
    are equivalent if they have the same unrefined index").  Here: the unrefined index
    below t^t_order term by term, and a and c to `CENTRAL_DIGITS` significant digits.  The
    later description is the duplicate only when its flavor rank does not exceed the
    earlier one's; otherwise it is accepted and expanded (a description with more manifest
    symmetry).  `equivalent` is this predicate: equal `fixed_point_key` (sha256 of the
    unrefined index below t^t_order) and `central_equal` (a relative tolerance of
    10^-CENTRAL_DIGITS).  The record also stores `identity` = model.identity_of_fixed_point
    (t_order, the unrefined index below t^t_order, a and c rounded half-even to
    CENTRAL_DIGITS significant digits), an exact key for a database: `identity(rec)`
    recomputes it.  The predicate compares the central charges with a relative tolerance
    (a non-transitive relation, which no hash reproduces exactly); the stored identity
    compares their roundings (transitive).  Equal identities are neither sufficient nor
    necessary for `equivalent` -- the two disagree at rounding boundaries in both
    directions (equal roundings with the tolerance failing, e.g. 1 and 1 + 4e-25; the
    tolerance holding with different roundings, e.g. 1 + 4.9e-25 and 1 + 5.1e-25) -- and
    gave the same marks on every checked record.  A database therefore retrieves the
    candidates of a record by `fixed_point_key` (the unrefined-index key at the cutoff,
    computable from any record with an index) and applies `equivalent` and the ordered rank
    rule, exactly as `mark_duplicates` does; a lookup by `identity` can only be an
    optimization with that complete fallback (an equality lookup alone misses tolerance
    matches with different roundings).  A duplicate keeps its physical verdict; the mark is
    `provenance.enumeration.duplicate_of`.

`fixed_point_key` and `central_equal` also serve the comparisons with records converted
from legacy sources (whose charges may be machine-precision floats, compared to fewer
digits)."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from typing import Dict, List, Optional, Sequence, Tuple

import mpmath as mp

from . import record as record_
from .model import CENTRAL_DIGITS, Theory, canonical_form, record_identity, unrefined_index


# --------------------------------------------------------------------------- #
# keys
# --------------------------------------------------------------------------- #
def input_key(th: Theory) -> str:
    """Canonical encoding of the exponent data (nodes, labels, terms) up to relabeling."""
    return canonical_form(th).hash()


def unrefined_terms(rec: dict) -> Optional[List[Tuple[int, int, str]]]:
    """[(milli, y, coefficient)] of the unrefined index below t^t_order -- the inherited strict
    truncation E < t_order: the reduced-index strings the old codes compared hold exactly this
    content (the terms at t^t_order of a stored index are complete, those of a converted old
    record are not)."""
    if rec.get("index") is None:
        return None
    return unrefined_index(rec["index"]["terms"], rec["index"]["t_order"])


def fixed_point_key(rec: dict) -> Optional[str]:
    """sha256 of the unrefined index (the equivalence of 2408.02953 sec. 3.1); None without an index."""
    terms = unrefined_terms(rec)
    if terms is None:
        return None
    return hashlib.sha256(json.dumps({"t_order": rec["index"]["t_order"], "unrefined": terms}).encode()).hexdigest()


def _central(s: Optional[str]):
    if s is None:
        return None
    if "/" in s:
        fr = F(s)
        return mp.mpf(fr.numerator) / fr.denominator
    return mp.mpf(s)


def central_equal(a: Optional[str], b: Optional[str], digits: int = CENTRAL_DIGITS) -> bool:
    if a is None or b is None:
        return a is b
    with mp.workdps(40):
        x, y = _central(a), _central(b)
        return abs(x - y) <= mp.mpf(10) ** (-digits) * max(abs(x), abs(y), mp.mpf(1))


def identity(rec: dict) -> Optional[str]:
    """The stored key of a record, recomputed (model.record_identity): the rounded form of
    the equivalence; record.build stores it for an unflagged record (null for a flagged one).  Not the predicate of `equivalent` (see the module docstring)."""
    return record_identity(rec)


def equivalent(rec: dict, other: dict) -> bool:
    """The same fixed point: equal unrefined index below t^t_order and central charges equal
    within the relative tolerance 10^-CENTRAL_DIGITS."""
    k = fixed_point_key(rec)
    return (k is not None and k == fixed_point_key(other)
            and central_equal(rec["charges"]["a"], other["charges"]["a"])
            and central_equal(rec["charges"]["c"], other["charges"]["c"]))


# --------------------------------------------------------------------------- #
# deformations of one record
# --------------------------------------------------------------------------- #
def children(rec: dict) -> List[Tuple[Theory, dict]]:
    """(theory, info) for every deformation of an accepted record: the relevant operators
    added to W (none when the flavor rank is 0), then the flipped operators as M.O."""
    th = Theory.from_json(rec["theory"])
    out: List[Tuple[Theory, dict]] = []
    ops = rec.get("operators") or {}
    if rec["flavor"]["rank"] > 0:
        for it in ops.get("relevant", []):
            op = {int(f): int(p) for f, p in it["monomial"]}
            child = Theory(nodes=list(th.nodes), fields=list(th.fields), terms=[dict(t) for t in th.terms] + [op],
                           flips={f: dict(o) for f, o in th.flips.items()}, names=list(th.names) if th.names else None)
            out.append((child, {"kind": "relevant", "operator": [list(x) for x in it["monomial"]]}))
    for it in ops.get("flipped", []):
        op = {int(f): int(p) for f, p in it["monomial"]}
        child = record_.flip(th, op, prefix="M")
        out.append((child, {"kind": "flip", "operator": [list(x) for x in it["monomial"]], "field": th.n_fields()}))
    return out


def is_expanded(rec: dict) -> bool:
    """A record is expanded to the next level when it is consistent and not a duplicate."""
    enum = (rec.get("provenance") or {}).get("enumeration") or {}
    return rec["verdict"] == "consistent" and enum.get("duplicate_of") is None


# --------------------------------------------------------------------------- #
# a level
# --------------------------------------------------------------------------- #
def next_level(records: Sequence[dict], level: int) -> List[dict]:
    """Inputs of `level` from the final records of level - 1 (in their order): one input per
    canonical encoding, the first occurrence kept, the merged relabelings counted."""
    inputs: List[dict] = []
    seen: Dict[str, dict] = {}
    for rec in records:
        if not is_expanded(rec):
            continue
        parent = rec["provenance"]["enumeration"]["key"]
        for child, info in children(rec):
            key = input_key(child)
            if key in seen:
                seen[key]["enumeration"]["merged"] += 1
                continue
            inp = {"key": key, "theory": child.to_json(),
                   "enumeration": {"level": level, "key": key, "parent": parent, "merged": 0, **info}}
            seen[key] = inp
            inputs.append(inp)
    return inputs


def mark_duplicates(records: Sequence[dict], accepted: Optional[List[dict]] = None) -> List[dict]:
    """The duplicate rule over `records` in order: a consistent record equivalent to an
    earlier accepted one (in `accepted`, then earlier in `records`) with a flavor rank not
    above it is marked `duplicate_of` that record's input key; the others are appended
    to `accepted`.  Returns the marked records (the same objects)."""
    accepted = accepted if accepted is not None else []
    by_key: Dict[str, List[dict]] = {}
    for a in accepted:
        k = fixed_point_key(a)
        if k is not None:
            by_key.setdefault(k, []).append(a)
    for rec in records:
        enum = rec["provenance"].setdefault("enumeration", {})
        enum["duplicate_of"] = None
        if rec["verdict"] != "consistent":
            enum["expanded"] = False
            continue
        k = fixed_point_key(rec)
        match = None
        for a in by_key.get(k, []) if k is not None else []:
            if equivalent(rec, a) and rec["flavor"]["rank"] <= a["flavor"]["rank"]:
                match = a
                break
        if match is not None:
            enum["duplicate_of"] = match["provenance"]["enumeration"]["key"]
            enum["expanded"] = False
        else:
            enum["expanded"] = True
            accepted.append(rec)
            if k is not None:
                by_key.setdefault(k, []).append(rec)
    return list(records)
