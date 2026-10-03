# n1-landscape

The computational pipeline of the 4d N=1 SCFT landscape program
([arXiv:2408.02953](https://arxiv.org/abs/2408.02953), building on
[arXiv:1806.08353](https://arxiv.org/abs/1806.08353) and
[arXiv:1610.05311](https://arxiv.org/abs/1610.05311)), generalized to a product of
simple gauge groups `G_1 x ... x G_k` and written on a structured data model.

Given a gauge theory (nodes, matter fields as Dynkin labels, superpotential terms) the
package

1. checks the gauge anomalies (cubic anomaly from the weight system, Witten anomaly as
   the mod-2 index of the representation),
2. solves the anomaly-free R-symmetry and maximizes the trial central charge `a`
   (a-maximization) exactly over the rationals where possible, otherwise to 30
   significant digits, with the flavor charge lattice in Hermite normal form,
3. computes the superconformal index as a series in `t` (the plethystic expansion and the
   gauge-singlet projection in a native Rust extension, with a persistent character store),
4. extracts the gauge-invariant operators (decoupled, relevant, flipped, marginal), flips
   the operators that hit the unitarity bound and applies the consistency conditions of the
   landscape papers,
5. enumerates the landscape level by level (one deformation per relevant or flipped
   operator) with canonical-form deduplication of inputs and the duplicate rule for
   fixed points (equal unrefined index and central charges).

Every record is a JSON document validated against `landscape/schema/record.schema.json`.

## Requirements

- Python >= 3.10, [mpmath](https://mpmath.org/) and [SymPy](https://www.sympy.org/) (installed by `pip install .`); `pip install .[validate]` adds `jsonschema` for `record.validator()`
- a Rust toolchain (`rustup`) and [maturin](https://www.maturin.rs/), to build the native
  extension `landscape_native`, which the package requires

No FORM, no LiE, no Mathematica and no database server are needed. The character store is a
sqlite file per gauge group, created automatically and filled on demand by the extension's
character engine.

```bash
pip install .
pip install ./native
```

## Native extension

The `native/` directory holds the Rust extension `landscape_native` (PyO3). It computes:

- the characters: Adams operations, tensor products and dominant characters (Freudenthal's
  recursion) with LiE's output format, and the number of gauge singlets of a product of symmetric
  powers (Newton's formula over Adams operations) for the ambiguity flag of a superpotential term;
- the expansion: the exponential of the single-letter series, truncated at `t^t_order`, its
  monomials carried as products of irreducible characters of the gauge group (Brauer-Klimyk
  tensor products on the dominant characters), the components that cannot return to the trivial
  representation within the remaining `t`-degree dropped, and the gauge-singlet multiplicity taken
  at the end; above `t^6` the monomials carry the flavor exponents of the record's basis instead of
  the field markers, and the rows of the full order through `t^6` are taken from the order-6
  expansion already computed for the early rejection;
- the post-processing pass over the engine's rows: the reduced index, its flavor projection, the
  net index and the physical index; the F-term substitution, the operator lists and the record
  stay in Python.

A step of the expansion whose entries exceed the in-memory budget is computed again in passes over
parts of its key space, and a part still beyond the budget is written to disk in shards and merged
by streaming, so a build's memory is bounded by the budget. Coefficients are 128-bit rationals
promoted in place to arbitrary precision on overflow; a row beyond 128 bits is post-processed by
the Python path in exact arithmetic. On six heavy theories of the development record a build takes
1-6 s and under 0.4 GB, against 14-791 s and 6.5-20.5 GB before the engine's memory work (one of
them then through FORM).
Every native function was checked against the pure-Python path, FORM or LiE on the same inputs
(those references are kept in the development record, not in the package) and gives the same
records.

Independently of the engine, `record.build` rejects a theory early when the C1/C2 conditions
already fail on the exact part of an order-6 expansion (`prefilter=(3, 6)`, the default;
`prefilter=None` disables it): such a record carries its index and identity at order 6
(`index.t_order`, `provenance.prefilter_order`) and the order-9 expansion is skipped -- in a
campaign, where most candidates are rejected, this halves the wall of a rejection-heavy batch. A
theory whose full-order expansion is beyond the stop of the expansion order (below) takes the
C1/C2 test on its order-3 expansion only.

Switches (environment variables): `LANDSCAPE_NATIVE_THREADS=n` the engines' thread count;
`LANDSCAPE_NATIVE_MAX_TERMS=n` the in-memory budget of a step (default 4,000,000 entries);
`LANDSCAPE_NATIVE_PASSES=0` spills a step beyond the budget instead of computing it in passes;
`LANDSCAPE_NATIVE_SPILL_DIR` the directory of the spill (the system's temporary directory by
default); `LANDSCAPE_NATIVE_SPLICE=0` computes the full order without the order-6 rows;
`LANDSCAPE_NATIVE_FLAVOR=0` keeps the field markers above `t^6`; `LANDSCAPE_NATIVE_POST=0` selects
the Python post-processing. `LANDSCAPE_NATIVE_COEF64=1` and `LANDSCAPE_NATIVE_EXACT=1` turn on
variants of the engine's arithmetic and truncation that were measured and not adopted; they give
the same records.

## Quick start

```python
from landscape import amax, index, record
from landscape.model import Node, Theory

# Sp(2) with ten fundamentals, W = 0
th = Theory(nodes=[Node("C", 2)],
            fields=[((1, 0),)] * 10,          # one Dynkin label per node, per field
            terms=[],                          # superpotential terms: {field index: power}
            names=[f"q{i + 1}" for i in range(10)])

res = amax.solve(th)
print(res.verdict, res.a, res.c)             # consistent 339/200 257/100

stores = index.open_stores(th, "stores")     # one sqlite store per node group
engine = index.IndexEngine(stores)
rec = record.build(th, engine, t_order=9)
print(rec["verdict"], rec["charges"])
print(rec["index"]["terms"][:3])            # [t*1000, y power, flavor exponents, coefficient]
```

`examples/sp2_nf5.py` is the same computation as a script; `examples/run_seed.py` runs a
seed through the level-by-level enumeration with `landscape.driver.run`, writing one JSON
lines file per level.

## The data model

- **Node**: a simple group given by its Cartan type and rank, `Node("A", 2)` for SU(3),
  `Node("C", 2)` for Sp(2), etc. Types A to G are supported (E and F are admitted by the
  same code but not covered by the regression checks). Nodes are the simply connected
  groups; the global form is out of scope.
- **Field**: a tuple of Dynkin labels, one per node, in LiE's convention (the zero label
  for a node the field does not charge). Dimension, Dynkin index and conjugation are
  computed from the highest weight, so any representation is admissible.
- **Superpotential term**: an exponent vector `{field index: power}`. A term whose field
  content has gauge-singlet multiplicity above one does not determine the contraction; such
  a theory is flagged (`canonical.ambiguous_terms`) and receives no canonical hash.
- **Flip field**: a trivial field whose entry in `Theory.flips` records the operator it
  flips.
- **Record**: charges (exact rationals as `p/q`, otherwise 30 significant digits), the
  flavor basis (rows = U(1) generators), the full refined index as a sparse polynomial
  (`t` on the 1/1000 grid, `y`, flavor exponents, coefficient), the operator sets as
  exponent vectors, the verdict, the canonical form, the fixed-point identity and
  provenance. See `landscape/schema/`.

Index convention: `I(t, y; x) = Tr (-1)^F t^{3(R + 2 j_1)} y^{2 j_2} x^f`; the reduced
index is `(1 - t^3 y)(1 - t^3 / y)(I - 1)`, marginal operators sit at `t^6`.
`landscape.index.render_19a` prints an index in the package's string notation
(`x1, x2, ...` for the flavor fugacities, negative exponents for inverse powers) and
`render_latex` in LaTeX; both are renderers of the stored polynomial, not identities.

Verdict classes (in the order they are tested): `gauge-anomaly`, `witten-anomaly`,
`no-r-symmetry`, `r-charge-undetermined`, `no-local-maximum`, `non-positive-r-charge`,
`negative-central-charge`, `hofman-maldacena-violation`, `index-not-computed`,
`post-processing-error`, `inconsistent-index`, `free-sector-higher-spin-current`,
`free-sector`, `vanishing-index`, `consistent`.

## Modules

| Module | Contents |
|---|---|
| `lie` | Cartan matrices, root systems, Weyl dimension formula, Dynkin index, conjugation, weight tensors for the anomaly validators; dominant characters from the extension |
| `model` | `Node`, `Theory`, the flavor lattice, canonical form, index interfaces, the identity of a fixed point, JSON (de)serialization |
| `amax` | anomaly validators, the constraint system, a-maximization (Newton at 60 digits, certificate at 80, exact rational detection) |
| `store` | the character store: sqlite per group, entries generated on a miss by the Adams/tensor recursion in the extension's character engine |
| `singlet` | the gauge-singlet projection for a product group and the call of the native expansion engine |
| `form` | the single-letter series of a theory, the expansion orders, the stop of the expansion order, the work bound and the lower truncations |
| `mass` | the fields a mass term (a degree-two superpotential monomial) makes massive, and the superpotential written in the remaining fields |
| `index` | the index engine: series -> native expansion and projection -> field-resolved expansion -> physical index in the canonical flavor basis; renderers |
| `post` | operator extraction, F-term substitution, consistency conditions C1/C2/C1'/C3/C4 |
| `record` | one theory to one record, including the flips of decoupled operators; JSON lines I/O and schema validation |
| `enumerate` | the next level of the landscape, input deduplication, the duplicate rule for fixed points |
| `driver` | a seed through the levels with a process pool, resumable |
| `notation` | the index-string notation: parser (Mathematica InputForm included) and printer |
| `convert` | converters from the legacy formats of the original landscape code into the model |

## Notes on the conventions

- The **equivalence of two fixed points** is that of arXiv:2408.02953 sec. 3: equal
  unrefined index (below the truncation order, term by term) and equal central charges
  (relative tolerance `10^-25`). The stored `identity` is a hash of the rounded data; it
  coincides with this predicate on every record it was checked on but is not the
  predicate itself. A database should retrieve candidates by `enumerate.fixed_point_key`
  and apply `enumerate.equivalent`, as `enumerate.mark_duplicates` does.
- A positive index term with a negative power of a field (a fermion of a field outside
  the superpotential) is never an operator-list entry; it is reported under
  `analysis.negative_power_terms`.
- The fields a **mass term** makes massive (a superpotential monomial of degree two, `a b` or `a^2`) are left out of the
  index expansion: their letters cancel in the flavor-refined index, so the index, the identity and the central charges
  are unchanged, and operators are named by the remaining fields through the superpotential with the massive fields
  eliminated by their F-term equations (`mass.effective_superpotential`). The theory, its superpotential and the
  a-maximization keep the massive fields.
- The **F-term substitution** of the operator extraction pairs each fermion term (a relation dW/df = 0 among the bosons
  of its block) with a boson term present in the block, by a maximum matching over the monomials of dW/df, the first
  superpotential monomial tried first (`post.FTERM_RULE`, "first" for the original rule, which always takes it); a
  bin where the index lies below the number of listed operators then holds a genuine fermionic operator.
- The plethystic expansion stops above order 100 (`form.MAX_ORDER`, at every truncation; 40 in the
  original code; `None` lifts it, though the memory of an expansion beyond it is not yet bounded) and at a remaining field whose boson or fermion letter falls at
  `t^0` on the 1/1000 grid (3R or 3(2 - R) below 0.0005), recording `index-not-computed` with
  `provenance.not_computed = {"cause": "expansion-order", "order": k}`.
- The limit of an expansion is a work bound, not a wall clock: an expansion that forms more than
  10^10 product terms (`form.WORK_BOUND`) is cut and the record is `index-not-computed` with the
  cause `"work-bound"` and the bound in force (`"bound"`), so that a record does not depend on the
  machine or its load and a later run with a larger bound rebuilds exactly those records. A 24-h
  wall-clock limit per expansion (`form.EXPANSION_TIMEOUT_S`) is a safety net (cause `"deadline"`).
- No descent to a lower expansion order: a theory whose expansion is not returned at the
  requested order is recorded as `index-not-computed`.
- The decoupling pass reads the exponents of the flavor-refined scalar part of the reduced index
  (terms of different flavor charges are different operators and do not cancel) and flips at the
  lowest exponent at or below `t^2` whose block, after the F-term substitution, holds a chiral
  operator. A theory whose expansion order at `t^3` exceeds 100 runs the pass on the truncations
  `t^0.008` ... `t^2` first (`form.LOW_TRUNCATIONS`), each deciding a flip only.
- `analysis.nonmanifest_symmetry` is true when the index proves conserved currents beyond the
  manifest U(1)s: at `t^6 y^0` of the flavor-refined reduced index a charge `q != 0` with a negative
  coefficient, or the neutral coefficient plus the flavor rank negative. It is a lower bound
  (currents cancelled by marginal operators are not seen).
- C1' and C3 (`j >= 1`) are evaluated per flavor charge of the reduced index, a condition holding
  when it holds at some charge. `analysis.extra_supercurrents` is the sum over the charges `q` of
  `max(c_q, 0)`, `c_q` the coefficient of `t^7 chi_{1/2}` at charge `q`: a lower bound on the extra
  supercurrent multiplets (N >= k + 1 or a free sector).
- `driver.run` uses one worker process per CPU by default (`core=`); `LANDSCAPE_NATIVE_THREADS`
  sets the engine's threads in each.

## Citing

If you use this code, please cite the landscape papers it implements:
arXiv:2408.02953 (Cho, Maruyoshi, Nardoni, Song), arXiv:1806.08353 (Maruyoshi, Nardoni,
Song) and arXiv:1610.05311 (Agarwal, Maruyoshi, Song).

## License

MIT, see `LICENSE`.
