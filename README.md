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
3. computes the superconformal index as a series in `t` (FORM for the plethystic
   expansion, LiE for the gauge-singlet projection, with a persistent character store),
4. extracts the gauge-invariant operators (decoupled, relevant, flipped, marginal), flips
   the operators that hit the unitarity bound and applies the consistency conditions of the
   landscape papers,
5. enumerates the landscape level by level (one deformation per relevant or flipped
   operator) with canonical-form deduplication of inputs and the duplicate rule for
   fixed points (equal unrefined index and central charges).

Every record is a JSON document validated against `landscape/schema/record.schema.json`.

## Requirements

- Python >= 3.10 and [mpmath](https://mpmath.org/)
- [FORM](https://www.nikhef.nl/~form/) (`form`, and `tform` for the parallel path) on `PATH`
- [LiE](http://wwwmathlabo.univ-poitiers.fr/~maavl/LiE/) (`lie`) on `PATH`

No Mathematica and no database server are needed. The character store is a sqlite file
per gauge group, created automatically and filled on demand through LiE.

```bash
pip install .
```

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
engine = index.IndexEngine(stores, workdir="work")
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
| `lie` | Cartan matrices, root systems, Weyl dimension formula, Dynkin index, conjugation, weight tensors for the anomaly validators, a LiE runner |
| `model` | `Node`, `Theory`, the flavor lattice, canonical form, index interfaces, the identity of a fixed point, JSON (de)serialization |
| `amax` | anomaly validators, the constraint system, a-maximization (Newton at 60 digits, certificate at 80, exact rational detection) |
| `store` | the character store: sqlite per group, entries generated on a miss by the Adams/tensor recursion through LiE |
| `singlet` | FORM-output parser and the gauge-singlet projection for a product group |
| `form` | the FORM program of the index and its runner (sequential `form` or `tform` by an automatic size policy) |
| `index` | the index engine: FORM -> projection -> field-resolved expansion -> physical index in the canonical flavor basis; renderers |
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
- No descent to a lower expansion order: a theory whose expansion is not returned at the
  requested order is recorded as `index-not-computed`.
- FORM's scratch files go to a per-process directory under the runner's work directory
  and are removed after a timeout.
- `driver.run` defaults to 15 worker processes; pass `core=` to match your machine.
  LiE deadlines are wall-clock, so the machine must stay awake during long runs.

## Citing

If you use this code, please cite the landscape papers it implements:
arXiv:2408.02953 (Cho, Maruyoshi, Nardoni, Song), arXiv:1806.08353 (Maruyoshi, Nardoni,
Song) and arXiv:1610.05311 (Agarwal, Maruyoshi, Song).

## License

MIT, see `LICENSE`.
