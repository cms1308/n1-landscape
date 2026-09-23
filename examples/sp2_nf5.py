"""One theory through the pipeline: Sp(2) with ten fundamentals and W = 0.

Expected: verdict `consistent`, R = 2/5 for every fundamental, a = 339/200, c = 257/100.
Needs `form` and `lie` on PATH.  The character store is created under ./stores on the
first run (a few minutes of LiE at t_order 9); later runs read it.

    python examples/sp2_nf5.py [--t-order 9]
"""
import argparse
import json
from fractions import Fraction
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # run from a checkout

from landscape import amax, index, record
from landscape.model import Node, PhysicalTerm, Theory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-order", type=int, default=9, help="truncation order of the index (t^t_order)")
    ap.add_argument("--stores", default="stores", help="directory of the character stores")
    ap.add_argument("--work", default="work", help="FORM work directory")
    args = ap.parse_args()

    th = Theory(nodes=[Node("C", 2)],
                fields=[((1, 0),)] * 10,
                terms=[],
                names=[f"q{i + 1}" for i in range(10)])

    res = amax.solve(th)
    print("a-maximization:", res.verdict, "a =", res.a, "c =", res.c)
    print("R-charges:", [str(r) for r in res.R])

    t0 = time.time()
    Path(args.stores).mkdir(parents=True, exist_ok=True)
    Path(args.work).mkdir(parents=True, exist_ok=True)
    stores = index.open_stores(th, args.stores)
    engine = index.IndexEngine(stores, workdir=args.work)
    rec = record.build(th, engine, t_order=args.t_order)
    print(f"record in {time.time() - t0:.1f} s")
    print("verdict:", rec["verdict"])
    print("charges:", json.dumps(rec["charges"]))
    print("flavor rank:", rec["flavor"]["rank"])
    terms = [PhysicalTerm(Fraction(c), m, y, tuple(fl)) for m, y, fl, c in rec["index"]["terms"]]
    print("reduced unrefined index:", index.render_19a(index.unrefined(index.reduced(terms, args.t_order))))
    print("refined index (first terms):", index.render_19a(terms[:6]), "...")
    print("operators:", {k: len(v) for k, v in rec["operators"].items() if v is not None})
    errors = record.schema_errors(rec)
    print("schema:", "valid" if not errors else errors)


if __name__ == "__main__":
    main()
