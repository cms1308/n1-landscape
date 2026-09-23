"""A seed through the level-by-level enumeration.

Writes level-<L>.jsonl (records with duplicate marks) under the output directory; the run
is resumable.  Seeds: `sp2nf5` (Sp(2), ten fundamentals) and `su2su2` (SU(2) x SU(2), one
adjoint per node and a bifundamental).

    python examples/run_seed.py sp2nf5 --levels 1 --core 4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # run from a checkout

from landscape import driver
from landscape.model import Node, Theory

SEEDS = {
    "sp2nf5": Theory(nodes=[Node("C", 2)], fields=[((1, 0),)] * 10, terms=[],
                     names=[f"q{i + 1}" for i in range(10)]),
    "su2su2": Theory(nodes=[Node("A", 1), Node("A", 1)],
                     fields=[((2,), (0,)), ((0,), (2,)), ((1,), (1,))], terms=[],
                     names=["Phi1", "Phi2", "Q"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", choices=sorted(SEEDS))
    ap.add_argument("--levels", type=int, default=1)
    ap.add_argument("--t-order", type=int, default=9)
    ap.add_argument("--core", type=int, default=4, help="worker processes")
    ap.add_argument("--out", default=None, help="output directory (default runs/<seed>)")
    ap.add_argument("--stores", default="stores")
    args = ap.parse_args()
    out = args.out or f"runs/{args.seed}"
    Path(args.stores).mkdir(parents=True, exist_ok=True)
    summary = driver.run(SEEDS[args.seed], out, args.levels, name=args.seed, store_dir=args.stores,
                         t_order=args.t_order, core=args.core)
    for level, info in summary["levels"].items():
        print(f"level {level}: {info}")


if __name__ == "__main__":
    main()
