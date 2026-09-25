"""Level-by-level driver of one seed: the main loop of the landscape enumeration on the
model, with a Pool of worker processes, resumable done-sets and
JSON lines per level.  A database adapter is not included; its interface is the
record schema.

Files under `out_dir`:
  config.json              the run's parameters and the complete seed payload (nodes, fields,
                           terms, flips, names); a resumed run must present the identical
                           payload -- a relabeled seed is refused
  level-<L>.inputs.jsonl   the inputs of level L (enumerate.next_level of level L-1, or the seed)
  level-<L>.raw.jsonl      records as the workers complete them (append-only; the done-set)
  level-<L>.errors.jsonl   inputs whose worker raised (a tool failure; the input is not retried)
  level-<L>.jsonl          the final records of level L in input order, with the duplicate
                           marks of enumerate.mark_duplicates
A level is complete when its final file exists.  Published files (inputs, final) are written
through a temporary file renamed into place, so a published file is complete or absent
; on resumption the inputs file is checked against the inputs
regenerated from the parent level and the final file against the input keys, a torn or
short published file is an explicit failure, a leftover temporary file is removed.  The raw
and errors files are append-only logs: a trailing fragment left by a killed writer is
truncated away before the next append (its input runs again), a malformed line anywhere
else is an explicit failure naming the file and the line.  Only the pending inputs run.

Each worker process opens its own character stores (sqlite, WAL) and its own FORM runner
(per-process TempDir under out_dir/frm); LiE runs as a subprocess per call
(landscape.store.run_lie by default).  A projector or FORM timeout makes the
record `index-not-computed`.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import enumerate as nextlevel
from . import index
from . import record as record_
from .model import Node, Theory

_W: Dict[str, object] = {}


class _Engine:
    """IndexEngine proxy: None on a LiE-chain timeout (FORM timeouts already return None)."""

    def __init__(self, eng):
        self.eng = eng

    def expansion(self, th, charges, t_order, basis=None):
        try:
            return self.eng.expansion(th, charges, t_order, basis=basis)
        except subprocess.TimeoutExpired:
            return None


def _init(cfg: dict) -> None:
    th0 = Theory([Node(t, int(r)) for t, r in cfg["nodes"]], [], [])
    stores = index.open_stores(th0, cfg["store_dir"])
    eng = index.IndexEngine(stores, Path(cfg["out_dir"]) / "frm", core=cfg["projector_core"],
                            tform_workers=cfg["tform_workers"], match_timeout=cfg["match_timeout"])
    _W["engine"] = _Engine(eng)
    _W["cfg"] = cfg


def _task(inp: dict) -> Tuple[str, dict]:
    cfg = _W["cfg"]
    t0 = time.time()
    try:
        th = Theory.from_json(inp["theory"])
        prov = {"seed": cfg["seed"], "enumeration": dict(inp["enumeration"])}
        rec = record_.build(th, _W["engine"], cfg["t_order"], cfg["low_order"], prov)
        rec["provenance"]["enumeration"]["seconds"] = round(time.time() - t0, 1)
        return "ok", rec
    except Exception as e:  # noqa: BLE001 -- a tool failure of one input; recorded, not retried
        return "error", {"key": inp["key"], "enumeration": inp["enumeration"], "error": f"{type(e).__name__}: {e}",
                         "traceback": traceback.format_exc()[-2000:], "seconds": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
class CorruptFile(RuntimeError):
    """A published or log file of a run directory is not in the state the driver wrote."""


def _tmp(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _publish(path: Path, records: Sequence[dict]) -> None:
    """Write `records` as JSON lines through a temporary file renamed into place."""
    tmp = _tmp(path)
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _drop_tmp(path: Path) -> None:
    tmp = _tmp(path)
    if tmp.exists():
        tmp.unlink()


def _parse_lines(path: Path, lines: List[bytes]) -> List[dict]:
    out = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise CorruptFile(f"malformed line {i + 1} of {path.name}: {e}") from None
    return out


def _read_published(path: Path) -> List[dict]:
    """A published file (inputs, final): every line must parse; a torn file is an error."""
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise CorruptFile(f"{path.name} is torn (no newline at the end); delete it to regenerate the level")
    return _parse_lines(path, data.split(b"\n"))


def _read_log(path: Path) -> List[dict]:
    """An append-only log (raw, errors): a trailing fragment without its newline is a killed
    writer's last line -- it is truncated away (its input runs again); any other malformed
    line is an error."""
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        keep = data.rfind(b"\n") + 1
        with open(path, "r+b") as f:
            f.truncate(keep)
        data = data[:keep]
    return _parse_lines(path, data.split(b"\n"))


def seed_input(seed: Theory) -> dict:
    key = nextlevel.input_key(seed)
    return {"key": key, "theory": seed.to_json(),
            "enumeration": {"level": 0, "key": key, "parent": None, "kind": "seed", "operator": None, "merged": 0}}


def _expected_inputs(seed: Theory, out: Path, L: int) -> List[dict]:
    return [seed_input(seed)] if L == 0 else nextlevel.next_level(_read_published(out / f"level-{L - 1}.jsonl"), L)


def _load_inputs(seed: Theory, out: Path, L: int) -> List[dict]:
    """The inputs of level L: the published file, verified against the inputs regenerated
    from the parent level, or generated and published now."""
    path = out / f"level-{L}.inputs.jsonl"
    _drop_tmp(path)
    expected = _expected_inputs(seed, out, L)
    if path.exists():
        inputs = _read_published(path)
        if [i["key"] for i in inputs] != [e["key"] for e in expected]:
            raise CorruptFile(f"{path.name} does not match the inputs regenerated from level {L - 1} "
                              f"({len(inputs)} against {len(expected)}); delete it to regenerate the level")
        return inputs
    _publish(path, expected)
    return expected


def _load_final(out: Path, L: int, inputs: Sequence[dict], errors: Sequence[dict]) -> Optional[List[dict]]:
    """The final records of a complete level, verified against the input keys; None when
    the level is not complete."""
    path = out / f"level-{L}.jsonl"
    _drop_tmp(path)
    if not path.exists():
        return None
    recs = _read_published(path)
    have = {r["provenance"]["enumeration"]["key"] for r in recs} | {e["key"] for e in errors}
    want = {i["key"] for i in inputs}
    if have != want or len(recs) + len(errors) != len(inputs):
        raise CorruptFile(f"{path.name} holds {len(recs)} records and {len(errors)} errors for {len(inputs)} inputs; "
                          f"delete it to re-finalize the level from its raw file")
    return recs


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def run(seed: Theory, out_dir, levels: int, *, name: str, store_dir, t_order: int = 9, low_order: int = 3,
        core: Optional[int] = None, tform_workers: int = 4, projector_core: int = 1, match_timeout: float = 600.0,
        stop_after: Optional[Tuple[int, int]] = None, log=print) -> dict:
    """Run `seed` through levels 0..levels; resumable.  `stop_after = (L, k)` stops the run
    once level L holds k records (for the resumability check) and reports `interrupted`.
    `core` = worker processes, the machine's CPU count by default."""
    if core is None:
        core = os.cpu_count() or 1
    out = Path(out_dir)
    cfg = {"seed": name, "seed_theory": seed.to_json(), "nodes": [[n.type, n.rank] for n in seed.nodes],
           "store_dir": str(store_dir), "out_dir": str(out), "t_order": t_order, "low_order": low_order,
           "tform_workers": tform_workers, "projector_core": projector_core, "match_timeout": match_timeout}
    cfg_path = out / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old != cfg:
            diff = sorted(k for k in set(old) | set(cfg) if old.get(k) != cfg.get(k))
            raise RuntimeError(f"{out} belongs to another run: the configuration differs in {diff} "
                               f"(the seed payload must be identical, a relabeled seed included)")
    else:
        out.mkdir(parents=True, exist_ok=True)
        _tmp(cfg_path).write_text(json.dumps(cfg, indent=1))
        os.replace(_tmp(cfg_path), cfg_path)
    accepted: List[dict] = []
    summary: Dict[str, dict] = {"seed": name, "levels": {}}
    for L in range(levels + 1):
        inputs = _load_inputs(seed, out, L)
        raw_path, err_path = out / f"level-{L}.raw.jsonl", out / f"level-{L}.errors.jsonl"
        errors = _read_log(err_path)
        recs = _load_final(out, L, inputs, errors)
        if recs is not None:
            accepted += [r for r in recs if nextlevel.is_expanded(r)]
            summary["levels"][str(L)] = _counts(recs, errors, resumed=True)
            log(f"[driver {name}] level {L} complete (from file): {summary['levels'][str(L)]}")
            continue
        raw = _read_log(raw_path)
        done = {r["provenance"]["enumeration"]["key"] for r in raw} | {e["key"] for e in errors}
        pending = [i for i in inputs if i["key"] not in done]
        limit = None
        if stop_after is not None and stop_after[0] == L:
            limit = max(0, stop_after[1] - len(done))
            pending = pending[:limit]
        log(f"[driver {name}] level {L}: {len(inputs)} inputs, {len(done)} done, {len(pending)} to run", flush=True)
        if pending:
            ctx = mp.get_context("fork")
            with ctx.Pool(min(core, len(pending)), initializer=_init, initargs=(cfg,)) as pool, \
                    open(raw_path, "ab") as fraw, open(err_path, "ab") as ferr:
                for status, payload in pool.imap_unordered(_task, pending, chunksize=1):
                    f = fraw if status == "ok" else ferr
                    f.write((json.dumps(payload, sort_keys=True) + "\n").encode())
                    f.flush()
                    if status == "ok":
                        e = payload["provenance"]["enumeration"]
                        log(f"[driver {name}] L{L} {payload['verdict']} {e['seconds']}s fields={len(payload['theory']['fields'])} "
                            f"|W|={len(payload['theory']['terms'])} kind={e['kind']}", flush=True)
                    else:
                        log(f"[driver {name}] L{L} ERROR {payload['error']}", flush=True)
        if limit is not None and len(pending) + len(done) < len(inputs):
            summary["interrupted"] = {"level": L, "records": len(_read_log(raw_path))}
            log(f"[driver {name}] interrupted at level {L} after {summary['interrupted']['records']} records")
            return summary
        raw_by_key = {r["provenance"]["enumeration"]["key"]: r for r in _read_log(raw_path)}
        errors = _read_log(err_path)
        err_keys = {e["key"] for e in errors}
        missing = [i["key"] for i in inputs if i["key"] not in raw_by_key and i["key"] not in err_keys]
        if missing:
            raise RuntimeError(f"level {L}: {len(missing)} inputs without a record")
        recs = [raw_by_key[i["key"]] for i in inputs if i["key"] in raw_by_key]
        nextlevel.mark_duplicates(recs, accepted)
        _publish(out / f"level-{L}.jsonl", recs)
        summary["levels"][str(L)] = _counts(recs, errors)
        log(f"[driver {name}] level {L} complete: {summary['levels'][str(L)]}")
    # the file holds the run's counts only; `from_file` marks this invocation's resumption and is
    # returned, not stored, so a resumed run leaves the directory byte-identical
    stored = {"seed": name, "levels": {L: {k: v for k, v in c.items() if k != "from_file"} for L, c in summary["levels"].items()}}
    if not (out / "summary.json").exists() or json.loads((out / "summary.json").read_text()) != stored:
        _tmp(out / "summary.json").write_text(json.dumps(stored, indent=1))
        os.replace(_tmp(out / "summary.json"), out / "summary.json")
    return summary


def _counts(recs: List[dict], errors: List[dict], resumed: bool = False) -> dict:
    verdicts: Dict[str, int] = {}
    for r in recs:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    return {"records": len(recs), "verdicts": verdicts,
            "duplicates": sum(1 for r in recs if r["provenance"]["enumeration"].get("duplicate_of")),
            "expanded": sum(1 for r in recs if nextlevel.is_expanded(r)),
            "ambiguous": sum(1 for r in recs if r["canonical"] and r["canonical"]["ambiguous_terms"]),
            "errors": len(errors), "seconds": round(sum(r["provenance"]["enumeration"].get("seconds", 0) for r in recs), 1),
            **({"from_file": True} if resumed else {})}
