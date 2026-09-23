"""Character store keyed by Dynkin label.

One sqlite file per node group (WAL journal).  The key of a character decomposition is (group_rank, label,
key_vec): `label` is the LiE Dynkin label of the representation ("[2,0]", no spaces),
`key_vec` = str([m1, ..., mN]) with sum_k k m_k = N, the value the normalized LiE virtual
character of prod_k psi^k(chi_R)^{m_k}.  Species names no longer exist in the store:
`q` and `qb` of A1 are both [1], `S`, `Sb` and `phi` of C2 are all [2,0] -- the old
tables were keyed by species and are migrated many-to-one (`migrate_species_store`),
with alias-conflict detection before insertion and a provenance row per migration.

Any highest weight is admissible: a missing key is generated on the spot by the arxivGen
recursion (pure top Adams key -> Adams(N, label, G); otherwise one tensor of two strictly
lower-order entries) and persisted -- the LiE conventions (banner sentinel, maxnodes/maxobjects preamble,
maxobjects retry, process-group kill on timeout, output validated against the
character-polynomial regex) are those of the original character tables, against which
the generated entries were verified byte-identical.  LiE has no C1/B1: the
group name of every LiE call comes from lie.lie_group_name (C1 -> A1, same labels),
while the store key keeps the node's own group_rank.

The tensor-step cache is keyed by sha256(group_rank|products|decomp), so caches migrated
from the original pipeline stay valid for the projector of landscape/singlet.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import lie

LIE_TIMEOUT_S = 180.0        # per LiE subprocess
MAX_OBJECTS_RETRIES = 3      # grow maxobjects and rerun
BANNER_SLICE = 53            # LiE startup banner length, sentinel-verified
BUSY_TIMEOUT_MS = 60_000     # sqlite lock wait across Pool processes
SENTINEL = "123456789"

# A normalized LiE virtual-character polynomial: signed integer multiplicities, X,
# bracketed weight vectors; '+-1X[...]' juxtaposition allowed.
_POLY_RE = re.compile(r"^[+-]?\d+X\[-?\d+(,-?\d+)*\]([+-]{1,2}\d+X\[-?\d+(,-?\d+)*\])*$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS char_decomp (
    group_rank TEXT NOT NULL,
    label      TEXT NOT NULL,
    key_vec    TEXT NOT NULL,
    value      TEXT NOT NULL,
    source     TEXT NOT NULL,
    PRIMARY KEY (group_rank, label, key_vec)
);
CREATE TABLE IF NOT EXISTS tensor_cache (
    ckey   TEXT PRIMARY KEY,
    result TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alias_map (
    group_rank TEXT NOT NULL,
    species    TEXT NOT NULL,
    label      TEXT NOT NULL,
    PRIMARY KEY (group_rank, species)
);
CREATE TABLE IF NOT EXISTS provenance (
    id      INTEGER PRIMARY KEY,
    kind    TEXT NOT NULL,
    created TEXT NOT NULL,
    detail  TEXT NOT NULL
);
"""


class LabelStoreError(RuntimeError):
    """Loud failure: wrong LiE build, malformed output, alias conflict."""


def label_key(label: Sequence[int]) -> str:
    """LiE label string without spaces: (2, 0) -> '[2,0]'."""
    return "[" + ",".join(str(int(x)) for x in label) + "]"


def key_str(key_vec: Sequence[int]) -> str:
    """The Adams multiplicity key as the table files spell it: [1, 0, 2]."""
    return str([int(x) for x in key_vec])


def parse_group_rank(group_rank: str) -> Tuple[str, int]:
    return group_rank[0], int(group_rank[1:])


def run_lie(lcode: str, timeout: float) -> str:
    """LiE subprocess with the pipeline's kill semantics."""
    proc = subprocess.Popen(
        ["lie"], shell=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        out, _ = proc.communicate(input=lcode, timeout=timeout)
        return out
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise


def split_key(key: Sequence[int]) -> Tuple[list, list]:
    """arxivGen split of a non-pure-Adams key into two lower-order keys."""
    first_nonzero = next(i for i, m in enumerate(key) if m)
    frob1 = list(key[:len(key) - first_nonzero - 1])
    frob1[first_nonzero] -= 1
    frob2 = [1 if i == first_nonzero else 0 for i in range(first_nonzero + 1)]
    return frob1, frob2


def singlet_coefficient(decomp: str) -> int:
    """Singlet multiplicity of a normalized LiE character string: the first listed
    weight is the zero weight iff a singlet is present (LiE's ascending weight order,
    verified on the original stored tables)."""
    weight = decomp[decomp.find("X") + 1:decomp.find("]") + 1]
    if not any(int(x) for x in weight.strip("[]").split(",")):
        return int(decomp[0:decomp.find("X")])
    return 0


class LabelStore:
    """One sqlite-backed store bound to a group_rank, keyed by Dynkin label.

    Thread-safe within a process (single connection + lock); safe across processes via
    WAL + busy_timeout + idempotent INSERT OR IGNORE writes; fork-safe (the connection is
    reopened when the pid changes)."""

    def __init__(self, path: str | Path, group_rank: str,
                 lie_timeout: float = LIE_TIMEOUT_S,
                 lie_runner: Callable[[str, float], str] = run_lie,
                 create: bool = True):
        self.group_rank = group_rank
        t, n = parse_group_rank(group_rank)
        self.type, self.rank = t, n
        self.lie_group = lie.lie_group_name(t, n)
        self._lie_timeout = lie_timeout
        self._lie_runner = lie_runner
        self._lock = threading.Lock()
        self._decomp_memo: Dict[Tuple[str, str], str] = {}
        self._banner_checked = False
        self.lie_calls = 0
        self.generated = 0
        self._path = str(path)
        if not create and not Path(self._path).exists():
            raise FileNotFoundError(self._path)
        self._connect()
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def _connect(self) -> None:
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._pid = os.getpid()

    def _db(self) -> sqlite3.Connection:
        """Call under self._lock."""
        if os.getpid() != self._pid:
            self._connect()
        return self._conn

    @property
    def path(self) -> str:
        return self._path

    # ------------------------------------------------------------------ #
    # LiE invocation
    # ------------------------------------------------------------------ #
    def _check_banner(self) -> None:
        out = self._lie_runner(f"maxnodes 9999999\n {SENTINEL}", self._lie_timeout)
        if out[BANNER_SLICE:].strip() != SENTINEL:
            raise LabelStoreError(
                f"LiE banner slice [{BANNER_SLICE}:] invalid on this build; "
                f"sentinel run began {out[:90]!r}")
        self._banner_checked = True

    def lie_eval(self, expr: str) -> str:
        """Evaluate one LiE expression; normalized output (banner sliced, whitespace
        removed), with the maxobjects retry."""
        if not self._banner_checked:
            self._check_banner()
        max_objects = "9999999"
        for _ in range(MAX_OBJECTS_RETRIES + 1):
            lcode = f"maxnodes 9999999\n maxobjects {max_objects}\n {expr}"
            with self._lock:
                self.lie_calls += 1
            raw = self._lie_runner(lcode, self._lie_timeout)
            out = raw[BANNER_SLICE:].strip().replace("\n", "").replace(" ", "")
            if "(" not in out and "line" not in out:
                return out
            max_objects += "9"
        raise LabelStoreError(f"LiE gave no clean output for: {expr[:120]}...")

    # ------------------------------------------------------------------ #
    # character decompositions (generation on miss)
    # ------------------------------------------------------------------ #
    def decomp(self, label: Sequence[int], key_vec: Sequence[int]) -> str:
        lab = label_key(label)
        ks = key_str(key_vec)
        if len(key_vec) != sum((k + 1) * int(m) for k, m in enumerate(key_vec)) or len(label) != self.rank:
            raise LabelStoreError(f"malformed key {ks} or label {lab} for {self.group_rank}: "
                                  "a key has length sum_k k m_k, a label has one entry per simple root")
        memo_key = (lab, ks)
        with self._lock:
            value = self._decomp_memo.get(memo_key)
        if value is not None:
            return value
        with self._lock:
            row = self._db().execute(
                "SELECT value FROM char_decomp WHERE group_rank=? AND label=? AND key_vec=?",
                (self.group_rank, lab, ks)).fetchone()
        if row is not None:
            value = row[0]
        else:
            value = self._generate(tuple(int(x) for x in label), [int(x) for x in key_vec])
            self._put_decomp(lab, ks, value, source="generated")
            with self._lock:
                self.generated += 1
        with self._lock:
            self._decomp_memo[memo_key] = value
        return value

    def _generate(self, label: Tuple[int, ...], key: list) -> str:
        order = len(key)
        if key[-1] == 1:  # pure top Adams term
            value = self.lie_eval(f"Adams({order},{label_key(label)},{self.lie_group})")
        else:
            frob1, frob2 = split_key(key)
            pol1 = self.decomp(label, frob1)  # recursion persists the cone
            pol2 = self.decomp(label, frob2)
            value = self.lie_eval(f"tensor({pol1},{pol2},{self.lie_group})")
        if not _POLY_RE.match(value):
            raise LabelStoreError(
                f"LiE output for {self.group_rank}/{label_key(label)} {key} is not a "
                f"character polynomial: {value[:200]!r}")
        return value

    def regenerate(self, label: Sequence[int], key_vec: Sequence[int]) -> str:
        """Recompute one entry by a single recursion step through LiE, reading only the
        two strictly lower-order entries; the entry itself is
        neither read nor written."""
        return self._generate(tuple(int(x) for x in label), [int(x) for x in key_vec])

    def _put_decomp(self, lab: str, ks: str, value: str, source: str) -> None:
        with self._lock, self._db():
            self._db().execute(
                "INSERT OR IGNORE INTO char_decomp VALUES (?,?,?,?,?)",
                (self.group_rank, lab, ks, value, source))

    def put_decomp_many(self, rows: Iterable[Tuple[str, str, str]], source: str = "import") -> int:
        """Bulk insert of (label_key, key_str, value) rows; returns rows added."""
        with self._lock, self._db():
            before = self._db().execute("SELECT COUNT(*) FROM char_decomp").fetchone()[0]
            self._db().executemany(
                "INSERT OR IGNORE INTO char_decomp VALUES (?,?,?,?,?)",
                ((self.group_rank, lab, ks, value, source) for lab, ks, value in rows))
            after = self._db().execute("SELECT COUNT(*) FROM char_decomp").fetchone()[0]
        return after - before

    def has(self, label: Sequence[int], key_vec: Sequence[int]) -> bool:
        with self._lock:
            row = self._db().execute(
                "SELECT 1 FROM char_decomp WHERE group_rank=? AND label=? AND key_vec=?",
                (self.group_rank, label_key(label), key_str(key_vec))).fetchone()
        return row is not None

    # ------------------------------------------------------------------ #
    # tensor-step cache (same sha256 keys as the pipeline)
    # ------------------------------------------------------------------ #
    def cache_key(self, products: str, decomp: str) -> str:
        return hashlib.sha256(f"{self.group_rank}|{products}|{decomp}".encode()).hexdigest()

    def cache_get(self, ckey: str) -> Optional[str]:
        with self._lock:
            row = self._db().execute("SELECT result FROM tensor_cache WHERE ckey=?", (ckey,)).fetchone()
        return row[0] if row else None

    def cache_put(self, ckey: str, result: str) -> None:
        with self._lock, self._db():
            self._db().execute("INSERT OR IGNORE INTO tensor_cache VALUES (?,?)", (ckey, result))

    # ------------------------------------------------------------------ #
    # provenance
    # ------------------------------------------------------------------ #
    def record_provenance(self, kind: str, detail: dict) -> None:
        with self._lock, self._db():
            self._db().execute("INSERT INTO provenance (kind, created, detail) VALUES (?,?,?)",
                               (kind, time.strftime("%Y-%m-%dT%H:%M:%S%z"), json.dumps(detail, sort_keys=True)))

    def provenance(self) -> List[dict]:
        with self._lock:
            rows = self._db().execute("SELECT id, kind, created, detail FROM provenance ORDER BY id").fetchall()
        return [{"id": i, "kind": k, "created": c, "detail": json.loads(d)} for i, k, c, d in rows]

    def stats(self) -> dict:
        with self._lock:
            total, generated = self._db().execute(
                "SELECT COUNT(*), COALESCE(SUM(source='generated'),0) FROM char_decomp").fetchone()
            labels = [r[0] for r in self._db().execute(
                "SELECT DISTINCT label FROM char_decomp WHERE group_rank=? ORDER BY label", (self.group_rank,))]
            cache = self._db().execute("SELECT COUNT(*) FROM tensor_cache").fetchone()[0]
        return {"char_decomp": total, "char_decomp_generated": generated, "labels": labels,
                "tensor_cache": cache, "lie_calls": self.lie_calls, "generated_this_instance": self.generated}

    def integrity_check(self) -> str:
        with self._lock:
            return self._db().execute("PRAGMA integrity_check").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --------------------------------------------------------------------------- #
# migration of a species-keyed store (the original charstore layout)
# --------------------------------------------------------------------------- #
def file_sha256(path: str | Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def find_alias_conflicts(old_conn: sqlite3.Connection, group_rank: str, label_of: Dict[str, str]) -> Tuple[List[dict], int]:
    """Keys that several species map onto, with their values compared byte for byte.
    Returns (conflicts, number of duplicated mapped keys).  A duplicated key with equal
    values is a duplicate alias (merged by the migration); differing values are a
    conflict (the migration is refused)."""
    old_conn.execute("CREATE TEMP TABLE IF NOT EXISTS map (species TEXT PRIMARY KEY, label TEXT NOT NULL)")
    old_conn.execute("DELETE FROM map")
    old_conn.executemany("INSERT INTO map VALUES (?,?)", list(label_of.items()))
    n_dup = old_conn.execute(
        "SELECT COUNT(*) FROM (SELECT 1 FROM char_decomp c JOIN map m ON m.species=c.species "
        "WHERE c.group_rank=? GROUP BY m.label, c.key_vec HAVING COUNT(*) > 1)", (group_rank,)).fetchone()[0]
    # pairwise comparison of the species sharing a label, one pass over the table with
    # primary-key lookups of the alias rows (CROSS JOIN fixes the loop order)
    rows = old_conn.execute(
        "SELECT m1.label, a.key_vec, a.species, b.species, a.value, b.value "
        "FROM char_decomp a CROSS JOIN map m1 CROSS JOIN map m2 CROSS JOIN char_decomp b "
        "WHERE a.group_rank=? AND m1.species=a.species AND m2.label=m1.label AND m2.species>a.species "
        "AND b.group_rank=a.group_rank AND b.species=m2.species AND b.key_vec=a.key_vec AND b.value<>a.value",
        (group_rank,)).fetchall()
    conflicts = [{"label": lab, "key_vec": ks, "species": [s1, s2],
                  "value_sha256": [hashlib.sha256(v1.encode()).hexdigest()[:16], hashlib.sha256(v2.encode()).hexdigest()[:16]]}
                 for lab, ks, s1, s2, v1, v2 in rows]
    return conflicts, n_dup


def migrate_species_store(old_path: str | Path, new_path: str | Path, group_rank: str,
                          label_of: Dict[str, str], *, hash_source: bool = True) -> dict:
    """Migrate an old (group_rank, species, key_vec) store into a NEW label-keyed file.

    The old file is opened read-only and never modified.  Every species present in the
    old store must be in `label_of` (species -> LiE label string); species sharing a
    label are merged after the alias-conflict check.  Raises LabelStoreError on a
    conflict or an unregistered species, leaving no new file behind."""
    old_path, new_path = str(old_path), str(new_path)
    if Path(new_path).exists():
        raise LabelStoreError(f"refusing to overwrite {new_path}")
    t0 = time.time()
    old = sqlite3.connect(f"file:{old_path}?mode=ro", uri=True)
    try:
        present = [r[0] for r in old.execute(
            "SELECT DISTINCT species FROM char_decomp WHERE group_rank=? ORDER BY species", (group_rank,))]
        missing = [s for s in present if s not in label_of]
        if missing:
            raise LabelStoreError(f"species without a label in the registry: {missing}")
        per_species = dict(old.execute(
            "SELECT species, COUNT(*) FROM char_decomp WHERE group_rank=? GROUP BY species", (group_rank,)).fetchall())
        old_rows = sum(per_species.values())
        conflicts, n_dup = find_alias_conflicts(old, group_rank, {s: label_of[s] for s in present})
        if conflicts:
            raise LabelStoreError(f"{len(conflicts)} conflicting alias values in {old_path}: {conflicts[:3]}")
        old_cache = old.execute("SELECT COUNT(*) FROM tensor_cache").fetchone()[0]
    finally:
        old.close()
    # bulk phase on a plain connection (no WAL yet), then the store adopts the file
    new = sqlite3.connect(new_path)
    try:
        new.execute("PRAGMA journal_mode=OFF")
        new.execute("PRAGMA synchronous=OFF")
        new.executescript(_SCHEMA)
        new.execute("ATTACH DATABASE ? AS old", (f"file:{old_path}?mode=ro",))
        new.execute("CREATE TEMP TABLE map (species TEXT PRIMARY KEY, label TEXT NOT NULL)")
        new.executemany("INSERT INTO map VALUES (?,?)", [(s, label_of[s]) for s in present])
        new.execute(
            "INSERT OR IGNORE INTO main.char_decomp (group_rank, label, key_vec, value, source) "
            "SELECT c.group_rank, m.label, c.key_vec, c.value, c.source FROM old.char_decomp c "
            "JOIN map m ON m.species=c.species WHERE c.group_rank=?", (group_rank,))
        new.execute("INSERT OR IGNORE INTO main.tensor_cache SELECT ckey, result FROM old.tensor_cache")
        new.executemany("INSERT OR REPLACE INTO main.alias_map VALUES (?,?,?)",
                        [(group_rank, s, label_of[s]) for s in present])
        new_rows = new.execute("SELECT COUNT(*) FROM main.char_decomp").fetchone()[0]
        new_cache = new.execute("SELECT COUNT(*) FROM main.tensor_cache").fetchone()[0]
        distinct = new.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT m.label, c.key_vec FROM old.char_decomp c "
            "JOIN map m ON m.species=c.species WHERE c.group_rank=?)", (group_rank,)).fetchone()[0]
        new.commit()
        new.execute("DETACH DATABASE old")
    finally:
        new.close()
    report = {
        "group_rank": group_rank, "source": old_path, "target": new_path,
        "source_size": os.path.getsize(old_path),
        "source_sha256": file_sha256(old_path) if hash_source else None,
        "label_map": {s: label_of[s] for s in present},
        "old_rows": old_rows, "old_rows_per_species": per_species,
        "distinct_mapped_keys": distinct, "new_rows": new_rows, "duplicate_alias_keys": n_dup,
        "conflicts": 0, "tensor_cache_old": old_cache, "tensor_cache_new": new_cache,
        "seconds": round(time.time() - t0, 1),
    }
    st = LabelStore(new_path, group_rank)
    st.record_provenance("migration", report)
    st.close()
    return report


def verify_migration(old_path: str | Path, new_path: str | Path, group_rank: str, label_of: Dict[str, str]) -> dict:
    """Independent check of a migrated store by a full SQL join: every old row's value
    is byte-identical to the value under its mapped key; new rows = distinct mapped keys;
    tensor-cache rows identical."""
    conn = sqlite3.connect(f"file:{new_path}?mode=ro", uri=True)
    try:
        conn.execute("ATTACH DATABASE ? AS old", (f"file:{old_path}?mode=ro",))
        conn.execute("CREATE TEMP TABLE map (species TEXT PRIMARY KEY, label TEXT NOT NULL)")
        conn.executemany("INSERT INTO map VALUES (?,?)", list(label_of.items()))
        unmapped = conn.execute(
            "SELECT COUNT(*) FROM old.char_decomp c LEFT JOIN map m ON m.species=c.species "
            "WHERE c.group_rank=? AND m.label IS NULL", (group_rank,)).fetchone()[0]
        mismatched = conn.execute(
            "SELECT COUNT(*) FROM old.char_decomp c JOIN map m ON m.species=c.species "
            "LEFT JOIN main.char_decomp n ON n.group_rank=c.group_rank AND n.label=m.label AND n.key_vec=c.key_vec "
            "WHERE c.group_rank=? AND (n.value IS NULL OR n.value <> c.value)", (group_rank,)).fetchone()[0]
        old_rows = conn.execute("SELECT COUNT(*) FROM old.char_decomp WHERE group_rank=?", (group_rank,)).fetchone()[0]
        distinct = conn.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT m.label, c.key_vec FROM old.char_decomp c "
            "JOIN map m ON m.species=c.species WHERE c.group_rank=?)", (group_rank,)).fetchone()[0]
        new_rows = conn.execute("SELECT COUNT(*) FROM main.char_decomp WHERE group_rank=?", (group_rank,)).fetchone()[0]
        new_other = conn.execute("SELECT COUNT(*) FROM main.char_decomp WHERE group_rank<>?", (group_rank,)).fetchone()[0]
        cache_old = conn.execute("SELECT COUNT(*) FROM old.tensor_cache").fetchone()[0]
        cache_new = conn.execute("SELECT COUNT(*) FROM main.tensor_cache").fetchone()[0]
        cache_diff = conn.execute(
            "SELECT COUNT(*) FROM (SELECT ckey, result FROM old.tensor_cache EXCEPT SELECT ckey, result FROM main.tensor_cache)").fetchone()[0]
        cache_diff += conn.execute(
            "SELECT COUNT(*) FROM (SELECT ckey, result FROM main.tensor_cache EXCEPT SELECT ckey, result FROM old.tensor_cache)").fetchone()[0]
        conflicts, n_dup = find_alias_conflicts(sqlite3.connect(f"file:{old_path}?mode=ro", uri=True), group_rank, label_of)
    finally:
        conn.close()
    return {"group_rank": group_rank, "old_rows": old_rows, "distinct_mapped_keys": distinct, "new_rows": new_rows,
            "new_rows_other_group": new_other, "unmapped_old_rows": unmapped, "value_mismatches": mismatched,
            "duplicate_alias_keys": n_dup, "conflicts": len(conflicts),
            "tensor_cache_old": cache_old, "tensor_cache_new": cache_new, "tensor_cache_differences": cache_diff,
            "ok": (unmapped == 0 and mismatched == 0 and new_rows == distinct and not conflicts
                   and cache_old == cache_new and cache_diff == 0)}
