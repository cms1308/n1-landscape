//! The character-arithmetic engine (step 39b): a port of the pipeline's Python engine
//! `store/pylie.py` (step 20, R18) to Rust, reproducing LiE's stdout for the lcode forms the
//! pipeline sends -- `Adams(n,[..],G)` and `tensor(P,Q,G)` with the `maxnodes` banner line,
//! an optional `maxobjects` line and the `res=...; print(res);` form -- byte for byte
//! (LiE's `print_poly` layout: LMARGIN 5, right margin 70, coefficient field width, per-column
//! label widths, the ` +` continuation before a signed coefficient).  LiE itself stays the
//! reference and the fallback; an lcode outside these forms raises NotImplementedError and the
//! caller falls back to the subprocess.
//!
//! Mathematics (LiE box/domchar.c, plethysm.c, decomp.c, tensor.c; notes/22 §3):
//!   * dominant characters by Freudenthal's recursion on dominant weights;
//!   * Adams(n, .): scale the dominant weights by n, then Vdecomp -- for each scaled dominant
//!     weight traverse the Weyl orbit of the unshifted weight, add rho, reflect to the dominant
//!     chamber with sign, subtract rho, drop walls;
//!   * tensor(V_lam, V_mu): for each weight eta of the factor of smaller dimension (orbits of
//!     its dominant weights) lam + rho + eta reflected with sign, minus rho, walls dropped;
//!     virtual characters distribute bilinearly.
//! Weights are Dynkin labels; A_ij = <alpha_i^vee, alpha_j> (LiE numbering, A-D and G2).
//! Exact integer arithmetic (i128 in the Freudenthal sums).  A deadline is checked between
//! the contributions of the hot loops, and its expiry raises subprocess.TimeoutExpired, the
//! exception the callers already translate (the bounded cancellation step 20 lacked).

use num_bigint::BigInt;
use num_traits::{Signed, Zero};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Mutex;
use std::time::Instant;

pub type Weight = Vec<i64>;
pub type Poly = BTreeMap<Weight, BigInt>;

/// A weight as a fixed array in the hot loops (ranks up to MAXR); the rank is carried by the group.
pub const MAXR: usize = 16;
type W = [i64; MAXR];

const LMARGIN: usize = 5;
const RIGHT_MARGIN: usize = 70;
/// Dominant characters memoized per group, at most this many (cleared when reached).
const MEMO_CAP: usize = 2048;

fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn lcm(a: i128, b: i128) -> i128 {
    if a == 0 || b == 0 {
        0
    } else {
        (a / gcd(a, b) * b).abs()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Frac {
    n: i128,
    d: i128,
}

impl Frac {
    fn new(n: i128, d: i128) -> Frac {
        let g = gcd(n, d);
        let (mut n, mut d) = if g > 1 { (n / g, d / g) } else { (n, d) };
        if d < 0 {
            n = -n;
            d = -d;
        }
        Frac { n, d }
    }
    fn int(v: i128) -> Frac {
        Frac { n: v, d: 1 }
    }
    fn sub(self, o: Frac) -> Frac {
        Frac::new(self.n * o.d - o.n * self.d, self.d * o.d)
    }
    fn mul(self, o: Frac) -> Frac {
        Frac::new(self.n * o.n, self.d * o.d)
    }
    fn div(self, o: Frac) -> Frac {
        Frac::new(self.n * o.d, self.d * o.n)
    }
    fn is_zero(self) -> bool {
        self.n == 0
    }
}

/// Cartan matrix A_ij = <alpha_i^vee, alpha_j> and the half-norms d_i relative to the long roots.
fn cartan(typ: char, n: usize) -> Result<(Vec<Vec<i64>>, Vec<Frac>), String> {
    let mut a = vec![vec![0i64; n]; n];
    for i in 0..n {
        a[i][i] = 2;
    }
    for i in 0..n.saturating_sub(1) {
        a[i][i + 1] = -1;
        a[i + 1][i] = -1;
    }
    let mut d = vec![Frac::int(1); n];
    match typ {
        'A' => {}
        'B' => {
            if n < 2 {
                return Err("B_n needs n >= 2".into());
            }
            a[n - 1][n - 2] = -2;
            d[n - 1] = Frac::new(1, 2);
        }
        'C' => {
            if n < 2 {
                return Err("C_n needs n >= 2 (LiE rejects C1 as well)".into());
            }
            a[n - 2][n - 1] = -2;
            for i in 0..n - 1 {
                d[i] = Frac::new(1, 2);
            }
        }
        'D' => {
            if n < 3 {
                return Err("D_n needs n >= 3".into());
            }
            a[n - 2][n - 1] = 0;
            a[n - 1][n - 2] = 0;
            a[n - 3][n - 1] = -1;
            a[n - 1][n - 3] = -1;
        }
        'G' => {
            if n != 2 {
                return Err("only G2".into());
            }
            a = vec![vec![2, -3], vec![-1, 2]];
            d = vec![Frac::new(1, 3), Frac::int(1)];
        }
        _ => return Err(format!("Cartan type {typ} not supported (A-D, G2 only)")),
    }
    for i in 0..n {
        for j in 0..n {
            if Frac::int(a[i][j] as i128).mul(d[i]) != Frac::int(a[j][i] as i128).mul(d[j]) {
                return Err(format!("inconsistent Cartan data for {typ}{n}"));
            }
        }
    }
    Ok((a, d))
}

fn exact_inverse(a: &[Vec<i64>]) -> Result<Vec<Vec<Frac>>, String> {
    let n = a.len();
    let mut m: Vec<Vec<Frac>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| Frac::int(a[i][j] as i128))
                .chain((0..n).map(|j| Frac::int((i == j) as i128)))
                .collect()
        })
        .collect();
    for c in 0..n {
        let p = (c..n).find(|&r| !m[r][c].is_zero()).ok_or("singular Cartan matrix")?;
        m.swap(c, p);
        let piv = m[c][c];
        for x in m[c].iter_mut() {
            *x = x.div(piv);
        }
        for r in 0..n {
            if r != c && !m[r][c].is_zero() {
                let f = m[r][c];
                let row_c = m[c].clone();
                for (x, y) in m[r].iter_mut().zip(row_c.iter()) {
                    *x = x.sub(f.mul(*y));
                }
            }
        }
    }
    Ok(m.into_iter().map(|row| row[n..].to_vec()).collect())
}

pub struct Group {
    #[allow(dead_code)]
    pub name: String,
    pub rank: usize,
    alpha: Vec<Vec<i64>>,      // row i = alpha_i in Dynkin coordinates
    qi: Vec<Vec<i64>>,         // scaled symmetrized inverse Cartan form: (w, v) = w Qi v / L
    roots: Vec<Vec<i64>>,      // positive roots, Dynkin coordinates
    rq: Vec<Vec<i64>>,         // roots @ Qi
    rho_ip: Vec<i64>,          // (rho, alpha) scaled, per positive root
    hvec: Vec<i128>,           // height of a weight in simple-root coordinates, scaled by hden
    memo_domchar: HashMap<Weight, BTreeMap<Weight, i64>>,
    memo_dim: HashMap<Weight, i128>,
}

impl Group {
    fn new(name: &str) -> Result<Group, String> {
        let mut chars = name.chars();
        let typ = chars.next().ok_or("bad group name")?;
        let rank: usize = chars.as_str().parse().map_err(|_| format!("bad group name {name:?}"))?;
        if !('A'..='G').contains(&typ) || rank == 0 || rank > MAXR {
            return Err(format!("bad group name {name:?}"));
        }
        let (a, d) = cartan(typ, rank)?;
        let n = rank;
        let alpha: Vec<Vec<i64>> = (0..n).map(|i| (0..n).map(|j| a[j][i]).collect()).collect();
        let ainv = exact_inverse(&a)?;
        // (omega_i, omega_j) = (A^-1)_{ji} d_j, scaled to integers by L
        let q: Vec<Vec<Frac>> = (0..n).map(|i| (0..n).map(|j| ainv[j][i].mul(d[j])).collect()).collect();
        let mut l: i128 = 1;
        for row in &q {
            for x in row {
                l = lcm(l, x.d);
            }
        }
        let qi: Vec<Vec<i64>> = q.iter().map(|row| row.iter().map(|x| (x.n * l / x.d) as i64).collect()).collect();
        let roots = positive_roots(&a);
        let rq: Vec<Vec<i64>> = roots.iter().map(|r| (0..n).map(|j| (0..n).map(|k| r[k] * qi[k][j]).sum()).collect()).collect();
        let rho_ip: Vec<i64> = rq.iter().map(|row| row.iter().sum()).collect();
        // hvec_i = sum_j Ainv[j][i], scaled by the lcm of the denominators
        let mut hden: i128 = 1;
        for j in 0..n {
            for i in 0..n {
                hden = lcm(hden, ainv[j][i].d);
            }
        }
        let hvec: Vec<i128> = (0..n).map(|i| (0..n).map(|j| ainv[j][i].n * hden / ainv[j][i].d).sum()).collect();
        Ok(Group {
            name: name.to_string(),
            rank,
            alpha,
            qi,
            roots,
            rq,
            rho_ip,
            hvec,
            memo_domchar: HashMap::new(),
            memo_dim: HashMap::new(),
        })
    }

    fn ip(&self, w: &[i64], v: &[i64]) -> i128 {
        // scaled inner product w Qi v
        let n = self.rank;
        let mut s: i128 = 0;
        for i in 0..n {
            let mut t: i128 = 0;
            for j in 0..n {
                t += self.qi[i][j] as i128 * v[j] as i128;
            }
            s += w[i] as i128 * t;
        }
        s
    }

    pub fn dim(&mut self, lam: &[i64]) -> Result<i128, String> {
        if let Some(d) = self.memo_dim.get(lam) {
            return Ok(*d);
        }
        let lr: Vec<i64> = lam.iter().map(|x| x + 1).collect();
        let mut num: i128 = 1;
        let mut den: i128 = 1;
        for (row, &r) in self.rq.iter().zip(self.rho_ip.iter()) {
            let a: i128 = row.iter().zip(lr.iter()).map(|(x, y)| *x as i128 * *y as i128).sum();
            num = num.checked_mul(a).ok_or("dimension overflow")?;
            den = den.checked_mul(r as i128).ok_or("dimension overflow")?;
            let g = gcd(num, den);
            if g > 1 {
                num /= g;
                den /= g;
            }
        }
        if num % den != 0 {
            return Err("Weyl dimension formula not integral".into());
        }
        let d = num / den;
        if self.memo_dim.len() >= 8 * MEMO_CAP {
            self.memo_dim.clear();
        }
        self.memo_dim.insert(lam.to_vec(), d);
        Ok(d)
    }

    /// Reflect to the dominant chamber (the first negative coordinate each time); the sign
    /// of the Weyl element.
    fn to_dominant(&self, v: &mut Vec<i64>) -> i64 {
        let mut sign = 1i64;
        loop {
            match v.iter().position(|x| *x < 0) {
                None => return sign,
                Some(i) => {
                    let vi = v[i];
                    for j in 0..self.rank {
                        v[j] -= vi * self.alpha[i][j];
                    }
                    sign = -sign;
                }
            }
        }
    }

    /// The same on a fixed array (the hot loops).
    #[inline]
    fn to_dominant_w(&self, v: &mut W) -> i64 {
        let n = self.rank;
        let mut sign = 1i64;
        loop {
            let mut neg = n;
            for i in 0..n {
                if v[i] < 0 {
                    neg = i;
                    break;
                }
            }
            if neg == n {
                return sign;
            }
            let vi = v[neg];
            let al = &self.alpha[neg];
            for j in 0..n {
                v[j] -= vi * al[j];
            }
            sign = -sign;
        }
    }

    /// The contributions of one orbit (of the dominant weight `dom`) shifted by `shift` (rho, or
    /// lam + rho): every orbit weight plus the shift reflected to the dominant chamber with its
    /// sign, walls dropped, rho subtracted; the signed counts per resulting weight.  The orbit is
    /// traversed without being stored: a non-dominant weight's parent is its reflection at its
    /// first negative coordinate (which moves it up and ends at the dominant weight), so the
    /// children of w are the s_i w with w_i > 0 whose first negative coordinate is i; a
    /// depth-first walk visits every orbit weight once.
    fn orbit_contributions(&self, dom: &[i64], shift: &[i64], deadline: Option<Instant>) -> Result<FxHashMap<W, i64>, String> {
        let n = self.rank;
        let mut out: FxHashMap<W, i64> = FxHashMap::default();
        let mut stack: Vec<W> = Vec::with_capacity(64);
        let mut w0: W = [0; MAXR];
        w0[..n].copy_from_slice(dom);
        stack.push(w0);
        let mut count: u64 = 0;
        while let Some(w) = stack.pop() {
            count += 1;
            if count % 4096 == 0 {
                check_deadline(deadline)?;
            }
            // the contribution of w
            let mut v: W = [0; MAXR];
            for j in 0..n {
                v[j] = w[j] + shift[j];
            }
            let sign = self.to_dominant_w(&mut v);
            let mut inside = true;
            for j in 0..n {
                if v[j] <= 0 {
                    inside = false;
                    break;
                }
            }
            if inside {
                for j in 0..n {
                    v[j] -= 1;
                }
                *out.entry(v).or_insert(0) += sign;
            }
            // the children of w
            for i in 0..n {
                if w[i] > 0 {
                    let wi = w[i];
                    let al = &self.alpha[i];
                    let mut c: W = [0; MAXR];
                    let mut child = true;
                    for j in 0..n {
                        c[j] = w[j] - wi * al[j];
                        if j < i && c[j] < 0 {
                            child = false;
                            break;
                        }
                    }
                    if child {
                        stack.push(c);
                    }
                }
            }
        }
        out.retain(|_, c| *c != 0);
        Ok(out)
    }

    /// Dominant weights of V_lam with multiplicities (Freudenthal).
    fn domchar(&mut self, lam: &[i64], deadline: Option<Instant>) -> Result<BTreeMap<Weight, i64>, String> {
        if let Some(m) = self.memo_domchar.get(lam) {
            return Ok(m.clone());
        }
        let n = self.rank;
        let mut doms: HashSet<Weight> = HashSet::new();
        doms.insert(lam.to_vec());
        let mut frontier: Vec<Weight> = vec![lam.to_vec()];
        while !frontier.is_empty() {
            let mut keep: Vec<Weight> = Vec::new();
            for w in &frontier {
                for r in &self.roots {
                    let c: Weight = w.iter().zip(r.iter()).map(|(x, y)| x - y).collect();
                    if c.iter().all(|x| *x >= 0) && doms.insert(c.clone()) {
                        keep.push(c);
                    }
                }
            }
            frontier = keep;
        }
        // process in increasing height difference (lam - mu in simple-root coordinates), ties by the weight
        let mut order: Vec<(i128, Weight)> = doms
            .iter()
            .map(|mu| {
                let h: i128 = (0..n).map(|i| (lam[i] - mu[i]) as i128 * self.hvec[i]).sum();
                (h, mu.clone())
            })
            .collect();
        order.sort();
        let mut mult: HashMap<Weight, i64> = HashMap::new();
        mult.insert(lam.to_vec(), 1);
        let lr: Vec<i64> = lam.iter().map(|x| x + 1).collect();
        let norm_lr = self.ip(&lr, &lr);
        for (_, mu) in order.iter() {
            if mu.as_slice() == lam {
                continue;
            }
            check_deadline(deadline)?;
            let mr: Vec<i64> = mu.iter().map(|x| x + 1).collect();
            let denom = norm_lr - self.ip(&mr, &mr);
            let mut total: i128 = 0;
            for (ri, root) in self.roots.iter().enumerate() {
                let mut k: i64 = 1;
                loop {
                    let w: Weight = mu.iter().zip(root.iter()).map(|(x, y)| x + k * y).collect();
                    let mut wd = w.clone();
                    self.to_dominant(&mut wd);
                    match mult.get(&wd) {
                        None => break,
                        Some(mm) => {
                            let ip: i128 = w.iter().zip(self.rq[ri].iter()).map(|(x, y)| *x as i128 * *y as i128).sum();
                            total += *mm as i128 * ip;
                        }
                    }
                    k += 1;
                }
            }
            let num = 2 * total;
            if denom == 0 || num % denom != 0 {
                return Err("Freudenthal recursion not integral".into());
            }
            let v = num / denom;
            mult.insert(mu.clone(), v as i64);
        }
        let out: BTreeMap<Weight, i64> = mult.into_iter().collect();
        if self.memo_domchar.len() >= MEMO_CAP {
            self.memo_domchar.clear();       // the memo is a per-process cache, not a store: bounded memory
        }
        self.memo_domchar.insert(lam.to_vec(), out.clone());
        Ok(out)
    }

    /// LiE Vdecomp of a W-invariant weight multiset given by its dominant weights and their
    /// (virtual) multiplicities: the orbits in parallel, the counts merged with the multiplicities.
    fn vdecomp(&self, weights: &BTreeMap<Weight, BigInt>, deadline: Option<Instant>) -> Result<Poly, String> {
        let rho: Weight = vec![1; self.rank];
        let tasks: Vec<(&Weight, &BigInt)> = weights.iter().filter(|(_, m)| !m.is_zero()).collect();
        let parts: Vec<(FxHashMap<W, i64>, &BigInt)> = tasks
            .par_iter()
            .map(|(dom, m)| self.orbit_contributions(dom, &rho, deadline).map(|c| (c, *m)))
            .collect::<Result<Vec<_>, String>>()?;
        let n = self.rank;
        let mut out: Poly = BTreeMap::new();
        for (counts, m) in parts {
            for (w, c) in counts {
                let e = out.entry(w[..n].to_vec()).or_insert_with(BigInt::zero);
                *e += m * BigInt::from(c);
            }
        }
        out.retain(|_, v| !v.is_zero());
        Ok(out)
    }

    fn adams(&mut self, poly: &Poly, n: i64, deadline: Option<Instant>) -> Result<Poly, String> {
        if n == 1 {
            return Ok(poly.clone());
        }
        let mut weights: BTreeMap<Weight, BigInt> = BTreeMap::new();
        for (lam, c) in poly {
            let dc = self.domchar(lam, deadline)?;
            for (mu, m) in dc {
                let t: Weight = mu.iter().map(|x| n * x).collect();
                let e = weights.entry(t).or_insert_with(BigInt::zero);
                *e += c * BigInt::from(m);
            }
        }
        self.vdecomp(&weights, deadline)
    }

    /// Product of two virtual characters: for every pair of irreducibles the orbits of the
    /// dominant weights of the factor of smaller dimension, shifted by the larger factor plus
    /// rho (Klimyk); the tasks (pair, dominant weight) run in parallel, the counts are merged
    /// with the multiplicities and the pair's coefficient.
    fn tensor(&mut self, p: &Poly, q: &Poly, deadline: Option<Instant>) -> Result<Poly, String> {
        // serial preparation: dimensions and dominant characters through the memos
        struct Task {
            shift: Weight,
            dom: Weight,
            m: i64,
            pair: usize,
        }
        let mut coeffs: Vec<BigInt> = Vec::new();
        let mut tasks: Vec<Task> = Vec::new();
        for (a, ca) in p {
            for (b, cb) in q {
                let (big, small) = if self.dim(a)? >= self.dim(b)? { (a, b) } else { (b, a) };
                let shift: Weight = big.iter().map(|x| x + 1).collect();
                let dc = self.domchar(small, deadline)?;
                let pair = coeffs.len();
                coeffs.push(ca * cb);
                for (dom, m) in dc {
                    tasks.push(Task { shift: shift.clone(), dom, m, pair });
                }
            }
        }
        let parts: Vec<(FxHashMap<W, i64>, i64, usize)> = tasks
            .par_iter()
            .map(|t| self.orbit_contributions(&t.dom, &t.shift, deadline).map(|c| (c, t.m, t.pair)))
            .collect::<Result<Vec<_>, String>>()?;
        let n = self.rank;
        // merge the parts of one pair first (i64 sums), then multiply by the pair's coefficient once
        let mut per_pair: Vec<FxHashMap<W, i128>> = (0..coeffs.len()).map(|_| FxHashMap::default()).collect();
        for (counts, m, pair) in parts {
            let acc = &mut per_pair[pair];
            for (w, c) in counts {
                *acc.entry(w).or_insert(0) += m as i128 * c as i128;
            }
        }
        let mut out: Poly = BTreeMap::new();
        for (pair, acc) in per_pair.into_iter().enumerate() {
            let cab = &coeffs[pair];
            for (w, c) in acc {
                if c != 0 {
                    let e = out.entry(w[..n].to_vec()).or_insert_with(BigInt::zero);
                    *e += cab * BigInt::from(c);
                }
            }
        }
        out.retain(|_, v| !v.is_zero());
        Ok(out)
    }
}

fn positive_roots(a: &[Vec<i64>]) -> Vec<Vec<i64>> {
    // in simple-root coordinates first (as the Python engine), then Dynkin coordinates
    let n = a.len();
    let simple: Vec<Vec<i64>> = (0..n).map(|i| (0..n).map(|j| (i == j) as i64).collect()).collect();
    let mut roots: HashSet<Vec<i64>> = simple.iter().cloned().collect();
    let mut level: Vec<Vec<i64>> = simple.clone();
    while !level.is_empty() {
        let mut next: Vec<Vec<i64>> = Vec::new();
        for r in &level {
            // rd_i = sum_j r_j A[i][j]
            let rd: Vec<i64> = (0..n).map(|i| (0..n).map(|j| r[j] * a[i][j]).sum()).collect();
            for i in 0..n {
                let mut p = 0i64;
                let mut rr = r.clone();
                loop {
                    rr[i] -= 1;
                    if roots.contains(&rr) {
                        p += 1;
                    } else {
                        break;
                    }
                }
                if p - rd[i] > 0 {
                    let mut new = r.clone();
                    new[i] += 1;
                    if roots.insert(new.clone()) {
                        next.push(new);
                    }
                }
            }
        }
        level = next;
    }
    let mut rs: Vec<Vec<i64>> = roots.into_iter().collect();
    rs.sort_by(|x, y| (x.iter().sum::<i64>(), x).cmp(&(y.iter().sum::<i64>(), y)));
    rs.iter().map(|r| (0..n).map(|i| (0..n).map(|j| r[j] * a[i][j]).sum()).collect()).collect()
}

fn check_deadline(deadline: Option<Instant>) -> Result<(), String> {
    if let Some(d) = deadline {
        if Instant::now() > d {
            return Err("__timeout__".into());
        }
    }
    Ok(())
}

// --------------------------------------------------------------------------- #
// LiE polynomial strings and the print layout
// --------------------------------------------------------------------------- #
pub fn parse_poly(s: &str, rank: usize) -> Result<Poly, String> {
    let t: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    let mut out: Poly = BTreeMap::new();
    let mut pos = 0usize;
    let bytes = t.as_bytes();
    if t.is_empty() {
        return Err("not a LiE polynomial: ''".into());
    }
    while pos < bytes.len() {
        // signed coefficient
        let start = pos;
        while pos < bytes.len() && (bytes[pos] == b'+' || bytes[pos] == b'-') {
            pos += 1;
        }
        let sign_part = &t[start..pos];
        let neg = sign_part.bytes().filter(|b| *b == b'-').count() % 2 == 1;
        if sign_part.len() > 1 && start > 0 {
            // LiE writes "+-" only after a line break; both signs count
        }
        let dstart = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        if dstart == pos {
            return Err(format!("not a LiE polynomial: {:?}", &s[..s.len().min(80)]));
        }
        let mut c: BigInt = t[dstart..pos].parse().map_err(|_| "bad coefficient")?;
        if neg {
            c = -c;
        }
        if pos >= bytes.len() || bytes[pos] != b'X' {
            return Err(format!("not a LiE polynomial: {:?}", &s[..s.len().min(80)]));
        }
        pos += 1;
        if pos >= bytes.len() || bytes[pos] != b'[' {
            return Err(format!("not a LiE polynomial: {:?}", &s[..s.len().min(80)]));
        }
        let close = t[pos..].find(']').ok_or("unterminated weight")? + pos;
        let inner = &t[pos + 1..close];
        let w: Vec<i64> = if inner.is_empty() {
            Vec::new()
        } else {
            inner.split(',').map(|x| x.parse::<i64>().map_err(|_| "bad weight".to_string())).collect::<Result<_, _>>()?
        };
        if w.len() != rank {
            return Err("weight length does not match the group rank".into());
        }
        let e = out.entry(w).or_insert_with(BigInt::zero);
        *e += c;
        pos = close + 1;
    }
    out.retain(|_, v| !v.is_zero());
    Ok(out)
}

/// LiE print.c::print_poly/print_row for a polynomial in LiE's sorted order (ascending
/// lexicographic on the labels); the text after the banner line, newline-terminated.
pub fn print_poly(poly: &Poly, rank: usize) -> String {
    let mut rows: Vec<(Weight, BigInt)> = poly.iter().map(|(w, c)| (w.clone(), c.clone())).collect();
    if rows.is_empty() {
        rows.push((vec![0; rank], BigInt::zero()));
    }
    let widths: Vec<usize> = (0..rank).map(|j| rows.iter().map(|r| r.0[j].to_string().len()).max().unwrap_or(1)).collect();
    let cmax = rows.iter().map(|r| &r.1).max().unwrap();
    let cmin = rows.iter().map(|r| &r.1).min().unwrap();
    let coef_width = cmax.to_string().len().max(cmin.to_string().len());
    let mut out = String::new();

    fn row_text(lab: &[i64], widths: &[usize], mut col: usize, start_col: usize) -> (String, usize) {
        let mut s = String::from("[");
        col += 1;
        for (i, x) in lab.iter().enumerate() {
            let piece = format!("{:>width$}", x, width = widths[i]);
            col += piece.len();
            s.push_str(&piece);
            if i < lab.len() - 1 {
                s.push(',');
                col += 1;
                if col >= RIGHT_MARGIN {
                    s.push('\n');
                    s.push_str(&" ".repeat(start_col + 1));
                    col = start_col + 1;
                }
            }
        }
        s.push(']');
        col += 1;
        (s, col)
    }

    let mut col = LMARGIN;
    out.push_str(&" ".repeat(LMARGIN));
    let start = col;
    let c0 = format!("{:>width$}", rows[0].1.to_string(), width = coef_width);
    out.push_str(&c0);
    out.push('X');
    col += c0.len() + 1;
    let (s, c) = row_text(&rows[0].0, &widths, col, col - c0.len() - 1);
    col = c;
    out.push_str(&s);
    let rowsize = col - start;
    for (lab, cf) in rows.iter().skip(1) {
        let cs;
        if col + rowsize >= RIGHT_MARGIN {
            out.push_str(" +\n");
            out.push_str(&" ".repeat(LMARGIN));
            col = LMARGIN;
            cs = format!("{:>width$}", cf.to_string(), width = coef_width);
        } else {
            out.push_str(if cf.is_negative() { " -" } else { " +" });
            col += 2;
            cs = format!("{:>width$}", cf.abs().to_string(), width = coef_width);
        }
        out.push_str(&cs);
        out.push('X');
        col += cs.len() + 1;
        let (s, c) = row_text(lab, &widths, col, col - 1);
        col = c;
        out.push_str(&s);
    }
    out.push('\n');
    out
}

// --------------------------------------------------------------------------- #
// the runner: lcode -> stdout
// --------------------------------------------------------------------------- #
static GROUPS: Mutex<Option<HashMap<String, Group>>> = Mutex::new(None);
static POOL_INIT: std::sync::Once = std::sync::Once::new();

/// The rayon pool: LANDSCAPE_NATIVE_THREADS threads when set, rayon's default (the core count)
/// otherwise; built once.
fn init_pool() {
    POOL_INIT.call_once(|| {
        if let Ok(v) = std::env::var("LANDSCAPE_NATIVE_THREADS") {
            if let Ok(n) = v.trim().parse::<usize>() {
                if n >= 1 {
                    let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
                }
            }
        }
    });
}

fn poly_arg(s: &str, rank: usize) -> Result<Poly, String> {
    let t = s.trim();
    if t.starts_with('[') && t.ends_with(']') && !t[1..].contains('[') {
        let inner = &t[1..t.len() - 1];
        let w: Vec<i64> = if inner.is_empty() {
            Vec::new()
        } else {
            inner.split(',').map(|x| x.trim().parse::<i64>().map_err(|_| "bad weight".to_string())).collect::<Result<_, _>>()?
        };
        if w.len() != rank {
            return Err("weight length does not match the group rank".into());
        }
        let mut p = BTreeMap::new();
        p.insert(w, BigInt::from(1));
        return Ok(p);
    }
    parse_poly(t, rank)
}

enum Outcome {
    Ok(String),
    Unsupported(String),
    Timeout,
    Fail(String),
}

fn evaluate(expr: &str, deadline: Option<Instant>) -> Outcome {
    let e: String = expr.chars().filter(|c| *c != ' ').collect();
    let mut guard = GROUPS.lock().unwrap();
    let groups = guard.get_or_insert_with(HashMap::new);
    let result: Result<(Poly, usize), String> = (|| {
        if let Some(rest) = e.strip_prefix("Adams(") {
            let rest = rest.strip_suffix(')').ok_or("unsupported LiE expression")?;
            let c1 = rest.find(',').ok_or("unsupported LiE expression")?;
            let n: i64 = rest[..c1].parse().map_err(|_| "unsupported LiE expression".to_string())?;
            let c2 = rest.rfind(',').ok_or("unsupported LiE expression")?;
            if c2 <= c1 {
                return Err("unsupported LiE expression".into());
            }
            let lab = &rest[c1 + 1..c2];
            let gname = &rest[c2 + 1..];
            if !groups.contains_key(gname) {
                groups.insert(gname.to_string(), Group::new(gname)?);
            }
            let g = groups.get_mut(gname).unwrap();
            let p = poly_arg(lab, g.rank)?;
            let r = g.adams(&p, n, deadline)?;
            Ok((r, g.rank))
        } else if let Some(rest) = e.strip_prefix("tensor(") {
            let rest = rest.strip_suffix(')').ok_or("unsupported LiE expression")?;
            let c2 = rest.rfind(',').ok_or("unsupported LiE expression")?;
            let gname = &rest[c2 + 1..];
            let args = &rest[..c2];
            let mut depth = 0i32;
            let mut split: Option<usize> = None;
            for (i, ch) in args.char_indices() {
                match ch {
                    '[' => depth += 1,
                    ']' => depth -= 1,
                    ',' if depth == 0 => {
                        if split.is_some() {
                            return Err("tensor with more than two characters".into());
                        }
                        split = Some(i);
                    }
                    _ => {}
                }
            }
            let split = split.ok_or("unsupported tensor arguments")?;
            if !groups.contains_key(gname) {
                groups.insert(gname.to_string(), Group::new(gname)?);
            }
            let g = groups.get_mut(gname).unwrap();
            let p = poly_arg(&args[..split], g.rank)?;
            let q = poly_arg(&args[split + 1..], g.rank)?;
            let r = g.tensor(&p, &q, deadline)?;
            Ok((r, g.rank))
        } else {
            Err("unsupported LiE expression".into())
        }
    })();
    match result {
        Ok((poly, rank)) => Outcome::Ok(print_poly(&poly, rank)),
        Err(m) if m == "__timeout__" => Outcome::Timeout,
        Err(m) if m.starts_with("unsupported") || m.starts_with("tensor with") => Outcome::Unsupported(m),
        Err(m) => Outcome::Fail(m),
    }
}

/// lie_run(lcode, timeout) -> stdout, as LiE would print it; NotImplementedError for an lcode
/// outside the supported forms, subprocess.TimeoutExpired past the deadline.
#[pyfunction]
#[pyo3(signature = (lcode, timeout=None))]
pub fn lie_run(py: Python<'_>, lcode: &str, timeout: Option<f64>) -> PyResult<String> {
    let deadline = timeout.map(|t| Instant::now() + std::time::Duration::from_secs_f64(t.max(0.0)));
    let lines: Vec<&str> = lcode.split('\n').map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
    if lines.is_empty() {
        return Err(PyNotImplementedError::new_err("empty lcode"));
    }
    let mut banner: Option<String> = None;
    let mut body: Vec<&str> = Vec::new();
    for ln in lines {
        if let Some(rest) = ln.strip_prefix("maxnodes") {
            let n = rest.trim();
            if n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) {
                return Err(PyNotImplementedError::new_err("unsupported maxnodes line"));
            }
            if banner.is_some() {
                return Err(PyNotImplementedError::new_err("two maxnodes lines"));
            }
            banner = Some(format!("New tree space with maximum number of nodes: {n}.\n"));
            continue;
        }
        if let Some(rest) = ln.strip_prefix("maxobjects") {
            let n = rest.trim();
            if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
        }
        body.push(ln);
    }
    let banner = banner.ok_or_else(|| PyNotImplementedError::new_err("lcode without a maxnodes line"))?;
    if body.len() == 1 && !body[0].is_empty() && body[0].bytes().all(|b| b.is_ascii_digit()) {
        return Ok(format!("{banner}{}{}\n", " ".repeat(LMARGIN), body[0]));
    }
    let expr: &str = if body.len() == 1 {
        body[0]
    } else if body.len() == 2 && body[0].starts_with("res=") && body[0].ends_with(';') && body[1] == "print(res);" {
        &body[0][4..body[0].len() - 1]
    } else {
        return Err(PyNotImplementedError::new_err(format!("unsupported lcode: {:?}", &lcode[..lcode.len().min(120)])));
    };
    init_pool();
    let outcome = py.allow_threads(|| evaluate(expr, deadline));
    match outcome {
        Outcome::Ok(text) => {
            if let Some(d) = deadline {
                if Instant::now() > d {
                    return Err(timeout_error(py, timeout));
                }
            }
            Ok(format!("{banner}{text}"))
        }
        Outcome::Timeout => Err(timeout_error(py, timeout)),
        Outcome::Unsupported(m) => Err(PyNotImplementedError::new_err(m)),
        Outcome::Fail(m) => Err(PyValueError::new_err(m)),
    }
}

fn timeout_error(py: Python<'_>, timeout: Option<f64>) -> PyErr {
    match py.import("subprocess").and_then(|m| m.getattr("TimeoutExpired")).and_then(|cls| cls.call1(("landscape_native.lie_run", timeout.unwrap_or(0.0)))) {
        Ok(exc) => PyErr::from_value(exc),
        Err(e) => e,
    }
}

/// lie_dim(group, label) -> int, the Weyl dimension (a check of the group data).
#[pyfunction]
pub fn lie_dim(group: &str, label: Vec<i64>) -> PyResult<i128> {
    let mut guard = GROUPS.lock().unwrap();
    let groups = guard.get_or_insert_with(HashMap::new);
    if !groups.contains_key(group) {
        groups.insert(group.to_string(), Group::new(group).map_err(PyValueError::new_err)?);
    }
    let g = groups.get_mut(group).unwrap();
    if label.len() != g.rank {
        return Err(PyValueError::new_err("weight length does not match the group rank"));
    }
    g.dim(&label).map_err(PyValueError::new_err)
}
