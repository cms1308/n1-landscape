//! The singlet multiplicity of a character product in the extension (step 56b), in place of a call
//! back into the Python projector for every distinct product of the expansion engine.
//!
//! The same quantity as `landscape.singlet.ProductProjector.multiplicity`: per node, the product of
//! the characters psi^k(chi_label) of the node's slots (each slot as often as its exponent) is cut
//! into two parts, each part's product decomposed into irreducibles, and the trivial multiplicity is
//! sum_lambda a_lambda b_{lambda*} (lambda* the conjugate highest weight, `landscape.lie.conjugate`);
//! the multiplicity of the product is the product over the nodes.  A part's decomposition is built
//! factor by factor with the Brauer-Klimyk rule: for a virtual character P = sum c_lambda V_lambda
//! and a factor F given by its weights mu with multiplicities m(mu),
//! P (x) F = sum_lambda c_lambda sum_mu m(mu) eps(w) V_{w(lambda + mu + rho) - rho}, w taking
//! lambda + mu + rho to the dominant chamber (a weight on a wall contributing nothing).  The weights
//! of psi^k(chi_label) are k times the weights of chi_label: the dominant character of the native
//! character engine (`lie.rs`, R43) and the Weyl orbits of its weights.  The parts are the first half
//! of the factors in slot order and the rest; every product of factors met on the way is memoized by
//! its exponents for the expansion (a part of one product is often a prefix of another's), the memo
//! emptied when it holds more than MEMO_TERMS irreducible terms.  An exponent outside 0..=255, a rank
//! above MAXR (16, the character engine's) or a coefficient beyond 128 bits returns None, and the caller asks the Python
//! projector.

use crate::lie::Group;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::time::Instant;

const MEMO_TERMS: usize = 4_000_000;
const MAXR: usize = 16;
type W = [i32; MAXR];
type Chi = FxHashMap<W, i128>;

fn conjugate(typ: char, rank: usize, lam: &W) -> W {
    // landscape.lie.conjugate: the Dynkin-diagram automorphism of the dual (LiE numbering)
    let mut out = *lam;
    match typ {
        'A' => {
            for i in 0..rank {
                out[i] = lam[rank - 1 - i];
            }
        }
        'D' if rank % 2 == 1 => out.swap(rank - 2, rank - 1),
        'E' if rank == 6 => {
            let p = [5usize, 1, 4, 3, 2, 0];
            for i in 0..6 {
                out[i] = lam[p[i]];
            }
        }
        _ => {}
    }
    out
}

struct Node {
    typ: char,
    rank: usize,
    alpha: Vec<W>,                      // alpha_i in Dynkin coordinates
    slots: Vec<usize>,                  // the global slot indices of this node, in slot order
    weights: Vec<Vec<(W, i128)>>,       // per local slot: the weights of psi^k(chi_label) with multiplicities
    hvec: Vec<i128>,                    // the scaled height of each Dynkin component (lie.rs Group::hvec)
    arena: Vec<Chi>,                    // memoized products (irreducible decompositions)
    memo: FxHashMap<Vec<u8>, usize>,    // local exponents -> arena index
    terms: usize,
}

impl Node {
    /// w(v) - rho with the sign of w for v = lambda + mu + rho brought to the dominant chamber;
    /// None on a wall.
    #[inline]
    fn dominant(&self, mut v: W) -> Option<(W, i128)> {
        let mut sign = 1i128;
        loop {
            let mut neg = None;
            for i in 0..self.rank {
                if v[i] == 0 {
                    return None;
                }
                if v[i] < 0 && neg.is_none() {
                    neg = Some(i);
                }
            }
            match neg {
                None => {
                    for x in v.iter_mut().take(self.rank) {
                        *x -= 1;
                    }
                    return Some((v, sign));
                }
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

    /// P (x) the factor of local slot j (Brauer-Klimyk); None on an overflow.
    fn times_factor(&self, p: &Chi, j: usize) -> Option<Chi> {
        let mut out: Chi = FxHashMap::default();
        for (lam, c) in p.iter() {
            for (mu, m) in self.weights[j].iter() {
                let mut v = [0i32; MAXR];
                for i in 0..self.rank {
                    v[i] = lam[i] + mu[i] + 1;
                }
                if let Some((w, s)) = self.dominant(v) {
                    let add = c.checked_mul(*m)?.checked_mul(s)?;
                    let e = out.entry(w).or_insert(0);
                    *e = e.checked_add(add)?;
                }
            }
        }
        out.retain(|_, x| *x != 0);
        Some(out)
    }

    /// The arena index of the product with local exponents `c` (not all zero); None on an overflow.
    fn product(&mut self, c: &[u8]) -> Option<usize> {
        if let Some(&i) = self.memo.get(c) {
            return Some(i);
        }
        let j = c.iter().rposition(|x| *x > 0).expect("a nonzero exponent");
        let mut prev = c.to_vec();
        prev[j] -= 1;
        let p = if prev.iter().all(|x| *x == 0) {
            let mut one: Chi = FxHashMap::default();
            one.insert([0i32; MAXR], 1);
            self.times_factor(&one, j)?
        } else {
            let i = self.product(&prev)?;
            self.times_factor(&self.arena[i], j)?
        };
        self.terms += p.len();
        self.arena.push(p);
        let i = self.arena.len() - 1;
        self.memo.insert(c.to_vec(), i);
        Some(i)
    }

    fn multiplicity(&mut self, c: &[u8]) -> Option<i128> {
        let total: usize = c.iter().map(|x| *x as usize).sum();
        if total == 0 {
            return Some(1);
        }
        if self.terms > MEMO_TERMS {
            self.arena.clear();
            self.memo.clear();
            self.terms = 0;
        }
        if total == 1 {
            let i = self.product(c)?;
            return Some(self.arena[i].get(&[0i32; MAXR]).copied().unwrap_or(0));
        }
        // the first half of the factors in slot order, and the rest
        let mut first = vec![0u8; c.len()];
        let mut need = total / 2;
        for (j, x) in c.iter().enumerate() {
            if need == 0 {
                break;
            }
            let take = (*x as usize).min(need);
            first[j] = take as u8;
            need -= take;
        }
        let second: Vec<u8> = c.iter().zip(first.iter()).map(|(a, b)| a - b).collect();
        let ia = self.product(&first)?;
        let ib = self.product(&second)?;
        let (a, b) = if self.arena[ia].len() <= self.arena[ib].len() { (ia, ib) } else { (ib, ia) };
        let mut s: i128 = 0;
        for (lam, x) in self.arena[a].iter() {
            if let Some(y) = self.arena[b].get(&conjugate(self.typ, self.rank, lam)) {
                s = s.checked_add(x.checked_mul(*y)?)?;
            }
        }
        Some(s)
    }
}

/// The Weyl orbit of a dominant weight (Dynkin coordinates).
fn orbit(dom: &W, rank: usize, alpha: &[W]) -> Vec<W> {
    let mut seen: FxHashMap<W, ()> = FxHashMap::default();
    let mut stack = vec![*dom];
    seen.insert(*dom, ());
    while let Some(v) = stack.pop() {
        for i in 0..rank {
            if v[i] != 0 {
                let mut u = v;
                for j in 0..rank {
                    u[j] -= v[i] * alpha[i][j];
                }
                if seen.insert(u, ()).is_none() {
                    stack.push(u);
                }
            }
        }
    }
    seen.into_keys().collect()
}

pub(crate) struct Projector {
    nodes: Vec<Node>,
}

impl Projector {
    /// `groups[node]` = (LiE group name, type letter, rank) of the node's store; `slots[j]` =
    /// (node, Dynkin label, Adams index).  None when a node's group is outside the engine.
    pub(crate) fn new(groups: &[(String, String, usize)], slots: &[(usize, Vec<i64>, i64)], deadline: Option<Instant>)
                      -> Result<Option<Projector>, String> {
        let mut lie: Vec<Group> = Vec::with_capacity(groups.len());
        let mut nodes = Vec::with_capacity(groups.len());
        for (lie_name, typ, rank) in groups {
            let g = match Group::new(lie_name) {
                Ok(g) => g,
                Err(_) => return Ok(None),
            };
            if g.rank > MAXR || g.rank != *rank {
                return Ok(None);
            }
            let mut alpha = vec![[0i32; MAXR]; g.rank];
            for i in 0..g.rank {
                for j in 0..g.rank {
                    alpha[i][j] = g.alpha[i][j] as i32;
                }
            }
            nodes.push(Node { typ: typ.chars().next().unwrap_or('?'), rank: *rank, alpha, slots: Vec::new(), weights: Vec::new(),
                              hvec: g.hvec.clone(), arena: Vec::new(), memo: FxHashMap::default(), terms: 0 });
            lie.push(g);
        }
        for (j, (node, label, k)) in slots.iter().enumerate() {
            if *node >= nodes.len() || label.len() != nodes[*node].rank {
                return Ok(None);
            }
            let dc = lie[*node].domchar(label, deadline)?;
            let nd = &mut nodes[*node];
            let mut ws: FxHashMap<W, i128> = FxHashMap::default();
            for (dom, m) in dc {
                let mut d = [0i32; MAXR];
                for i in 0..nd.rank {
                    d[i] = i32::try_from(dom[i]).map_err(|_| "weight out of range".to_string())?;
                }
                for mut w in orbit(&d, nd.rank, &nd.alpha) {
                    for x in w.iter_mut().take(nd.rank) {
                        *x *= *k as i32;
                    }
                    *ws.entry(w).or_insert(0) += m as i128;
                }
            }
            nd.slots.push(j);
            nd.weights.push(ws.into_iter().filter(|(_, m)| *m != 0).collect());
        }
        Ok(Some(Projector { nodes }))
    }

    /// The singlet multiplicity of the product whose slot exponents are `cexps`; None when the
    /// extension cannot give it (an exponent outside 0..=255, a value beyond 128 bits).
    pub(crate) fn multiplicity(&mut self, cexps: &[i32], _deadline: Option<Instant>) -> Result<Option<i128>, String> {
        let mut out: i128 = 1;
        for nd in self.nodes.iter_mut() {
            let mut c = Vec::with_capacity(nd.slots.len());
            for &j in &nd.slots {
                let x = cexps[j];
                if !(0..=255).contains(&x) {
                    return Ok(None);
                }
                c.push(x as u8);
            }
            match nd.multiplicity(&c) {
                Some(0) => return Ok(Some(0)),
                Some(m) => match out.checked_mul(m) {
                    Some(v) => out = v,
                    None => return Ok(None),
                },
                None => return Ok(None),
            }
        }
        Ok(Some(out))
    }
}

/// The highest-weight arithmetic the engine's irreducible mode (series.rs, step 61) needs.
pub(crate) type Lam = W;
pub(crate) const LAM_RANK: usize = MAXR;

impl Projector {
    pub(crate) fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
    pub(crate) fn rank(&self, node: usize) -> usize {
        self.nodes[node].rank
    }
    /// (node, local slot) of the global slot j.
    pub(crate) fn locate(&self, j: usize) -> Option<(usize, usize)> {
        for (i, nd) in self.nodes.iter().enumerate() {
            if let Some(l) = nd.slots.iter().position(|x| *x == j) {
                return Some((i, l));
            }
        }
        None
    }
    /// V_lam (x) F^x for the factor F of `local` on `node` (Brauer-Klimyk, x times): (highest weight, multiplicity);
    /// None on an overflow.
    pub(crate) fn tensor(&self, node: usize, lam: &Lam, local: usize, x: u32) -> Option<Vec<(Lam, i128)>> {
        let nd = &self.nodes[node];
        let mut chi: Chi = FxHashMap::default();
        chi.insert(*lam, 1);
        for _ in 0..x {
            chi = nd.times_factor(&chi, local)?;
        }
        Some(chi.into_iter().collect())
    }
    /// The scaled height sum_c hvec_c w_c of a weight on `node`.
    pub(crate) fn height(&self, node: usize, w: &Lam) -> i128 {
        let nd = &self.nodes[node];
        (0..nd.rank).map(|c| nd.hvec[c] * w[c] as i128).sum()
    }
    /// The largest scaled height over the weights of `local` on `node`.
    pub(crate) fn max_height(&self, node: usize, local: usize) -> i128 {
        self.nodes[node].weights[local].iter().map(|(w, _)| self.height(node, w)).max().unwrap_or(0)
    }
    pub(crate) fn hvec(&self, node: usize) -> &[i128] {
        &self.nodes[node].hvec
    }
}

// --------------------------------------------------------------------------- //
// the singlets of a product of symmetric powers (step 59)
// --------------------------------------------------------------------------- //
/// The partitions of p (parts in decreasing order).
fn partitions(p: u32) -> Vec<Vec<u32>> {
    fn rec(rest: u32, max: u32, cur: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
        if rest == 0 {
            out.push(cur.clone());
            return;
        }
        for part in (1..=rest.min(max)).rev() {
            cur.push(part);
            rec(rest - part, part, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    rec(p, p, &mut Vec::new(), &mut out);
    out
}

fn factorial(n: u32) -> Option<i128> {
    (1..=n as i128).try_fold(1i128, |a, b| a.checked_mul(b))
}

/// p!/z_mu, the number of permutations of cycle type mu (z_mu = prod_j j^{m_j} m_j!).
fn class_size(mu: &[u32], p: u32) -> Option<i128> {
    let mut z: i128 = 1;
    let mut j = 0;
    while j < mu.len() {
        let part = mu[j];
        let mut m = 0u32;
        while j < mu.len() && mu[j] == part {
            m += 1;
            j += 1;
        }
        z = z.checked_mul((part as i128).checked_pow(m)?)?.checked_mul(factorial(m)?)?;
    }
    Some(factorial(p)? / z)
}

/// sym_singlet(groups, factors) -> the number of gauge singlets of (x)_f Sym^{p_f}(R_f), or None when the projector cannot
/// give it (a group outside it, an overflow).  `groups[i]` = (LiE group name, type letter, rank) of node i; `factors[f]` =
/// (the Dynkin label of R_f on every node, p_f).  Newton's formula Sym^p = sum_{mu |- p} z_mu^{-1} prod_j psi^{mu_j}: Adams
/// operations act node by node on a representation of the product group, so every term of the expansion is an external
/// product over the nodes of products of psi^k(chi_label), whose trivial multiplicity is the product of the per-node
/// multiplicities (`Projector::multiplicity`, as in the expansion engine); the sum is taken over the choices of a partition
/// for every factor with the weights prod_f p_f!/z_{mu_f} and divided by prod_f p_f! at the end, an integer.  The same
/// number as LiE's trivial coefficient of tensor(sym_tensor(p_1,R_1,G), ..., G) (`landscape.model.singlet_multiplicity_lie`).
#[pyfunction]
pub fn sym_singlet(py: Python<'_>, groups: Vec<(String, String, usize)>, factors: Vec<(Vec<Vec<i64>>, u32)>) -> PyResult<Option<i128>> {
    py.allow_threads(|| -> Result<Option<i128>, String> {
        // the slots: (node, label, Adams index) for every node on which a factor is charged, k = 1..=p
        let mut slots: Vec<(usize, Vec<i64>, i64)> = Vec::new();
        let mut index: FxHashMap<(usize, Vec<i64>, i64), usize> = FxHashMap::default();
        for (labels, p) in &factors {
            if labels.len() != groups.len() {
                return Err("a factor's labels do not match the nodes".into());
            }
            for (i, lab) in labels.iter().enumerate() {
                if lab.iter().all(|x| *x == 0) {
                    continue;
                }
                for k in 1..=(*p as i64) {
                    let key = (i, lab.clone(), k);
                    if !index.contains_key(&key) {
                        index.insert(key.clone(), slots.len());
                        slots.push(key);
                    }
                }
            }
        }
        if slots.is_empty() {
            return Ok(Some(1));
        }
        let mut proj = match Projector::new(&groups, &slots, None)? {
            Some(p) => p,
            None => return Ok(None),
        };
        let parts: Vec<Vec<(Vec<u32>, i128)>> = factors
            .iter()
            .map(|(_, p)| partitions(*p).into_iter().map(|mu| { let c = class_size(&mu, *p); (mu, c) }).collect::<Vec<_>>())
            .map(|v| v.into_iter().map(|(mu, c)| c.map(|c| (mu, c))).collect::<Option<Vec<_>>>())
            .collect::<Option<Vec<_>>>()
            .ok_or("overflow in the class sizes")?;
        let mut den: i128 = 1;
        for (_, p) in &factors {
            den = den.checked_mul(factorial(*p).ok_or("overflow")?).ok_or("overflow")?;
        }
        let mut num: i128 = 0;
        let mut choice = vec![0usize; factors.len()];
        loop {
            let mut cexps = vec![0i32; slots.len()];
            let mut weight: i128 = 1;
            for (f, (labels, _)) in factors.iter().enumerate() {
                let (mu, c) = &parts[f][choice[f]];
                weight = match weight.checked_mul(*c) { Some(w) => w, None => return Ok(None) };
                for (i, lab) in labels.iter().enumerate() {
                    if lab.iter().all(|x| *x == 0) {
                        continue;
                    }
                    for part in mu {
                        cexps[index[&(i, lab.clone(), *part as i64)]] += 1;
                    }
                }
            }
            let m = match proj.multiplicity(&cexps, None)? {
                Some(m) => m,
                None => return Ok(None),
            };
            num = match weight.checked_mul(m).and_then(|x| num.checked_add(x)) { Some(v) => v, None => return Ok(None) };
            // the next choice (an odometer over the factors' partitions)
            let mut f = 0;
            loop {
                if f == choice.len() {
                    if num % den != 0 {
                        return Err(format!("non-integral singlet count {num}/{den}"));
                    }
                    return Ok(Some(num / den));
                }
                choice[f] += 1;
                if choice[f] < parts[f].len() {
                    break;
                }
                choice[f] = 0;
                f += 1;
            }
        }
    })
    .map_err(PyValueError::new_err)
}
