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
//! above MAXR or a coefficient beyond 128 bits returns None, and the caller asks the Python projector.

use crate::lie::Group;
use rustc_hash::FxHashMap;
use std::time::Instant;

const MEMO_TERMS: usize = 4_000_000;
const MAXR: usize = 8;
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
                              arena: Vec::new(), memo: FxHashMap::default(), terms: 0 });
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
