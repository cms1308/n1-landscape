//! The native expansion engine (step 39f; refined in steps 44, 54 and 56b): the exponential of the
//! explicit itotal of `landscape.form.program`, truncated at the t-power limit and at `max_order`
//! powers, in place of the FORM run -- the same polynomial FORM prints as `result`, and from it the
//! rows of the field-resolved expansion.
//!
//! A monomial is an exponent vector [t, s, r, y, f_1..f_n, c_1..c_m] (`form.Series`); the product of
//! two monomials adds the vectors.  With P_1 = itotal and P_k = P_{k-1} itotal / k, the result is
//! 1 + sum_k P_k for k = 1..max_order, every product whose t-power exceeds the limit dropped before
//! it is formed (the terms of itotal sorted by t-power, the multiplication of a term of P_{k-1}
//! stopping at the bound) -- the products FORM's bounded loop generates and keeps, and none of the
//! ones it discards.  The products of the terms of P_{k-1} run in parallel (rayon), the partial maps
//! merged; a deadline is checked inside the loops and raises subprocess.TimeoutExpired; a power
//! beyond the monomial cap raises "capacity" and the caller falls back to FORM.
//!
//! Step 56b (extension 0.8.0), the memory of the engine:
//!   * the rows are accumulated from every power as it is formed; no map of the whole exponential
//!     is kept (the monomial mode, the comparison with FORM, keeps it), so the cap applies to one
//!     power;
//!   * a monomial is a fixed-width array of machine words: each exponent a bit field with an offset,
//!     its width from a bound valid for every monomial within the t-limit (a field is at most the
//!     t-limit times the largest ratio of that exponent to the t-power over the letters, and at most
//!     max_order times the largest letter), so that the product of two monomials is one addition and
//!     one subtraction per word with no carry between the fields;
//!   * a coefficient is a reduced rational of two `i128`, promoted in place to arbitrary precision
//!     when an operation overflows (step 54 ran the whole expansion a second time instead);
//!   * the singlet multiplicity of a character product is computed in the extension (`project.rs`),
//!     the Python projector (`lookup`) its fallback and, with LANDSCAPE_NATIVE_VERIFY_MULT=1, the
//!     check of every value.
//!
//! The refinements of step 44, each a parameter, off by default unless the caller sets it:
//!   * `basis` (the flavor projection above t^6): a monomial whose milli exponent -- from its
//!     t, s, r sum, the rule of the rows -- exceeds `project_above` (6000) holds the flavor
//!     exponents basis . markers in place of the markers, [t, s, r, y, x_1..x_rank, c_1..c_m].
//!     The projection is linear on the exponents, so a product is projected as soon as its milli
//!     crosses the bound and a projected factor projects the product; the terms of itotal and of
//!     every power are kept in two lists, field-resolved and flavor-refined, and the rows come in
//!     the two kinds (`post::emit_rows`).  Weights are positive, so a field-resolved product arises
//!     from field-resolved factors only: the field-resolved part is complete through t^6.
//!   * `coef64`: the 128-bit arithmetic without a 128-bit division where both operands fit in 62
//!     bits (a binary gcd on 64-bit integers) and without any where a denominator is one.
//!   * `exact`: a product whose milli exceeds `milli_limit` (1000 t_order) is not formed; the
//!     integer t-power bound stays the outer filter.
//!
//! `expand_series(..., monomials=true)` returns the polynomial itself, (numerator, denominator,
//! exponents) per monomial, for the comparison with FORM's parsed output (without a basis and
//! without `exact`, the polynomial FORM prints); `monomials=false` returns the rows of the
//! field-resolved expansion (as a `FieldResolvedRows` object when `native_rows` is set; with a
//! basis, the object only): the terms above t_order dropped by their milli exponent,
//! coefficient x singlet multiplicity summed per (milli, y power, markers) or per (milli, y power,
//! flavor).

use crate::project::{Lam, Projector, LAM_RANK};
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

type Coef = (i128, i128);

/// A row value beyond 128 bits after the projection: the caller's FORM path follows.
const COEF_OVERFLOW: &str = "coefficient overflow";

// --------------------------------------------------------------------------- //
// coefficients
// --------------------------------------------------------------------------- //
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

/// Binary gcd on 64-bit integers (no division).
#[inline]
fn gcd64(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            return a << shift;
        }
    }
}

#[inline]
fn fits62(x: i128) -> bool {
    x.unsigned_abs() < (1u128 << 62)
}

#[inline]
fn reduce(n: i128, d: i128, fast: bool) -> Result<Coef, String> {
    if d == 0 {
        return Err("zero denominator".into());
    }
    if fast {
        if d == 1 {
            return Ok((n, 1));
        }
        if n == 0 {
            return Ok((0, 1));
        }
        if fits62(n) && fits62(d) {
            let g = gcd64(n.unsigned_abs() as u64, d.unsigned_abs() as u64) as i128;
            let (mut n, mut d) = if g > 1 { (n / g, d / g) } else { (n, d) };
            if d < 0 {
                n = -n;
                d = -d;
            }
            return Ok((n, d));
        }
    }
    let g = gcd(n, d);
    let (mut n, mut d) = if g > 1 { (n / g, d / g) } else { (n, d) };
    if d < 0 {
        n = -n;
        d = -d;
    }
    Ok((n, d))
}

#[inline]
fn mul(a: Coef, b: Coef, fast: bool) -> Option<Coef> {
    let n = a.0.checked_mul(b.0)?;
    let d = a.1.checked_mul(b.1)?;
    if fast && d == 1 {
        return Some((n, 1));
    }
    reduce(n, d, fast).ok()
}

#[inline]
fn add(a: Coef, b: Coef, fast: bool) -> Option<Coef> {
    if a.1 == b.1 {
        return reduce(a.0.checked_add(b.0)?, a.1, fast).ok();
    }
    let n = a.0.checked_mul(b.1)?.checked_add(b.0.checked_mul(a.1)?)?;
    let d = a.1.checked_mul(b.1)?;
    reduce(n, d, fast).ok()
}

/// An arbitrary-precision reduced rational (denominator positive).
#[derive(Clone, Debug, PartialEq)]
struct Big {
    n: BigInt,
    d: BigInt,
}

impl Big {
    fn reduced(mut n: BigInt, mut d: BigInt) -> Result<Big, String> {
        if d.is_zero() {
            return Err("zero denominator".into());
        }
        if n.is_zero() {
            return Ok(Big { n, d: BigInt::one() });
        }
        let g = n.gcd(&d);
        if !g.is_one() {
            n /= &g;
            d /= &g;
        }
        if d.is_negative() {
            n = -n;
            d = -d;
        }
        Ok(Big { n, d })
    }
}

/// The engine's coefficient: a reduced rational of two `i128`, or, once an operation on it has
/// overflowed, of two `BigInt` (brought back to `i128` whenever the reduced value fits).
#[derive(Clone, Debug)]
enum Q {
    S(i128, i128),
    B(Box<Big>),
}

impl Q {
    fn from_pair(n: i128, d: i128) -> Result<Q, String> {
        reduce(n, d, false).map(|(n, d)| Q::S(n, d))
    }
    fn one() -> Q {
        Q::S(1, 1)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        match self {
            Q::S(n, _) => *n == 0,
            Q::B(b) => b.n.is_zero(),
        }
    }
    fn big(&self) -> Big {
        match self {
            Q::S(n, d) => Big { n: BigInt::from(*n), d: BigInt::from(*d) },
            Q::B(b) => (**b).clone(),
        }
    }
    fn of_big(b: Big) -> Q {
        match (b.n.to_i128(), b.d.to_i128()) {
            (Some(n), Some(d)) => Q::S(n, d),
            _ => Q::B(Box::new(b)),
        }
    }
    #[inline]
    fn times(&self, o: &Q, fast: bool) -> Result<Q, String> {
        if let (Q::S(a, b), Q::S(c, d)) = (self, o) {
            if let Some((n, d)) = mul((*a, *b), (*c, *d), fast) {
                return Ok(Q::S(n, d));
            }
        }
        let (x, y) = (self.big(), o.big());
        Ok(Q::of_big(Big::reduced(&x.n * &y.n, &x.d * &y.d)?))
    }
    #[inline]
    fn plus(&self, o: &Q, fast: bool) -> Result<Q, String> {
        if let (Q::S(a, b), Q::S(c, d)) = (self, o) {
            if let Some((n, d)) = add((*a, *b), (*c, *d), fast) {
                return Ok(Q::S(n, d));
            }
        }
        let (x, y) = (self.big(), o.big());
        if x.d == y.d {
            return Ok(Q::of_big(Big::reduced(&x.n + &y.n, x.d)?));
        }
        Ok(Q::of_big(Big::reduced(&x.n * &y.d + &y.n * &x.d, &x.d * &y.d)?))
    }
    fn over(&self, k: i128, fast: bool) -> Result<Q, String> {
        if let Q::S(n, d) = self {
            if let Some(dk) = d.checked_mul(k) {
                return reduce(*n, dk, fast).map(|(n, d)| Q::S(n, d));
            }
        }
        let x = self.big();
        Ok(Q::of_big(Big::reduced(x.n, &x.d * BigInt::from(k))?))
    }
    fn times_int(&self, m: i128) -> Result<Q, String> {
        self.times(&Q::S(m, 1), false)
    }
    /// The value as two `i128`, or COEF_OVERFLOW.
    fn pair(&self) -> Result<Coef, String> {
        match self {
            Q::S(n, d) if within_test_bits(*n) && within_test_bits(*d) => Ok((*n, *d)),
            _ => Err(COEF_OVERFLOW.to_string()),
        }
    }
    fn big_pair(&self) -> (BigInt, BigInt) {
        let b = self.big();
        (b.n, b.d)
    }
}

/// The product terms formed by the last expansion of the irreducible mode in this process (step 67), up to the cut when it
/// was cut by its work bound; `last_work()` reads it.
static LAST_FORMED: AtomicU64 = AtomicU64::new(0);

/// last_work(): the product terms the last expansion of the irreducible mode formed (up to the cut when its work bound ended
/// it).
#[pyfunction]
pub fn last_work() -> u64 {
    LAST_FORMED.load(Ordering::Relaxed)
}

fn check_deadline(deadline: Option<Instant>) -> Result<(), String> {
    if let Some(d) = deadline {
        if Instant::now() > d {
            return Err("__timeout__".into());
        }
    }
    Ok(())
}

/// The monomial cap of the engine (LANDSCAPE_NATIVE_MAX_TERMS, default 8,000,000): a power of the
/// series with more monomials than this raises "__capacity__" and the caller falls back to FORM,
/// which sorts on disk.  Found on the June-2026 Sp2nf5 landscape (a theory of 25 fields and
/// expansion order 38: 10 M monomials at k = 5 and 20 GB at k = 6 before the flavor projection;
/// FORM had computed it).  Since step 56b the cap applies to one power, the whole exponential no
/// longer being held.
/// The in-memory budget of the irreducible mode (entries held by a step before it spills): LANDSCAPE_NATIVE_MAX_TERMS when
/// set, else 4,000,000 -- an entry of that mode is about 100-150 bytes, and a step holds at most about three budgets (the
/// previous power, the shards, the power being written out).
fn irrep_budget() -> usize {
    std::env::var("LANDSCAPE_NATIVE_MAX_TERMS").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|n| *n > 0).unwrap_or(4_000_000)
}

fn capacity() -> usize {
    std::env::var("LANDSCAPE_NATIVE_MAX_TERMS").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|n| *n > 0).unwrap_or(8_000_000)
}

#[inline]
fn milli_exponent(tpow: i64, spow: i64, rpow: i64) -> i128 {
    let n: i128 = 25_000_000i128 * tpow as i128 + 5_000i128 * spow as i128 + rpow as i128;
    let d: i128 = 12_500_000;
    if n >= 0 {
        (n + d / 2) / d
    } else {
        -((-n + d / 2) / d)
    }
}

/// The scaled weight 25,000,000 t + 5,000 s + r of a monomial (milli = floor((n + 6,250,000) / 12,500,000));
/// additive under products, so a product's milli is classified by one comparison of the sum with a
/// threshold (no division per product).
#[inline]
fn weight_n(t: i64, s: i64, r: i64) -> i64 {
    25_000_000 * t + 5_000 * s + r
}

/// The smallest scaled weight whose milli exceeds `milli`.
#[inline]
fn threshold_n(milli: i64) -> i64 {
    (milli + 1) * 12_500_000 - 6_250_000
}

// --------------------------------------------------------------------------- //
// packed monomials
// --------------------------------------------------------------------------- //
/// The bounds [lo, hi] of every exponent over the monomials within the t-limit, from the letters.
fn field_bounds(letters: &[Vec<i64>], t_limit: i64, max_order: i64) -> (Vec<i64>, Vec<i64>) {
    let w = letters.first().map(|e| e.len()).unwrap_or(0);
    let mut lo = vec![0i64; w];
    let mut hi = vec![0i64; w];
    if w == 0 {
        return (lo, hi);
    }
    hi[0] = t_limit.max(0);
    for j in 1..w {
        let (mut maxpos, mut minneg) = (0i64, 0i64);
        let (mut rpos, mut rneg) = (0i64, 0i64);
        let mut zero_t = false;
        for e in letters {
            let x = e[j];
            maxpos = maxpos.max(x);
            minneg = minneg.min(x);
            if e[0] > 0 {
                let prod = t_limit as i128 * x as i128;
                let d = e[0] as i128;
                if x > 0 {
                    rpos = rpos.max(((prod + d - 1) / d) as i64);
                } else if x < 0 {
                    rneg = rneg.min(-(((-prod) + d - 1) / d) as i64);
                }
            } else if x != 0 {
                zero_t = true;
            }
        }
        hi[j] = max_order.saturating_mul(maxpos);
        lo[j] = max_order.saturating_mul(minneg);
        if !zero_t {
            hi[j] = hi[j].min(rpos);
            lo[j] = lo[j].max(rneg);
        }
    }
    (lo, hi)
}

/// The fields of a layout: (word, shift, width in bits) for the bounds, stored value = v + offset,
/// the width holding every intermediate sum of two stored values of a valid product.
fn field_layout(lo: &[i64], hi: &[i64]) -> Result<(Vec<(usize, u32, u32, i64)>, usize), String> {
    let mut out = Vec::with_capacity(lo.len());
    let (mut w, mut used) = (0usize, 0u32);
    for (l, h) in lo.iter().zip(hi.iter()) {
        let off = (-*l).max(0);
        let top = (*h as i128) + 2 * (off as i128);
        if top < 0 || top >= (1i128 << 62) {
            return Err("__capacity__".into());
        }
        let bits = (128 - (top as u128).leading_zeros()).max(1);
        if used + bits > 64 {
            w += 1;
            used = 0;
        }
        out.push((w, used, bits, off));
        used += bits;
    }
    Ok((out, if lo.is_empty() { 0 } else { w + 1 }))
}

struct Packer<const N: usize> {
    fields: Vec<(usize, u32, u64, i64)>, // (word, shift, mask, offset)
    bias: [u64; N],
}

impl<const N: usize> Packer<N> {
    fn new(lo: &[i64], hi: &[i64]) -> Result<Self, String> {
        let (layout, words) = field_layout(lo, hi)?;
        if words > N {
            return Err("__capacity__".into());
        }
        let mut bias = [0u64; N];
        let mut fields = Vec::with_capacity(layout.len());
        for (w, s, bits, off) in layout {
            let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
            bias[w] |= (off as u64) << s;
            fields.push((w, s, mask, off));
        }
        Ok(Packer { fields, bias })
    }
    #[inline]
    fn pack(&self, v: &[i64]) -> [u64; N] {
        let mut k = [0u64; N];
        for ((w, s, _, off), x) in self.fields.iter().zip(v.iter()) {
            k[*w] |= ((x + off) as u64) << s;
        }
        k
    }
    #[inline]
    fn get(&self, k: &[u64; N], j: usize) -> i64 {
        let (w, s, m, off) = self.fields[j];
        ((k[w] >> s) & m) as i64 - off
    }
    /// The words with only the fields `from..` kept (the character slots of a layout).
    fn mask_from(&self, from: usize) -> [u64; N] {
        let mut m = [0u64; N];
        for (w, s, mask, _) in self.fields[from..].iter() {
            m[*w] |= mask << s;
        }
        m
    }
    fn unpack(&self, k: &[u64; N]) -> Vec<i64> {
        (0..self.fields.len()).map(|j| self.get(k, j)).collect()
    }
    #[inline]
    fn add(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        let mut o = [0u64; N];
        for i in 0..N {
            o[i] = a[i].wrapping_add(b[i]).wrapping_sub(self.bias[i]);
        }
        o
    }
}

type Map<const N: usize> = FxHashMap<[u64; N], Q>;

// --------------------------------------------------------------------------- //
// the layout of the monomials
// --------------------------------------------------------------------------- //
struct Ctx<const N: usize> {
    pf: Packer<N>,       // field-resolved [t, s, r, y, f_1..f_n, c_1..c_m]
    pl: Packer<N>,       // flavor-refined [t, s, r, y, x_1..x_rank, c_1..c_m] (a basis only)
    n_fields: usize,
    n_slots: usize,
    basis: Option<Vec<Vec<i64>>>,
    rank: usize,
    above_n: i64,
    limit_n: i64,
    exact: bool,
    fast: bool,
}

impl<const N: usize> Ctx<N> {
    #[inline]
    fn classify(&self) -> bool {
        self.basis.is_some() || self.exact
    }

    /// basis . markers of an unpacked field-resolved vector, as the flavor-refined vector.
    fn to_flavor(&self, e: &[i64]) -> Result<Vec<i64>, String> {
        let basis = self.basis.as_ref().expect("a basis");
        let mut out = Vec::with_capacity(4 + self.rank + self.n_slots);
        out.extend_from_slice(&e[..4]);
        for row in basis {
            let mut s: i64 = 0;
            for (b, x) in row.iter().zip(e[4..4 + self.n_fields].iter()) {
                s = s.checked_add(b.checked_mul(*x).ok_or("overflow")?).ok_or("overflow")?;
            }
            out.push(s);
        }
        out.extend_from_slice(&e[4 + self.n_fields..]);
        Ok(out)
    }

    fn image(&self, k: &[u64; N]) -> Result<[u64; N], String> {
        Ok(self.pl.pack(&self.to_flavor(&self.pf.unpack(k))?))
    }
}

/// itotal's letters sorted by t-power: the field-resolved ones with their flavor images, the
/// flavor-refined ones (a basis only).
struct Letters<const N: usize> {
    fr: Vec<([u64; N], Q)>,
    fr_t: Vec<i64>,
    fr_n: Vec<i64>,
    fr_img: Vec<[u64; N]>,
    fl: Vec<([u64; N], Q)>,
    fl_t: Vec<i64>,
    fl_n: Vec<i64>,
}

struct Power<const N: usize> {
    fr: Vec<([u64; N], Q)>,
    fl: Vec<([u64; N], Q)>,
}

impl<const N: usize> Power<N> {
    fn len(&self) -> usize {
        self.fr.len() + self.fl.len()
    }
}

struct Maps<const N: usize> {
    fr: Map<N>,
    fl: Map<N>,
}

impl<const N: usize> Maps<N> {
    fn new() -> Self {
        Maps { fr: FxHashMap::default(), fl: FxHashMap::default() }
    }
    fn len(&self) -> usize {
        self.fr.len() + self.fl.len()
    }
}

/// Adds c at key; true when the key was new.
#[inline]
fn insert<const N: usize>(map: &mut Map<N>, key: [u64; N], c: Q, fast: bool) -> Result<bool, String> {
    match map.get_mut(&key) {
        Some(e) => {
            *e = e.plus(&c, fast)?;
            Ok(false)
        }
        None => {
            map.insert(key, c);
            Ok(true)
        }
    }
}

fn merge_into<const N: usize>(into: &mut Map<N>, from: Map<N>, fast: bool) -> Result<(), String> {
    for (k, c) in from {
        insert(into, k, c, fast)?;
    }
    Ok(())
}

// --------------------------------------------------------------------------- //
// the powers
// --------------------------------------------------------------------------- //
/// The products of one chunk of terms of P_{k-1} (of one kind) with itotal.  `live` counts the
/// monomials held by every chunk of the step so far; past twice the cap the step stops with
/// "__capacity__" before the merge.
#[allow(clippy::too_many_arguments)]
fn products<const N: usize>(rows: &[([u64; N], Q)], rows_fr: bool, it: &Letters<N>, ctx: &Ctx<N>, t_limit: i64,
                            deadline: Option<Instant>, cap: usize, live: &AtomicUsize) -> Result<Maps<N>, String> {
    check_deadline(deadline)?;
    let mut reported = 0usize;
    let fast = ctx.fast;
    let p = if rows_fr { &ctx.pf } else { &ctx.pl };
    let mut local = Maps::new();
    for (row, (ep, cp)) in rows.iter().enumerate() {
        if row % 256 == 255 {
            check_deadline(deadline)?;
            let held = local.len();
            if held > reported {
                live.fetch_add(held - reported, Ordering::Relaxed);
                reported = held;
            }
            if live.load(Ordering::Relaxed) > 2 * cap {
                return Err("__capacity__".into());
            }
        }
        let t_ep = p.get(ep, 0);
        let bound = t_limit - t_ep;
        let n_ep = if ctx.classify() { weight_n(t_ep, p.get(ep, 1), p.get(ep, 2)) } else { 0 };
        let ep_img: Option<[u64; N]> = if rows_fr && ctx.basis.is_some() { Some(ctx.image(ep)?) } else { None };
        // x field-resolved letters
        for j in 0..it.fr.len() {
            if it.fr_t[j] > bound {
                break;
            }
            let mut n_p = 0i64;
            if ctx.classify() {
                n_p = n_ep + it.fr_n[j];
                if ctx.exact && n_p >= ctx.limit_n {
                    continue;
                }
            }
            let c = cp.times(&it.fr[j].1, fast)?;
            if rows_fr && (ctx.basis.is_none() || n_p < ctx.above_n) {
                insert(&mut local.fr, ctx.pf.add(ep, &it.fr[j].0), c, fast)?;
            } else {
                let a = if rows_fr { ep_img.as_ref().expect("an image") } else { ep };
                insert(&mut local.fl, ctx.pl.add(a, &it.fr_img[j]), c, fast)?;
            }
        }
        // x flavor-refined letters (a basis only)
        for j in 0..it.fl.len() {
            if it.fl_t[j] > bound {
                break;
            }
            if ctx.exact && n_ep + it.fl_n[j] >= ctx.limit_n {
                continue;
            }
            let c = cp.times(&it.fl[j].1, fast)?;
            let a = if rows_fr { ep_img.as_ref().expect("an image") } else { ep };
            insert(&mut local.fl, ctx.pl.add(a, &it.fl[j].0), c, fast)?;
        }
    }
    Ok(local)
}

/// P_{k-1} x itotal with the t-bound, divided by k.  The deadline is checked every 256 rows of
/// every chunk and at every merge; the chunks together past twice the cap during the step, or
/// past the cap after it, stop with "__capacity__" before the merge.
fn step<const N: usize>(prev: &Power<N>, it: &Letters<N>, ctx: &Ctx<N>, t_limit: i64, k: i128, deadline: Option<Instant>,
                        cap: usize) -> Result<Power<N>, String> {
    let total = prev.len();
    let chunk = ((total + 63) / 64).max(64);
    let fast = ctx.fast;
    let live = AtomicUsize::new(0);
    let mut parts: Vec<Maps<N>> = prev.fr
        .par_chunks(chunk)
        .map(|rows| products(rows, true, it, ctx, t_limit, deadline, cap, &live))
        .collect::<Result<Vec<_>, String>>()?;
    let parts_fl: Vec<Maps<N>> = prev.fl
        .par_chunks(chunk)
        .map(|rows| products(rows, false, it, ctx, t_limit, deadline, cap, &live))
        .collect::<Result<Vec<_>, String>>()?;
    parts.extend(parts_fl);
    if parts.iter().map(|p| p.len()).sum::<usize>() > 2 * cap {
        return Err("__capacity__".into());
    }
    let merged = parts
        .into_par_iter()
        .map(Ok)
        .reduce(|| Ok(Maps::new()), |a: Result<Maps<N>, String>, b| {
            check_deadline(deadline)?;
            let mut a = a?;
            let mut b = b?;
            if a.len() < b.len() {
                std::mem::swap(&mut a, &mut b);
            }
            merge_into(&mut a.fr, b.fr, fast)?;
            merge_into(&mut a.fl, b.fl, fast)?;
            Ok(a)
        })?;
    let mut out = Power { fr: Vec::with_capacity(merged.fr.len()), fl: Vec::with_capacity(merged.fl.len()) };
    for (src, dst) in [(merged.fr, &mut out.fr), (merged.fl, &mut out.fl)] {
        for (key, c) in src {
            if c.is_zero() {
                continue;
            }
            dst.push((key, c.over(k, fast)?));
        }
    }
    Ok(out)
}

// --------------------------------------------------------------------------- //
// the multiplicities and the rows
// --------------------------------------------------------------------------- //
/// The character product of a monomial's slot exponents as the projector's tuple, for the lookup.
fn chars_of<'py>(py: Python<'py>, cexps: &[i32], slots: &[(usize, Vec<i64>, i64)], n_nodes: usize) -> PyResult<Bound<'py, PyAny>> {
    let mut per_node: Vec<BTreeMap<Vec<i64>, BTreeMap<i64, i64>>> = (0..n_nodes).map(|_| BTreeMap::new()).collect();
    for (j, x) in cexps.iter().enumerate() {
        if *x != 0 {
            let (node, label, k) = &slots[j];
            *per_node[*node].entry(label.clone()).or_default().entry(*k).or_insert(0) += *x as i64;
        }
    }
    let mut chars: Vec<crate::NodeChars> = Vec::with_capacity(n_nodes);
    for per_label in per_node.iter() {
        let mut node: crate::NodeChars = Vec::new();
        for (label, adams) in per_label.iter() {
            if adams.values().any(|m| *m != 0) {
                node.push((label.clone(), crate::key_vec_of_map(adams).map_err(PyValueError::new_err)?));
            }
        }
        node.sort();
        chars.push(node);
    }
    crate::chars_object(py, &chars)
}

struct Mults<'a, 'py> {
    py: Python<'py>,
    lookup: &'a Bound<'py, PyAny>,
    slots: &'a [(usize, Vec<i64>, i64)],
    n_nodes: usize,
    native: Option<Projector>,
    timeout: Option<f64>,
    verify: bool,
    verified: usize,
    fallback: usize,
}

impl<'a, 'py> Mults<'a, 'py> {
    fn python(&self, cexps: &[i32]) -> PyResult<i128> {
        let obj = chars_of(self.py, cexps, self.slots, self.n_nodes)?;
        self.lookup.call1((obj,))?.extract()
    }

    fn get(&mut self, cexps: &[i32], deadline: Option<Instant>) -> PyResult<i128> {
        let native = match self.native.as_mut() {
            Some(p) => p.multiplicity(cexps, deadline).map_err(|m| engine_error(self.py, m, self.timeout))?,
            None => None,
        };
        match native {
            Some(m) => {
                if self.verify {
                    let want = self.python(cexps)?;
                    if want != m {
                        return Err(PyValueError::new_err(format!(
                            "multiplicity check: the extension gives {m}, the projector {want}, for slot exponents {cexps:?}")));
                    }
                    self.verified += 1;
                }
                Ok(m)
            }
            None => {
                self.fallback += 1;
                self.python(cexps)
            }
        }
    }
}

type Acc = FxHashMap<(i128, i64, Vec<i64>), Q>;

/// A test hook (LANDSCAPE_NATIVE_TEST_I128_BITS = b): a value beyond b bits counts as an overflow where a coefficient leaves
/// the 128-bit form (the rows' emission here, the sums of the native post pass), so that the arbitrary-precision paths
/// after an overflow can be exercised on small expansions.  Unset: every i128 is within.
pub(crate) fn within_test_bits(x: i128) -> bool {
    static BITS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let b = *BITS.get_or_init(|| {
        std::env::var("LANDSCAPE_NATIVE_TEST_I128_BITS").ok().and_then(|v| v.trim().parse::<u32>().ok()).unwrap_or(127).min(127)
    });
    b >= 127 || x.unsigned_abs() >> b == 0
}

/// The rows of one power, added into the accumulators: the terms above the limit dropped by their
/// milli exponent, coefficient x multiplicity per (milli, y power, markers or flavor).
#[allow(clippy::too_many_arguments)]
fn accumulate<const N: usize>(power: &Power<N>, ctx: &Ctx<N>, limit: i64, mults: &mut Mults<'_, '_>, acc_fr: &mut Acc, acc_fl: &mut Acc,
                              deadline: Option<Instant>, fl_floor: i128) -> PyResult<()> {
    for (fr, list) in [(true, &power.fr), (false, &power.fl)] {
        if list.is_empty() {
            continue;                   // without a basis the flavor-refined layout has no fields
        }
        let p = if fr { &ctx.pf } else { &ctx.pl };
        let cstart = if fr { 4 + ctx.n_fields } else { 4 + ctx.rank };
        let width = cstart + ctx.n_slots;
        let acc = if fr { &mut *acc_fr } else { &mut *acc_fl };
        // the distinct character products of this power, keyed by the words of their slot fields
        let slot_mask = p.mask_from(cstart);
        let mut seen: FxHashMap<[u64; N], i128> = FxHashMap::default();
        for (row, (k, c)) in list.iter().enumerate() {
            if row % 65536 == 65535 {
                check_deadline(deadline).map_err(|m| engine_error(mults.py, m, mults.timeout))?;
            }
            let milli = milli_exponent(p.get(k, 0), p.get(k, 1), p.get(k, 2));
            if milli > limit as i128 || (!fr && milli <= fl_floor) {
                continue;
            }
            let mut ck = [0u64; N];
            for i in 0..N {
                ck[i] = k[i] & slot_mask[i];
            }
            let mult: i128 = if let Some(m) = seen.get(&ck) {
                *m
            } else {
                let cexps: Vec<i32> = (cstart..width).map(|j| p.get(k, j) as i32).collect();
                let m = if cexps.iter().all(|x| *x == 0) { 1 } else { mults.get(&cexps, deadline)? };
                seen.insert(ck, m);
                m
            };
            if mult == 0 {
                continue;
            }
            let xs: Vec<i64> = (4..cstart).map(|j| p.get(k, j)).collect();
            let cm = c.times_int(mult).map_err(PyValueError::new_err)?;
            let key = (milli, p.get(k, 3), xs);
            match acc.get_mut(&key) {
                Some(x) => {
                    *x = x.plus(&cm, false).map_err(PyValueError::new_err)?;
                }
                None => {
                    acc.insert(key, cm);
                }
            }
        }
    }
    Ok(())
}

fn timeout_error(py: Python<'_>, timeout: Option<f64>) -> PyErr {
    match py.import("subprocess").and_then(|m| m.getattr("TimeoutExpired")).and_then(|cls| cls.call1(("landscape_native.expand_series", timeout.unwrap_or(0.0)))) {
        Ok(exc) => PyErr::from_value(exc),
        Err(e) => e,
    }
}

fn engine_error(py: Python<'_>, m: String, timeout: Option<f64>) -> PyErr {
    if m == "__timeout__" {
        return timeout_error(py, timeout);
    }
    if m == "__capacity__" {
        return PyValueError::new_err(format!("capacity: the expansion exceeds {} monomials (LANDSCAPE_NATIVE_MAX_TERMS)", capacity()));
    }
    if let Some(n) = m.strip_prefix("__work_bound__:") {
        return PyValueError::new_err(format!("work-bound: {n} product terms formed"));
    }
    PyValueError::new_err(m)
}

// --------------------------------------------------------------------------- //
// the engine
// --------------------------------------------------------------------------- //
struct Input<'a> {
    terms: &'a [(i128, i128, Vec<i32>)],
    t_limit: i64,
    max_order: usize,
    n_fields: usize,
    slots: &'a [(usize, Vec<i64>, i64)],
    n_nodes: usize,
    limit: i64,
    basis: Option<Vec<Vec<i64>>>,
    exact: bool,
    fast: bool,
    lo_fr: Vec<i64>,
    hi_fr: Vec<i64>,
    lo_fl: Vec<i64>,
    hi_fl: Vec<i64>,
    flavor_only: bool,
    work_bound: Option<u64>,
}

#[allow(clippy::too_many_arguments)]
fn run<'py, const N: usize>(py: Python<'py>, inp: &Input, lookup: &Bound<'py, PyAny>, groups: &Option<Vec<(String, String, usize)>>,
                       timeout: Option<f64>, deadline: Option<Instant>, monomials: bool, native_rows: bool) -> PyResult<Py<PyAny>> {
    let err = |m: String| engine_error(py, m, timeout);
    let rank = inp.basis.as_ref().map(|b| b.len()).unwrap_or(0);
    let ctx: Ctx<N> = Ctx {
        pf: Packer::new(&inp.lo_fr, &inp.hi_fr).map_err(err)?,
        pl: Packer::new(&inp.lo_fl, &inp.hi_fl).map_err(err)?,
        n_fields: inp.n_fields,
        n_slots: inp.slots.len(),
        basis: inp.basis.clone(),
        rank,
        above_n: threshold_n(6000),
        limit_n: threshold_n(inp.limit),
        exact: inp.exact,
        fast: inp.fast,
    };
    let trace = std::env::var("LANDSCAPE_NATIVE_TRACE").is_ok();
    let verify = std::env::var("LANDSCAPE_NATIVE_VERIFY_MULT").map(|v| v.trim() == "1").unwrap_or(false);
    // itotal: the terms within the limit, sorted by t-power, in two lists when a basis is set
    let mut sorted: Vec<&(i128, i128, Vec<i32>)> = inp.terms.iter().filter(|(_, _, e)| e[0] as i64 <= inp.t_limit).collect();
    sorted.sort_by_key(|(_, _, e)| e[0]);
    let mut it = Letters { fr: Vec::new(), fr_t: Vec::new(), fr_n: Vec::new(), fr_img: Vec::new(), fl: Vec::new(), fl_t: Vec::new(), fl_n: Vec::new() };
    for (n, d, e) in sorted {
        let v: Vec<i64> = e.iter().map(|x| *x as i64).collect();
        let c = Q::from_pair(*n, *d).map_err(err)?;
        let above = ctx.basis.is_some() && (inp.flavor_only || milli_exponent(v[0], v[1], v[2]) as i64 > 6000);
        let wn = weight_n(v[0], v[1], v[2]);
        if above {
            it.fl.push((ctx.pl.pack(&ctx.to_flavor(&v).map_err(err)?), c));
            it.fl_t.push(v[0]);
            it.fl_n.push(wn);
        } else {
            let key = ctx.pf.pack(&v);
            if ctx.basis.is_some() {
                it.fr_img.push(ctx.pl.pack(&ctx.to_flavor(&v).map_err(err)?));
            }
            it.fr.push((key, c));
            it.fr_t.push(v[0]);
            it.fr_n.push(wn);
        }
    }
    // the multiplicities: the extension, the Python projector as the fallback and the check
    let native = match groups {
        Some(g) => Projector::new(g, inp.slots, deadline).map_err(err)?,
        None => None,
    };
    let mut mults = Mults { py, lookup, slots: inp.slots, n_nodes: inp.n_nodes, native, timeout, verify, verified: 0, fallback: 0 };
    let mut acc_fr: Acc = FxHashMap::default();
    let mut acc_fl: Acc = FxHashMap::default();
    let mut result: Map<N> = FxHashMap::default();
    // the constant term
    let zero_fr = ctx.pf.pack(&vec![0i64; 4 + inp.n_fields + inp.slots.len()]);
    acc_fr.insert((0, 0, vec![0i64; inp.n_fields]), Q::one());
    if monomials {
        result.insert(zero_fr, Q::one());
    }
    let cap = capacity();
    let t_exp = Instant::now();
    let mut power = Power { fr: it.fr.clone(), fl: it.fl.clone() };
    for k in 1..=inp.max_order {
        if k > 1 {
            let t0 = Instant::now();
            let prev = std::mem::replace(&mut power, Power { fr: Vec::new(), fl: Vec::new() });
            let next = py.allow_threads(|| step(&prev, &it, &ctx, inp.t_limit, k as i128, deadline, cap));
            drop(prev);
            power = next.map_err(err)?;
            if trace {
                eprintln!("[series] k={k}: {} + {} terms, {:.3} s", power.fr.len(), power.fl.len(), t0.elapsed().as_secs_f64());
            }
            if power.len() == 0 {
                break;
            }
            if power.len() > cap || (monomials && result.len() + power.len() > cap) {
                return Err(err("__capacity__".into()));
            }
        }
        if monomials {
            for (key, c) in power.fr.iter() {
                insert(&mut result, *key, c.clone(), ctx.fast).map_err(err)?;
            }
        } else {
            accumulate(&power, &ctx, inp.limit, &mut mults, &mut acc_fr, &mut acc_fl, deadline, if inp.flavor_only { 6000 } else { i128::MIN })?;
        }
        check_deadline(deadline).map_err(err)?;
    }
    drop(power);
    if trace {
        eprintln!("[series] exponential {:.3} s", t_exp.elapsed().as_secs_f64());
    }
    if verify {
        eprintln!("[mult] verified {}, fallback {}", mults.verified, mults.fallback);
    }
    if monomials {
        let out = PyList::empty(py);
        let mut rows: Vec<(Vec<i64>, &Q)> = result.iter().filter(|(_, c)| !c.is_zero()).map(|(k, c)| (ctx.pf.unpack(k), c)).collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        for (e, c) in rows {
            let (n, d) = c.big_pair();
            out.append((n, d, PyList::new(py, e.iter().map(|x| *x as i32))?))?;
        }
        return Ok(out.into_any().unbind());
    }
    // integral after the projection: back to 128 bits (a row beyond raises the overflow; the caller's FORM path follows)
    let pairs = |acc: Acc| -> PyResult<Vec<((i128, i64, Vec<i64>), Coef)>> {
        let mut rows = Vec::with_capacity(acc.len());
        for (k, c) in acc {
            if !c.is_zero() {
                rows.push((k, c.pair().map_err(PyValueError::new_err)?));
            }
        }
        rows.sort();
        Ok(rows)
    };
    let rows_fr = pairs(acc_fr)?;
    let rows_fl = pairs(acc_fl)?;
    crate::post::emit_rows(py, rows_fr, rows_fl, inp.n_fields, inp.basis.clone(), native_rows)
}

// --------------------------------------------------------------------------- //
// the irreducible mode (the default engine; engine="slot" selects the slot engine, the checks' reference)
// --------------------------------------------------------------------------- //
// The character slots of a monomial are replaced by one highest weight per node: a monomial stands for
// (t, s, r, y, markers or flavor) x V_lam_1 (x) ... (x) V_lam_n, and a product with a letter decomposes
// V_lam (x) psi^k(chi_label) at once by the Brauer-Klimyk rule (`Projector::tensor`, the weights of the
// native character engine).  The rows are the entries whose highest weights are all zero, so no
// projection of character products is left.  An entry at t-power t whose highest weight on a node has
// height h (<lam, rho^vee>, the sum of its simple-root coordinates) can reach the trivial representation
// only through letters of total t-power at most t_limit - t; every irreducible component of a product of
// letters has height at most the sum of the letters' largest weight heights (the component of largest
// height is a weight of the product with its own coefficient, also for virtual characters), and the
// height of lam* equals that of lam; so an entry with h > ratio (t_limit - t), ratio the largest
// quotient of a letter's largest weight height by its t-power on that node, is dropped -- exactly, no
// row depends on it.

/// A letter of itotal: its non-character fields packed (the highest-weight fields zero) in the
/// field-resolved layout (`key`, with its flavor image `img` when a basis is set) or in the
/// flavor-refined one (`key`, above t^6 with a basis), its coefficient, t-power and scaled weight, and
/// its character as (node, local slot, exponent) factors.
struct ILetter<const N: usize> {
    key: [u64; N],
    img: [u64; N],
    c: Q,
    t: i64,
    n: i64,
    factors: Vec<(usize, usize, u32)>,
}

struct IInfo<const N: usize> {
    lam_fr: usize,
    lam_fl: usize,
    offs: Vec<usize>,
    ranks: Vec<usize>,
    mask_fr: [u64; N],
    mask_fl: [u64; N],
    ratio: Vec<Option<(i128, i128)>>,
}

#[inline]
fn lam_of<const N: usize>(p: &Packer<N>, k: &[u64; N], start: usize, info: &IInfo<N>, node: usize) -> Lam {
    let mut w = [0i32; LAM_RANK];
    for c in 0..info.ranks[node] {
        w[c] = p.get(k, start + info.offs[node] + c) as i32;
    }
    w
}

#[inline]
fn with_lams<const N: usize>(p: &Packer<N>, base: &[u64; N], mask: &[u64; N], start: usize, info: &IInfo<N>, lams: &[Lam])
                             -> Result<[u64; N], String> {
    let mut k = *base;
    for i in 0..N {
        k[i] &= !mask[i];
    }
    for (node, lam) in lams.iter().enumerate() {
        for c in 0..info.ranks[node] {
            let (w, s, m, off) = p.fields[start + info.offs[node] + c];
            let v = lam[c] as i64 + off;
            if v < 0 || (v as u64) > m {
                return Err("a highest weight outside its field".into());
            }
            k[w] |= (v as u64) << s;
        }
    }
    Ok(k)
}

type TensorMemo = FxHashMap<(usize, Lam, usize, u32), Vec<(Lam, i128)>>;

/// Below this many terms a power is multiplied in one thread (no table, no merge of partial maps).
const SERIAL_BELOW: usize = 4096;
/// The shards of a power's map in the parallel step (a power of two), and the products one batch of
/// the previous power may form before its partial maps are merged into the shards: the memory of a
/// step is the power itself plus one batch's partial maps, not the partial maps of the whole step.
const SHARDS: usize = 64;
const BATCH_PRODUCTS: usize = 2_000_000;
/// The tensor products of a step computed once for all threads only when there are at most this many (highest weight,
/// factor) pairs; above it each thread memoizes its own, emptied past MEMO_WEIGHTS highest weights (both targets lowered with
/// the budget: a batch at most half the budget, a memo at most a quarter).
const TABLE_JOBS: usize = 100_000;
const MEMO_WEIGHTS: usize = 1_000_000;

#[inline]
fn shard_of<const N: usize>(k: &[u64; N], shards: usize) -> usize {
    let mut h: u64 = 0xcbf29ce484222325;
    for w in k.iter() {
        h = (h ^ *w).wrapping_mul(0x100000001b3);
        h ^= h >> 29;
    }
    (h as usize) & (shards - 1)
}

/// The part of a product among `parts`, from its key without the highest-weight fields (`mask`): every irreducible component
/// of the product of an entry with a letter has the same part, which is known before the product is decomposed.
#[inline]
fn part_of<const N: usize>(k: &[u64; N], mask: &[u64; N], parts: u64) -> u64 {
    let mut h: u64 = 0x9e3779b97f4a7c15;
    for i in 0..N {
        h = (h ^ (k[i] & !mask[i])).wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
    }
    h % parts
}

/// The products of rows of the previous power with the letters in the irreducible mode, into `shards` maps.  `formed` gains
/// the number of product terms formed (the irreducible components kept by the height test, the work the bound of an
/// expansion counts); with `part` = (p, P) only the products of part p among P are formed.
#[allow(clippy::too_many_arguments)]
fn products_irrep<const N: usize>(rows: &[([u64; N], Q)], rows_fr: bool, it_fr: &[ILetter<N>], it_fl: &[ILetter<N>], ctx: &Ctx<N>,
                                  info: &IInfo<N>, proj: &Projector, table: &TensorMemo, t_limit: i64, deadline: Option<Instant>,
                                  cap: usize, live: &AtomicUsize, shards: usize, memo_cap: usize, stop_at: usize,
                                  formed: &AtomicU64, part: Option<(u64, u64)>)
                                  -> Result<(Vec<Maps<N>>, usize), String> {
    check_deadline(deadline)?;
    let mut n_formed = 0u64;
    let mut reported = 0usize;
    let fast = ctx.fast;
    let p = if rows_fr { &ctx.pf } else { &ctx.pl };
    let start = if rows_fr { info.lam_fr } else { info.lam_fl };
    let nn = info.ranks.len();
    let mut memo: TensorMemo = FxHashMap::default();
    let mut memo_weights = 0usize;
    let mut local: Vec<Maps<N>> = (0..shards).map(|_| Maps::new()).collect();
    let mut held_local = 0usize;
    for (row, (ep, cp)) in rows.iter().enumerate() {
        if row % 256 == 255 {
            check_deadline(deadline)?;
            let held: usize = local.iter().map(|m| m.len()).sum();
            if held > reported {
                live.fetch_add(held - reported, Ordering::Relaxed);
                reported = held;
            }
            if live.load(Ordering::Relaxed) > 2 * cap {
                return Err("__capacity__".into());
            }
        }
        let t_ep = p.get(ep, 0);
        let bound = t_limit - t_ep;
        let n_ep = if ctx.classify() { weight_n(t_ep, p.get(ep, 1), p.get(ep, 2)) } else { 0 };
        let ep_img: Option<[u64; N]> = if rows_fr && ctx.basis.is_some() { Some(ctx.image(ep)?) } else { None };
        let lams: Vec<Lam> = (0..nn).map(|i| lam_of(p, ep, start, info, i)).collect();
        for (fl_letters, letters) in [(false, it_fr), (true, it_fl)] {
            for l in letters.iter() {
                if l.t > bound {
                    break;
                }
                let n_p = if ctx.classify() { n_ep + l.n } else { 0 };
                if ctx.exact && n_p >= ctx.limit_n {
                    continue;
                }
                let to_fr = !fl_letters && rows_fr && (ctx.basis.is_none() || n_p < ctx.above_n);
                let base = if to_fr {
                    ctx.pf.add(ep, &l.key)
                } else if fl_letters {
                    let a = if rows_fr { ep_img.as_ref().expect("an image") } else { ep };
                    ctx.pl.add(a, &l.key)
                } else {
                    let a = if rows_fr { ep_img.as_ref().expect("an image") } else { ep };
                    ctx.pl.add(a, &l.img)
                };
                if let Some((p_i, parts)) = part {
                    if part_of(&base, if to_fr { &info.mask_fr } else { &info.mask_fl }, parts) != p_i {
                        continue;
                    }
                }
                // the highest weights of the product
                let mut combos: Vec<(Vec<Lam>, i128)> = vec![(lams.clone(), 1)];
                for &(node, slot, x) in l.factors.iter() {
                    let mut next = Vec::with_capacity(combos.len() * 4);
                    for (ls, m) in combos.iter() {
                        let key = (node, ls[node], slot, x);
                        let res = match table.get(&key) {
                            Some(r) => r,
                            None => {
                                if !memo.contains_key(&key) {
                                    let r = proj.tensor(node, &ls[node], slot, x).ok_or("overflow in a tensor product")?;
                                    memo_weights += r.len();
                                    if memo_weights > memo_cap {
                                        memo.clear();
                                        memo_weights = r.len();
                                    }
                                    memo.insert(key, r);
                                }
                                &memo[&key]
                            }
                        };
                        for (lam2, m2) in res.iter() {
                            let mut ls2 = ls.clone();
                            ls2[node] = *lam2;
                            next.push((ls2, m.checked_mul(*m2).ok_or("overflow in a multiplicity")?));
                        }
                    }
                    combos = next;
                }
                let budget = (bound - l.t) as i128;
                let c = cp.times(&l.c, fast)?;
                for (ls, m) in combos {
                    let mut keep = true;
                    for node in 0..nn {
                        if let Some((num, den)) = info.ratio[node] {
                            if proj.height(node, &ls[node]) * den > num * budget {
                                keep = false;
                                break;
                            }
                        }
                    }
                    if !keep {
                        continue;
                    }
                    n_formed += 1;
                    let cm = c.times_int(m)?;
                    if to_fr {
                        let k = with_lams(&ctx.pf, &base, &info.mask_fr, info.lam_fr, info, &ls)?;
                        held_local += insert(&mut local[shard_of(&k, shards)].fr, k, cm, fast)? as usize;
                    } else {
                        let k = with_lams(&ctx.pl, &base, &info.mask_fl, info.lam_fl, info, &ls)?;
                        held_local += insert(&mut local[shard_of(&k, shards)].fl, k, cm, fast)? as usize;
                    }
                }
            }
        }
        // the task's share of a round reached: the rows after this one go to the next round
        if held_local >= stop_at {
            formed.fetch_add(n_formed, Ordering::Relaxed);
            return Ok((local, row + 1));
        }
    }
    formed.fetch_add(n_formed, Ordering::Relaxed);
    Ok((local, rows.len()))
}

// --------------------------------------------------------------------------- //
// a power on disk (step 62): hash-partitioned files, no sort
// --------------------------------------------------------------------------- //
/// The spill directory of one expansion (LANDSCAPE_NATIVE_SPILL_DIR, else the system temporary
/// directory), removed when the expansion ends; directories left by ended processes are removed when
/// the next one is made.
struct SpillDir {
    path: std::path::PathBuf,
}

fn pid_alive(pid: i32) -> bool {
    // kill(pid, 0): 0 for a live process, EPERM for a live process of another user
    let r = unsafe { libc::kill(pid, 0) };
    r == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

static SPILL_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl SpillDir {
    fn new() -> Result<SpillDir, String> {
        let base = std::env::var("LANDSCAPE_NATIVE_SPILL_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| std::env::temp_dir());
        std::fs::create_dir_all(&base).map_err(|e| format!("spill directory {base:?}: {e}"))?;
        if let Ok(rd) = std::fs::read_dir(&base) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if let Some(rest) = name.strip_prefix("landscape-spill-") {
                    if let Some(pid) = rest.split('-').next().and_then(|p| p.parse::<i32>().ok()) {
                        if pid != std::process::id() as i32 && !pid_alive(pid) {
                            let _ = std::fs::remove_dir_all(e.path());
                        }
                    }
                }
            }
        }
        let path = base.join(format!("landscape-spill-{}-{}", std::process::id(), SPILL_COUNTER.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&path).map_err(|e| format!("spill directory {path:?}: {e}"))?;
        Ok(SpillDir { path })
    }
}

impl Drop for SpillDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn io_err(e: std::io::Error) -> String {
    format!("spill: {e}")
}

fn write_rec<W: std::io::Write, const N: usize>(w: &mut W, fl: bool, key: &[u64; N], c: &Q) -> Result<(), String> {
    w.write_all(&[fl as u8]).map_err(io_err)?;
    for x in key.iter() {
        w.write_all(&x.to_le_bytes()).map_err(io_err)?;
    }
    match c {
        Q::S(n, d) => {
            w.write_all(&[0u8]).map_err(io_err)?;
            w.write_all(&n.to_le_bytes()).map_err(io_err)?;
            w.write_all(&d.to_le_bytes()).map_err(io_err)?;
        }
        Q::B(b) => {
            w.write_all(&[1u8]).map_err(io_err)?;
            for v in [&b.n, &b.d] {
                let bytes = v.to_signed_bytes_le();
                w.write_all(&(bytes.len() as u32).to_le_bytes()).map_err(io_err)?;
                w.write_all(&bytes).map_err(io_err)?;
            }
        }
    }
    Ok(())
}

/// f on every record of a file, in order, one at a time (no vector of the file's records).
fn each_rec<const N: usize>(path: &std::path::Path, mut f: impl FnMut(bool, [u64; N], Q) -> Result<(), String>) -> Result<(), String> {
    use std::io::Read;
    let mut r = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(path).map_err(io_err)?);
    let mut b1 = [0u8; 1];
    let mut b8 = [0u8; 8];
    let mut b16 = [0u8; 16];
    let mut b4 = [0u8; 4];
    loop {
        match r.read_exact(&mut b1) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(io_err(e)),
        }
        let fl = b1[0] == 1;
        let mut key = [0u64; N];
        for x in key.iter_mut() {
            r.read_exact(&mut b8).map_err(io_err)?;
            *x = u64::from_le_bytes(b8);
        }
        r.read_exact(&mut b1).map_err(io_err)?;
        let c = if b1[0] == 0 {
            r.read_exact(&mut b16).map_err(io_err)?;
            let n = i128::from_le_bytes(b16);
            r.read_exact(&mut b16).map_err(io_err)?;
            Q::S(n, i128::from_le_bytes(b16))
        } else {
            let mut vals = Vec::with_capacity(2);
            for _ in 0..2 {
                r.read_exact(&mut b4).map_err(io_err)?;
                let mut bytes = vec![0u8; u32::from_le_bytes(b4) as usize];
                r.read_exact(&mut bytes).map_err(io_err)?;
                vals.push(BigInt::from_signed_bytes_le(&bytes));
            }
            let d = vals.pop().unwrap();
            let n = vals.pop().unwrap();
            Q::B(Box::new(Big { n, d }))
        };
        f(fl, key, c)?;
    }
    Ok(())
}

/// A power in memory, or on disk as one file per shard (after a spill).
enum PStore<const N: usize> {
    Mem(Power<N>),
    Disk { files: Vec<std::path::PathBuf>, len: usize },
}

impl<const N: usize> PStore<N> {
    fn len(&self) -> usize {
        match self {
            PStore::Mem(p) => p.len(),
            PStore::Disk { len, .. } => *len,
        }
    }
    fn on_disk(&self) -> bool {
        matches!(self, PStore::Disk { .. })
    }
    /// f on the whole power in memory, or on chunks of it read back from its files in turn: whole files, joined until a chunk
    /// holds at least a quarter of the budget (a power written in passes has many small files), each record read straight
    /// into the chunk.
    fn for_each_chunk(&self, mut f: impl FnMut(&Power<N>) -> Result<(), String>) -> Result<(), String> {
        match self {
            PStore::Mem(p) => f(p),
            PStore::Disk { files, .. } => {
                let target = (irrep_budget() / 4).max(1);
                let mut p = Power { fr: Vec::new(), fl: Vec::new() };
                for path in files {
                    each_rec::<N>(path, |fl, key, c| {
                        if fl { p.fl.push((key, c)) } else { p.fr.push((key, c)) }
                        Ok(())
                    })?;
                    if p.len() >= target {
                        f(&p)?;
                        p = Power { fr: Vec::new(), fl: Vec::new() };
                    }
                }
                if p.len() > 0 {
                    f(&p)?;
                }
                Ok(())
            }
        }
    }
    fn remove(&self) {
        if let PStore::Disk { files, .. } = self {
            for f in files {
                let _ = std::fs::remove_file(f);
            }
        }
    }
}

/// The distinct highest weights per node of a power's entries.
fn lams_of<const N: usize>(p: &Power<N>, ctx: &Ctx<N>, info: &IInfo<N>, into: &mut [std::collections::HashSet<Lam>]) {
    for (list, pk, start) in [(&p.fr, &ctx.pf, info.lam_fr), (&p.fl, &ctx.pl, info.lam_fl)] {
        for (k, _) in list.iter() {
            for (node, set) in into.iter_mut().enumerate() {
                set.insert(lam_of(pk, k, start, info, node));
            }
        }
    }
}

/// The bound of an expansion's work (step 67): "__work_bound__:<n>" once the product terms formed so far exceed it.  The
/// count of a whole expansion depends on the series and the engine only, so whether it exceeds the bound does not depend on
/// the threads, the batches, the budget, the spill or the passes; it is read at the end of every round and of every power.
fn check_work(formed: &AtomicU64, bound: Option<u64>) -> Result<(), String> {
    if let Some(b) = bound {
        let n = formed.load(Ordering::Relaxed);
        if n > b {
            return Err(format!("__work_bound__:{n}"));
        }
    }
    Ok(())
}

/// Whether a step whose entries exceed the budget is computed in passes over parts of the key space (step 68; the default)
/// rather than spilled: LANDSCAPE_NATIVE_PASSES unset or not 0/off/no.
fn passes_enabled() -> bool {
    !matches!(std::env::var("LANDSCAPE_NATIVE_PASSES").map(|v| v.trim().to_ascii_lowercase()).as_deref(), Ok("0") | Ok("off") | Ok("no"))
}

/// The most passes of one step: a pass forms the keys of its part only, but runs over every product of the step to find them.
const MAX_PASSES: u64 = 64;

/// What one pass over the previous power gives: its shards in memory; its shards written to files (after a spill); or, in
/// the first attempt of a step that may go to passes, the entries held and the rows done when the budget was passed.
enum Pass<const N: usize> {
    Mem(Vec<Maps<N>>),
    Files(Vec<std::path::PathBuf>, usize, Vec<std::collections::HashSet<Lam>>),
    Overflow { held: usize, rows_done: usize },
}

/// One pass over the previous power: batches of its rows, the products of part `part` (all when None) into 64 hash shards;
/// when the shards hold more than `budget` entries, the pass returns Overflow when `may_abort`, and otherwise every shard is
/// appended to its run file and emptied, each run file merged alone into a file of the power at the end.  Files are named
/// by the power `k` and `tag`.
#[allow(clippy::too_many_arguments)]
fn run_pass<const N: usize>(prev: &PStore<N>, it_fr: &[ILetter<N>], it_fl: &[ILetter<N>], ctx: &Ctx<N>, info: &IInfo<N>,
                            proj: &Projector, table: &TensorMemo, t_limit: i64, k: i128, deadline: Option<Instant>, budget: usize,
                            spill: &mut Option<SpillDir>, formed: &AtomicU64, bound: Option<u64>, part: Option<(u64, u64)>,
                            may_abort: bool, tag: &str, batch_products: usize, memo_cap: usize) -> Result<Pass<N>, String> {
    let fast = ctx.fast;
    let nn = info.ranks.len();
    let live = AtomicUsize::new(0);
    let nocap = usize::MAX / 4;
    let threads = rayon::current_num_threads().max(1);
    let mut global: Vec<Maps<N>> = (0..SHARDS).map(|_| Maps::new()).collect();
    let mut runs: Option<Vec<std::path::PathBuf>> = None;
    let mut written = 0usize;
    let mut rows_done = 0usize;
    let mut overflow: Option<usize> = None;
    let spill_global = |global: &mut Vec<Maps<N>>, runs: &Vec<std::path::PathBuf>| -> Result<usize, String> {
        let counts = global
            .par_iter_mut()
            .zip(runs.par_iter())
            .map(|(g, path)| -> Result<usize, String> {
                let f = std::fs::OpenOptions::new().create(true).append(true).open(path).map_err(io_err)?;
                let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
                let mut n = 0usize;
                for (key, c) in g.fr.drain() {
                    write_rec(&mut w, false, &key, &c)?;
                    n += 1;
                }
                for (key, c) in g.fl.drain() {
                    write_rec(&mut w, true, &key, &c)?;
                    n += 1;
                }
                std::io::Write::flush(&mut w).map_err(io_err)?;
                g.fr.shrink_to_fit();
                g.fl.shrink_to_fit();
                Ok(n)
            })
            .collect::<Result<Vec<usize>, String>>()?;
        Ok(counts.iter().sum())
    };
    // the batch adapts to the products a row actually gives (a Brauer-Klimyk product can give many
    // highest weights): the partial maps of a batch stay near BATCH_PRODUCTS entries
    let mut batch = 512usize;
    let trace_steps = std::env::var("LANDSCAPE_NATIVE_TRACE").is_ok();
    let mut max_produced = 0usize;
    let walked = prev.for_each_chunk(|chunk| {
        for (list, is_fr) in [(&chunk.fr, true), (&chunk.fl, false)] {
            let mut pos = 0usize;
            while pos < list.len() {
                let rows = &list[pos..(pos + batch).min(list.len())];
                pos += rows.len();
                let sub = ((rows.len() + threads - 1) / threads).max(64);
                // rounds: the partial maps of one round hold at most about twice batch_products entries (each task stops at
                // its share), so that a batch of the usual size ends in one round and a batch of rows that give far more products
                // than the batch's size predicted is cut; the rows a task did not reach are split again over the threads
                let mut tasks: Vec<&[([u64; N], Q)]> = rows.chunks(sub).collect();
                let mut produced_batch = 0usize;
                while !tasks.is_empty() {
                    let share = (2 * batch_products / tasks.len()).max(1);
                    let results: Vec<(Vec<Maps<N>>, usize)> = tasks
                        .par_iter()
                        .map(|r| products_irrep(r, is_fr, it_fr, it_fl, ctx, info, proj, table, t_limit, deadline, nocap, &live, SHARDS, memo_cap,
                                                share, formed, part))
                        .collect::<Result<Vec<_>, String>>()?;
                    let mut parts: Vec<Vec<Maps<N>>> = Vec::with_capacity(results.len());
                    let mut pending: Vec<&[([u64; N], Q)]> = Vec::new();
                    for (r, (maps, done)) in tasks.iter().zip(results) {
                        rows_done += done;
                        if done < r.len() {
                            pending.push(&r[done..]);
                        }
                        parts.push(maps);
                    }
                    let per = threads.div_ceil(pending.len().max(1));
                    tasks = pending.iter().flat_map(|r| r.chunks(r.len().div_ceil(per).max(1))).collect();
                    let produced: usize = parts.iter().map(|p| p.iter().map(|m| m.len()).sum::<usize>()).sum();
                    produced_batch += produced;
                    if trace_steps {
                        max_produced = max_produced.max(produced);
                    }
                    // transpose: shard s gets every thread's shard-s map
                    let mut by_shard: Vec<Vec<Maps<N>>> = (0..SHARDS).map(|_| Vec::with_capacity(parts.len())).collect();
                    for part in parts {
                        for (s_i, m) in part.into_iter().enumerate() {
                            by_shard[s_i].push(m);
                        }
                    }
                    global
                        .par_iter_mut()
                        .zip(by_shard.into_par_iter())
                        .map(|(g, ms)| -> Result<(), String> {
                            for m in ms {
                                merge_into(&mut g.fr, m.fr, fast)?;
                                merge_into(&mut g.fl, m.fl, fast)?;
                            }
                            Ok(())
                        })
                        .collect::<Result<Vec<()>, String>>()?;
                    check_deadline(deadline)?;
                    check_work(formed, bound)?;
                    let held: usize = global.iter().map(|g| g.len()).sum();
                    if held > budget {
                        if may_abort {
                            overflow = Some(held);
                            return Err("__passes__".into());
                        }
                        if runs.is_none() {
                            if spill.is_none() {
                                *spill = Some(SpillDir::new()?);
                            }
                            let dir = &spill.as_ref().unwrap().path;
                            runs = Some((0..SHARDS).map(|s_i| dir.join(format!("run-{k}-{tag}-{s_i}"))).collect());
                        }
                        written += spill_global(&mut global, runs.as_ref().unwrap())?;
                        if trace_steps {
                            eprintln!("[series]   spill at {held} held, {written} written, batch {batch} rows, largest round {max_produced} entries");
                        }
                    }
                }
                batch = (batch_products * rows.len() / produced_batch.max(1)).clamp(256, 1 << 22);
            }
        }
        Ok(())
    });
    match walked {
        Err(e) if e == "__passes__" => {
            return Ok(Pass::Overflow { held: overflow.unwrap_or(budget), rows_done });
        }
        other => other?,
    }
    let runs = match runs {
        None => return Ok(Pass::Mem(global)),
        Some(r) => r,
    };
    // spilled: the rest of the shards to their run files, then each shard merged alone into the power's file
    written += spill_global(&mut global, &runs)?;
    drop(global);
    let dir = spill.as_ref().unwrap().path.clone();
    let files: Vec<std::path::PathBuf> = (0..SHARDS).map(|s_i| dir.join(format!("power-{k}-{tag}-{s_i}"))).collect();
    // shards merged concurrently only as far as the budget allows (each holds about written / SHARDS records)
    let per = (written / SHARDS).max(1);
    let par = (budget / per).clamp(1, threads);
    if trace_steps {
        eprintln!("[series]   merge: {written} written, {per} per shard, {par} shards at once, largest round {max_produced} entries");
    }
    let mut lams = vec![std::collections::HashSet::new(); nn];
    let mut len = 0usize;
    for group in (0..SHARDS).collect::<Vec<_>>().chunks(par) {
        let results = group
            .par_iter()
            .map(|&s_i| -> Result<(usize, Vec<std::collections::HashSet<Lam>>), String> {
                let mut m = Maps::<N>::new();
                each_rec::<N>(&runs[s_i], |fl, key, c| insert(if fl { &mut m.fl } else { &mut m.fr }, key, c, fast).map(|_| ()))?;
                let _ = std::fs::remove_file(&runs[s_i]);
                let f = std::fs::File::create(&files[s_i]).map_err(io_err)?;
                let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
                let mut n = 0usize;
                let mut ls = vec![std::collections::HashSet::new(); nn];
                for (fl, map, pk, start) in [(false, m.fr, &ctx.pf, info.lam_fr), (true, m.fl, &ctx.pl, info.lam_fl)] {
                    for (key, c) in map {
                        if c.is_zero() {
                            continue;
                        }
                        write_rec(&mut w, fl, &key, &c.over(k, fast)?)?;
                        for (node, set) in ls.iter_mut().enumerate() {
                            set.insert(lam_of(pk, &key, start, info, node));
                        }
                        n += 1;
                    }
                }
                std::io::Write::flush(&mut w).map_err(io_err)?;
                Ok((n, ls))
            })
            .collect::<Result<Vec<_>, String>>()?;
        for (n, ls) in results {
            len += n;
            for (a, b) in lams.iter_mut().zip(ls.into_iter()) {
                a.extend(b);
            }
        }
        check_deadline(deadline)?;
    }
    Ok(Pass::Files(files, len, lams))
}

/// The shards of a pass held in memory, divided by k, each written to its own file of the power (`power-{k}-{tag}-{s}`, so
/// that the next step reads a pass's output a shard at a time), with their count and the distinct highest weights per node.
fn write_pass<const N: usize>(global: Vec<Maps<N>>, ctx: &Ctx<N>, info: &IInfo<N>, k: i128, dir: &std::path::Path, tag: &str)
                              -> Result<(Vec<std::path::PathBuf>, usize, Vec<std::collections::HashSet<Lam>>), String> {
    let fast = ctx.fast;
    let nn = info.ranks.len();
    let results = global
        .into_par_iter()
        .enumerate()
        .filter(|(_, g)| g.len() > 0)
        .map(|(s_i, g)| -> Result<(std::path::PathBuf, usize, Vec<std::collections::HashSet<Lam>>), String> {
            let path = dir.join(format!("power-{k}-{tag}-{s_i}"));
            let f = std::fs::File::create(&path).map_err(io_err)?;
            let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
            let mut n = 0usize;
            let mut ls = vec![std::collections::HashSet::new(); nn];
            for (fl, map, pk, start) in [(false, g.fr, &ctx.pf, info.lam_fr), (true, g.fl, &ctx.pl, info.lam_fl)] {
                for (key, c) in map {
                    if c.is_zero() {
                        continue;
                    }
                    write_rec(&mut w, fl, &key, &c.over(k, fast)?)?;
                    for (node, set) in ls.iter_mut().enumerate() {
                        set.insert(lam_of(pk, &key, start, info, node));
                    }
                    n += 1;
                }
            }
            std::io::Write::flush(&mut w).map_err(io_err)?;
            Ok((path, n, ls))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut files = Vec::with_capacity(results.len());
    let mut len = 0usize;
    let mut lams = vec![std::collections::HashSet::new(); nn];
    for (path, n, ls) in results {
        files.push(path);
        len += n;
        for (a, b) in lams.iter_mut().zip(ls.into_iter()) {
            a.extend(b);
        }
    }
    Ok((files, len, lams))
}

/// P_{k-1} x itotal / k in the irreducible mode.  A small power in one thread; otherwise one pass over the previous power
/// (`run_pass`).  When that pass holds more than `budget` entries (step 68): with passes enabled the step starts again as P
/// passes over the parts of the key space (by the part of a product's key without its highest weights, so that each key is
/// formed and written in one pass only), P sized from the entries held and the rows done when the budget was passed so that
/// a part holds about three quarters of the budget; a part that still exceeds the budget spills as before.  With passes disabled the
/// pass spills: every shard appended to its run file when the shards hold more than the budget, each run file merged alone
/// at the end.  `formed` counts the product terms of the expansion; a step that starts again does not count its first
/// attempt.  Returns the power and the distinct highest weights per node of its entries (the tensor products the next step
/// needs).
#[allow(clippy::too_many_arguments)]
fn step_irrep<const N: usize>(prev: &PStore<N>, prev_lams: &[std::collections::HashSet<Lam>], it_fr: &[ILetter<N>], it_fl: &[ILetter<N>],
                              ctx: &Ctx<N>, info: &IInfo<N>, proj: &Projector, t_limit: i64, k: i128, deadline: Option<Instant>,
                              budget: usize, spill: &mut Option<SpillDir>, formed: &AtomicU64, bound: Option<u64>)
                              -> Result<(PStore<N>, Vec<std::collections::HashSet<Lam>>), String> {
    let fast = ctx.fast;
    let nn = info.ranks.len();
    let live = AtomicUsize::new(0);
    let nocap = usize::MAX / 4;
    // the transient structures scale with the budget: a batch's partial maps and each thread's memo
    let batch_products = (budget / 2).clamp(100_000, BATCH_PRODUCTS);
    let memo_cap = (budget / 4).clamp(50_000, MEMO_WEIGHTS);
    let finish = |maps: Vec<Maps<N>>| -> Result<(PStore<N>, Vec<std::collections::HashSet<Lam>>), String> {
        let n_fr: usize = maps.iter().map(|g| g.fr.len()).sum();
        let n_fl: usize = maps.iter().map(|g| g.fl.len()).sum();
        let mut out = Power { fr: Vec::with_capacity(n_fr), fl: Vec::with_capacity(n_fl) };
        for g in maps {
            for (src, dst) in [(g.fr, &mut out.fr), (g.fl, &mut out.fl)] {
                for (key, c) in src {
                    if !c.is_zero() {
                        dst.push((key, c.over(k, fast)?));
                    }
                }
            }
        }
        let mut lams = vec![std::collections::HashSet::new(); nn];
        lams_of(&out, ctx, info, &mut lams);
        Ok((PStore::Mem(out), lams))
    };
    if let PStore::Mem(p) = prev {
        if p.len() < SERIAL_BELOW {
            // a small power: one pass in this thread, the tensor products memoized on the way
            let empty: TensorMemo = FxHashMap::default();
            let mut a = products_irrep(&p.fr, true, it_fr, it_fl, ctx, info, proj, &empty, t_limit, deadline, nocap, &live, 1, memo_cap, usize::MAX,
                                       formed, None)?.0.pop().unwrap();
            let b = products_irrep(&p.fl, false, it_fr, it_fl, ctx, info, proj, &empty, t_limit, deadline, nocap, &live, 1, memo_cap, usize::MAX,
                                   formed, None)?.0.pop().unwrap();
            check_work(formed, bound)?;
            merge_into(&mut a.fr, b.fr, fast)?;
            merge_into(&mut a.fl, b.fl, fast)?;
            return finish(vec![a]);
        }
    }
    // the tensor products the step needs, once: every highest weight of the previous power on a node
    // with every factor of a letter on that node
    let mut factors: std::collections::HashSet<(usize, usize, u32)> = std::collections::HashSet::new();
    for l in it_fr.iter().chain(it_fl.iter()) {
        for f in l.factors.iter() {
            factors.insert(*f);
        }
    }
    let n_jobs: usize = factors.iter().map(|&(node, _, _)| prev_lams[node].len()).sum();
    let table: TensorMemo = if n_jobs <= TABLE_JOBS {
        let jobs: Vec<(usize, Lam, usize, u32)> = factors.iter()
            .flat_map(|&(node, slot, x)| prev_lams[node].iter().map(move |lam| (node, *lam, slot, x)).collect::<Vec<_>>())
            .collect();
        let computed: Vec<((usize, Lam, usize, u32), Vec<(Lam, i128)>)> = jobs
            .into_par_iter()
            .map(|key| proj.tensor(key.0, &key.1, key.2, key.3).map(|r| (key, r)).ok_or_else(|| "overflow in a tensor product".to_string()))
            .collect::<Result<Vec<_>, String>>()?;
        computed.into_iter().collect()
    } else {
        FxHashMap::default()                       // many highest weights: each thread's capped memo instead
    };
    let trace = std::env::var("LANDSCAPE_NATIVE_TRACE").is_ok();
    if trace {
        eprintln!("[series] tensor table: {} products, {} highest weights in all; previous power's weights per node {:?}",
                  table.len(), table.values().map(|v| v.len()).sum::<usize>(), prev_lams.iter().map(|x| x.len()).collect::<Vec<_>>());
    }
    check_deadline(deadline)?;
    let start = formed.load(Ordering::Relaxed);
    let first = run_pass(prev, it_fr, it_fl, ctx, info, proj, &table, t_limit, k, deadline, budget, spill, formed, bound, None,
                         passes_enabled(), "w", batch_products, memo_cap)?;
    let (held, rows_done) = match first {
        Pass::Mem(global) => return finish(global),
        Pass::Files(files, len, lams) => return Ok((PStore::Disk { files, len }, lams)),
        Pass::Overflow { held, rows_done } => (held, rows_done),
    };
    // the step in passes: the first attempt's terms not counted
    formed.store(start, Ordering::Relaxed);
    let total = prev.len().max(1);
    // the distinct entries grow more slowly than the rows done (later rows mostly meet keys already held): held / f
    // over-estimates them, held / sqrt(f) was within a factor of two above them on the forced-budget sets; a part is
    // sized at three quarters of the budget
    let f = (rows_done.max(1) as f64 / total as f64).min(1.0);
    let estimate = held as f64 / f.sqrt();
    let parts = ((estimate / (budget as f64 * 0.75)).ceil() as u64).clamp(2, MAX_PASSES);
    if trace {
        eprintln!("[series]   {held} held after {rows_done} of {total} rows: about {estimate:.0} entries, {parts} passes");
    }
    if spill.is_none() {
        *spill = Some(SpillDir::new()?);
    }
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let mut len = 0usize;
    let mut lams: Vec<std::collections::HashSet<Lam>> = vec![std::collections::HashSet::new(); nn];
    for p_i in 0..parts {
        let tag = format!("p{p_i}");
        let (f, n, ls) = match run_pass(prev, it_fr, it_fl, ctx, info, proj, &table, t_limit, k, deadline, budget, spill, formed, bound,
                                        Some((p_i, parts)), false, &tag, batch_products, memo_cap)? {
            Pass::Mem(global) => {
                let dir = spill.as_ref().unwrap().path.clone();
                write_pass(global, ctx, info, k, &dir, &tag)?
            }
            Pass::Files(f, n, ls) => (f, n, ls),
            Pass::Overflow { .. } => unreachable!("a pass that may not abort"),
        };
        files.extend(f);
        len += n;
        for (a, b) in lams.iter_mut().zip(ls.into_iter()) {
            a.extend(b);
        }
        check_deadline(deadline)?;
    }
    if trace {
        eprintln!("[series]   {parts} passes: {len} entries");
    }
    Ok((PStore::Disk { files, len }, lams))
}

/// The rows of one power in the irreducible mode: the entries whose highest weights are all zero.
fn accumulate_irrep<const N: usize>(power: &Power<N>, ctx: &Ctx<N>, info: &IInfo<N>, limit: i64, acc_fr: &mut Acc, acc_fl: &mut Acc,
                                    fl_floor: i128) -> Result<(), String> {
    for (fr, list) in [(true, &power.fr), (false, &power.fl)] {
        let p = if fr { &ctx.pf } else { &ctx.pl };
        let start = if fr { info.lam_fr } else { info.lam_fl };
        let mask = if fr { &info.mask_fr } else { &info.mask_fl };
        let acc = if fr { &mut *acc_fr } else { &mut *acc_fl };
        for (k, c) in list.iter() {
            if (0..N).any(|i| k[i] & mask[i] != 0) {
                continue;
            }
            let milli = milli_exponent(p.get(k, 0), p.get(k, 1), p.get(k, 2));
            if milli > limit as i128 || (!fr && milli <= fl_floor) {
                continue;
            }
            let xs: Vec<i64> = (4..start).map(|j| p.get(k, j)).collect();
            let key = (milli, p.get(k, 3), xs);
            match acc.get_mut(&key) {
                Some(x) => {
                    *x = x.plus(c, false)?;
                }
                None => {
                    acc.insert(key, c.clone());
                }
            }
        }
    }
    Ok(())
}

struct IInput<'a> {
    inp: &'a Input<'a>,
    proj: Projector,
    letters: Vec<(Q, Vec<i64>, Vec<(usize, usize, u32)>)>, // within the limit, sorted by t: coefficient, non-character vector, factors
    lo_fr: Vec<i64>,
    hi_fr: Vec<i64>,
    lo_fl: Vec<i64>,
    hi_fl: Vec<i64>,
    offs: Vec<usize>,
    ranks: Vec<usize>,
    ratio: Vec<Option<(i128, i128)>>,
}

#[allow(clippy::too_many_arguments)]
fn run_irrep<'py, const N: usize>(py: Python<'py>, ii: &IInput, timeout: Option<f64>, deadline: Option<Instant>, native_rows: bool)
                                  -> PyResult<Py<PyAny>> {
    let err = |m: String| engine_error(py, m, timeout);
    let inp = ii.inp;
    let rank = inp.basis.as_ref().map(|b| b.len()).unwrap_or(0);
    let n_lam: usize = ii.ranks.iter().sum();
    let ctx: Ctx<N> = Ctx {
        pf: Packer::new(&ii.lo_fr, &ii.hi_fr).map_err(err)?,
        pl: Packer::new(&ii.lo_fl, &ii.hi_fl).map_err(err)?,
        n_fields: inp.n_fields,
        n_slots: n_lam,
        basis: inp.basis.clone(),
        rank,
        above_n: threshold_n(6000),
        limit_n: threshold_n(inp.limit),
        exact: inp.exact,
        fast: inp.fast,
    };
    let lam_fr = 4 + inp.n_fields;
    let lam_fl = 4 + rank;
    let info = IInfo {
        lam_fr,
        lam_fl,
        offs: ii.offs.clone(),
        ranks: ii.ranks.clone(),
        mask_fr: ctx.pf.mask_from(lam_fr),
        mask_fl: if inp.basis.is_some() { ctx.pl.mask_from(lam_fl) } else { [0u64; N] },
        ratio: ii.ratio.clone(),
    };
    let trace = std::env::var("LANDSCAPE_NATIVE_TRACE").is_ok();
    let zeros_lam = vec![0i64; n_lam];
    let mut it_fr: Vec<ILetter<N>> = Vec::new();
    let mut it_fl: Vec<ILetter<N>> = Vec::new();
    for (c, v, factors) in ii.letters.iter() {
        let mut full = v.clone();
        full.extend_from_slice(&zeros_lam);
        let above = ctx.basis.is_some() && (inp.flavor_only || milli_exponent(v[0], v[1], v[2]) as i64 > 6000);
        let wn = weight_n(v[0], v[1], v[2]);
        if above {
            let key = ctx.pl.pack(&ctx.to_flavor(&full).map_err(err)?);
            it_fl.push(ILetter { key, img: [0u64; N], c: c.clone(), t: v[0], n: wn, factors: factors.clone() });
        } else {
            let key = ctx.pf.pack(&full);
            let img = if ctx.basis.is_some() { ctx.pl.pack(&ctx.to_flavor(&full).map_err(err)?) } else { [0u64; N] };
            it_fr.push(ILetter { key, img, c: c.clone(), t: v[0], n: wn, factors: factors.clone() });
        }
    }
    let mut acc_fr: Acc = FxHashMap::default();
    let mut acc_fl: Acc = FxHashMap::default();
    acc_fr.insert((0, 0, vec![0i64; inp.n_fields]), Q::one());
    let budget = irrep_budget();
    let t_exp = Instant::now();
    let mut power: PStore<N> = if inp.flavor_only && ctx.basis.is_some() {
        // everything flavor-refined: the constant term in the flavor-refined layout
        PStore::Mem(Power { fr: Vec::new(), fl: vec![(ctx.pl.pack(&vec![0i64; 4 + rank + n_lam]), Q::one())] })
    } else {
        PStore::Mem(Power { fr: vec![(ctx.pf.pack(&vec![0i64; 4 + inp.n_fields + n_lam]), Q::one())], fl: Vec::new() })
    };
    let mut lams: Vec<std::collections::HashSet<Lam>> = (0..ii.ranks.len()).map(|_| [[0i32; LAM_RANK]].into_iter().collect()).collect();
    let mut spill: Option<SpillDir> = None;
    let formed = AtomicU64::new(0);
    LAST_FORMED.store(0, Ordering::Relaxed);
    for k in 1..=inp.max_order {
        let t0 = Instant::now();
        let next = py.allow_threads(|| step_irrep(&power, &lams, &it_fr, &it_fl, &ctx, &info, &ii.proj, inp.t_limit, k as i128, deadline,
                                                  budget, &mut spill, &formed, inp.work_bound));
        power.remove();
        LAST_FORMED.store(formed.load(Ordering::Relaxed), Ordering::Relaxed);
        let (p, l) = next.map_err(err)?;
        power = p;
        lams = l;
        if trace {
            eprintln!("[series] k={k}: {} terms{}, {:.3} s (irreducible), {} product terms formed so far", power.len(),
                      if power.on_disk() { " on disk" } else { "" }, t0.elapsed().as_secs_f64(), formed.load(Ordering::Relaxed));
        }
        if power.len() == 0 {
            break;
        }
        let fl_floor = if inp.flavor_only { 6000 } else { i128::MIN };
        power.for_each_chunk(|p| accumulate_irrep(p, &ctx, &info, inp.limit, &mut acc_fr, &mut acc_fl, fl_floor)).map_err(err)?;
        check_deadline(deadline).map_err(err)?;
    }
    power.remove();
    drop(spill);
    if trace {
        eprintln!("[series] exponential {:.3} s (irreducible)", t_exp.elapsed().as_secs_f64());
    }
    let pairs = |acc: Acc| -> PyResult<Vec<((i128, i64, Vec<i64>), Coef)>> {
        let mut rows = Vec::with_capacity(acc.len());
        for (k, c) in acc {
            if !c.is_zero() {
                rows.push((k, c.pair().map_err(PyValueError::new_err)?));
            }
        }
        rows.sort();
        Ok(rows)
    };
    if acc_fl.is_empty() && acc_fr.values().any(|c| !c.is_zero() && c.pair().is_err()) {
        // a row beyond 128 bits of a field-resolved expansion: the rows as a list with arbitrary-precision coefficients, which
        // the Python post-processing reads (with flavor-refined rows the overflow is raised and the caller asks for the
        // expansion field-resolved throughout)
        let mut rows: Vec<((i128, i64, Vec<i64>), (BigInt, BigInt))> =
            acc_fr.into_iter().filter(|(_, c)| !c.is_zero()).map(|(k, c)| (k, c.big_pair())).collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        let out = PyList::empty(py);
        for ((milli, ypow, markers), (num, den)) in rows {
            out.append((num, den, milli, ypow, pyo3::types::PyTuple::new(py, markers.iter().copied())?))?;
        }
        return Ok(out.into_any().unbind());
    }
    let rows_fr = pairs(acc_fr)?;
    let rows_fl = pairs(acc_fl)?;
    crate::post::emit_rows(py, rows_fr, rows_fl, inp.n_fields, inp.basis.clone(), native_rows)
}

/// The irreducible mode, when the projector takes every node and every letter's slot exponents are
/// non-negative; None otherwise (the slot engine follows).
#[allow(clippy::too_many_arguments)]
fn try_irrep<'py>(py: Python<'py>, inp: &Input, groups: &[(String, String, usize)], timeout: Option<f64>, deadline: Option<Instant>,
                  native_rows: bool) -> PyResult<Option<Py<PyAny>>> {
    let err = |m: String| engine_error(py, m, timeout);
    let proj = match Projector::new(groups, inp.slots, deadline).map_err(err)? {
        Some(p) => p,
        None => return Ok(None),
    };
    let nn = proj.n_nodes();
    let ranks: Vec<usize> = (0..nn).map(|i| proj.rank(i)).collect();
    let mut offs = Vec::with_capacity(nn);
    let mut acc = 0usize;
    for r in &ranks {
        offs.push(acc);
        acc += r;
    }
    let nf = 4 + inp.n_fields;
    let mut sorted: Vec<&(i128, i128, Vec<i32>)> = inp.terms.iter().filter(|(_, _, e)| e[0] as i64 <= inp.t_limit).collect();
    sorted.sort_by_key(|(_, _, e)| e[0]);
    let mut letters = Vec::with_capacity(sorted.len());
    // per node: the largest quotient (scaled height) / t over the letters charged there; None if one has t <= 0
    let mut ratio: Vec<Option<(i128, i128)>> = vec![Some((0, 1)); nn];
    let mut hsum_max: Vec<i128> = vec![0; nn];
    for (n, d, e) in sorted {
        let v: Vec<i64> = e.iter().map(|x| *x as i64).collect();
        let mut factors = Vec::new();
        for (j, x) in v[nf..].iter().enumerate() {
            if *x < 0 {
                return Ok(None);
            }
            if *x > 0 {
                let (node, local) = match proj.locate(j) {
                    Some(nl) => nl,
                    None => return Ok(None),
                };
                factors.push((node, local, *x as u32));
                let h = proj.max_height(node, local) * (*x as i128);
                hsum_max[node] = hsum_max[node].max(h);
                ratio[node] = match ratio[node] {
                    None => None,
                    Some(_) if v[0] <= 0 => None,
                    Some((num, den)) => {
                        if h * den > num * (v[0] as i128) { Some((h, v[0] as i128)) } else { Some((num, den)) }
                    }
                };
            }
        }
        letters.push((Q::from_pair(*n, *d).map_err(err)?, v[..nf].to_vec(), factors));
    }
    // bounds: the non-character fields as the slot engine's, then per node the components of a highest weight
    let nonchar: Vec<Vec<i64>> = letters.iter().map(|(_, v, _)| v.clone()).collect();
    let (mut lo_fr, mut hi_fr) = field_bounds(&nonchar, inp.t_limit, inp.max_order as i64);
    let (mut lo_fl, mut hi_fl) = match &inp.basis {
        Some(b) => {
            let images: Vec<Vec<i64>> = nonchar.iter().map(|e| {
                let mut w = e[..4].to_vec();
                for row in b {
                    w.push(row.iter().zip(e[4..].iter()).map(|(x, y)| x * y).sum());
                }
                w
            }).collect();
            field_bounds(&images, inp.t_limit, inp.max_order as i64)
        }
        None => (Vec::new(), Vec::new()),
    };
    for node in 0..nn {
        let hmax = match ratio[node] {
            Some((num, den)) => (num * inp.t_limit as i128) / den,
            None => hsum_max[node] * inp.max_order as i128,
        }
        .min(hsum_max[node] * inp.max_order as i128);
        let hv = proj.hvec(node).to_vec();
        for c in 0..ranks[node] {
            let u = (hmax / hv[c].max(1)) as i64;
            lo_fr.push(0);
            hi_fr.push(u);
            if inp.basis.is_some() {
                lo_fl.push(0);
                hi_fl.push(u);
            }
        }
    }
    let words = field_layout(&lo_fr, &hi_fr).map_err(err)?.1.max(field_layout(&lo_fl, &hi_fl).map_err(err)?.1);
    let ii = IInput { inp, proj, letters, lo_fr, hi_fr, lo_fl, hi_fl, offs, ranks, ratio };
    macro_rules! go {
        ($n:literal) => {
            run_irrep::<$n>(py, &ii, timeout, deadline, native_rows).map(Some)
        };
    }
    match words {
        0..=1 => go!(1),
        2 => go!(2),
        3 => go!(3),
        4 => go!(4),
        5 => go!(5),
        6 => go!(6),
        7 => go!(7),
        8 => go!(8),
        9..=10 => go!(10),
        11..=12 => go!(12),
        13..=16 => go!(16),
        17..=24 => go!(24),
        25..=32 => go!(32),
        _ => Ok(None),
    }
}

/// expand_series(terms, t_limit, max_order, n_fields, slots, n_nodes, t_order, lookup, timeout, monomials, native_rows,
///               basis, exact, coef64, groups): the rows as a FieldResolvedRows object when native_rows is set (the
/// native post-processing pass reads them there; with a basis, the only form); see the module.  `groups[node]` =
/// (LiE group name, type letter, rank) of the node's store, for the multiplicities in the extension; without it
/// every multiplicity comes from `lookup`.  `work_bound` (the irreducible mode): the expansion is cut with a ValueError
/// "work-bound: <n> product terms formed" once the product terms it forms exceed it (`last_work()` gives the count of the
/// last expansion either way); the slot engine does not count.  `milli_limit`: the rows' bound on the milli exponent (and
/// `exact`'s), 1000 t_order when absent -- given for a truncation order that is a fraction (a multiple of 1/500), whose
/// rows above 1000 t_order, between it and the next integer, are incomplete.
#[pyfunction]
#[pyo3(signature = (terms, t_limit, max_order, n_fields, slots, n_nodes, t_order, lookup, timeout=None, monomials=false, native_rows=false,
                    basis=None, exact=false, coef64=false, groups=None, flavor_only=false, engine="irrep", work_bound=None, milli_limit=None))]
#[allow(clippy::too_many_arguments)]
pub fn expand_series<'py>(
    py: Python<'py>,
    terms: Vec<(i128, i128, Vec<i32>)>,
    t_limit: i32,
    max_order: usize,
    n_fields: usize,
    slots: Vec<(usize, Vec<i64>, i64)>,
    n_nodes: usize,
    t_order: i64,
    lookup: Bound<'py, PyAny>,
    timeout: Option<f64>,
    monomials: bool,
    native_rows: bool,
    basis: Option<Vec<Vec<i64>>>,
    exact: bool,
    coef64: bool,
    groups: Option<Vec<(String, String, usize)>>,
    flavor_only: bool,
    engine: &str,
    work_bound: Option<u64>,
    milli_limit: Option<i64>,
) -> PyResult<Py<PyAny>> {
    if engine != "irrep" && engine != "slot" {
        return Err(PyValueError::new_err(format!("engine {engine:?}: 'irrep' (the irreducible mode) or 'slot' (the slot engine)")));
    }
    if engine == "irrep" && (monomials || groups.is_none()) {
        return Err(PyValueError::new_err("the irreducible mode needs the nodes' groups and gives rows, not the polynomial (engine='slot')"));
    }
    let deadline = timeout.map(|t| Instant::now() + std::time::Duration::from_secs_f64(t.max(0.0)));
    let width = 4 + n_fields + slots.len();
    if flavor_only && (basis.is_none() || monomials || !native_rows) {
        return Err(PyValueError::new_err("flavor_only needs a basis and native rows"));
    }
    if basis.is_some() && (monomials || !native_rows) {
        return Err(PyValueError::new_err("a flavor basis needs native rows (the rows above t^6 are flavor-refined)"));
    }
    for (_, _, e) in terms.iter() {
        if e.len() != width {
            return Err(PyValueError::new_err("exponent vector of the wrong length"));
        }
    }
    crate::lie::init_pool();
    // the bounds of the two layouts from the letters within the limit
    let letters: Vec<Vec<i64>> = terms.iter().filter(|(_, _, e)| e[0] <= t_limit).map(|(_, _, e)| e.iter().map(|x| *x as i64).collect()).collect();
    let (lo_fr, hi_fr) = field_bounds(&letters, t_limit as i64, max_order as i64);
    let (lo_fl, hi_fl) = match &basis {
        Some(b) => {
            let rank = b.len();
            let mut images = Vec::with_capacity(letters.len());
            for e in &letters {
                let mut v = Vec::with_capacity(4 + rank + slots.len());
                v.extend_from_slice(&e[..4]);
                for row in b {
                    let mut s: i64 = 0;
                    for (x, y) in row.iter().zip(e[4..4 + n_fields].iter()) {
                        s = s.checked_add(x.checked_mul(*y).ok_or_else(|| PyValueError::new_err("overflow"))?)
                            .ok_or_else(|| PyValueError::new_err("overflow"))?;
                    }
                    v.push(s);
                }
                v.extend_from_slice(&e[4 + n_fields..]);
                images.push(v);
            }
            field_bounds(&images, t_limit as i64, max_order as i64)
        }
        None => (Vec::new(), Vec::new()),
    };
    let words = field_layout(&lo_fr, &hi_fr).map_err(|m| engine_error(py, m, timeout))?.1
        .max(field_layout(&lo_fl, &hi_fl).map_err(|m| engine_error(py, m, timeout))?.1);
    let inp = Input { terms: &terms, t_limit: t_limit as i64, max_order, n_fields, slots: &slots, n_nodes, limit: milli_limit.unwrap_or(1000 * t_order), basis,
                      exact, fast: coef64, lo_fr, hi_fr, lo_fl, hi_fl, flavor_only, work_bound };
    if engine == "irrep" {
        return match try_irrep(py, &inp, groups.as_ref().unwrap(), timeout, deadline, native_rows)? {
            Some(obj) => Ok(obj),
            None => Err(PyValueError::new_err(
                "the irreducible mode cannot take this expansion: a node outside the character engine (types A-D and G, rank at most \
                 16), a negative character exponent or a monomial key wider than 32 words")),
        };
    }
    macro_rules! go {
        ($n:literal) => {
            run::<$n>(py, &inp, &lookup, &groups, timeout, deadline, monomials, native_rows)
        };
    }
    match words {
        0..=1 => go!(1),
        2 => go!(2),
        3 => go!(3),
        4 => go!(4),
        5 => go!(5),
        6 => go!(6),
        7 => go!(7),
        8 => go!(8),
        9..=10 => go!(10),
        11..=12 => go!(12),
        13..=16 => go!(16),
        17..=24 => go!(24),
        25..=32 => go!(32),
        _ => Err(engine_error(py, "__capacity__".into(), timeout)),
    }
}
